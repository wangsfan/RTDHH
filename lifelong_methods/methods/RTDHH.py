import copy
import warnings
from typing import Optional, Union, List, Dict, Callable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PIL import Image
from iirc.definitions import NO_LABEL_PLACEHOLDER
from iirc.lifelong_dataset.torch_dataset import Dataset
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.methods.base_method_RTDHH import BaseMethod
from lifelong_methods.utils import SubsetSampler, copy_freeze

import ot
import hyptorch.pmath as pmath

EPSILON = 1e-8


class Model(BaseMethod):
    """
    An  implementation of RTDHH
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(Model, self).__init__(n_cla_per_tsk, class_names_to_idx, config)
        self.epochs_per_task = config["epochs_per_task"]
        self.old_net = copy_freeze(self.net)

        if self.DP:
            self.old_net = nn.DataParallel(self.old_net, device_ids=config["device_ids"])

        self.ft_weight = config["ft_weight"]
        self.ada_ft_weight = config["ada_ft_weight"]

        self.bt_weight = config["bt_weight"]
        self.sinkhorn_reg = config['sinkhorn']
        # setup losses
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded

        Args:
        state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
        scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
        prepare_model_for_task function).
        It copies the old network and freezes it's gradients.
        """
        self.old_net = copy_freeze(self.net)
        self.old_net.eval()

    def update_C(self, Dt_train:Dataset, buffer:BufferBase):
        with torch.no_grad():
            offset1, offset2 = self._compute_offsets(self.cur_task_id)
            # construct C in hyperbolic space by the hyperbolic metric
            self._extract_class_means_with_memory(Dt_train, buffer, offset1, offset2, self.device)
            # hyperbolic distance using the prototypes of the new classes and the all hyperbolic embeddings of the old classes in buffer
            # # n_memories_per_class_C_hyp = pmath.cdist(self.old_class_vecs_hyp, self.new_class_mean_vecs_hyp, self.c) + EPSILON  # 同下
            n_memories_per_class_C_hyp = pmath.dist_matrix_A(self.old_class_vecs_hyp, self.new_class_mean_vecs_hyp, self.c) + EPSILON
            # print(f"self.old_class_vecs_hyp.shape = {self.old_class_vecs_hyp.shape}") # (n_memories_per_class, old classes number, self.latent_dim)
            # print(f"self.new_class_mean_vecs_hyp.shape = {self.new_class_mean_vecs_hyp.shape}") # (n_memories_per_class, new classes number, self.latent_dim)
            # print(f"n_memories_per_class_C_hyp.shape = {n_memories_per_class_C_hyp.shape}") # (n_memories_per_class, old classes number, new classes number)
            # The minimum value of 20 distances between the prototype of the new class and the features of the 20 replay samples of the old class
            self.C = torch.min(n_memories_per_class_C_hyp, dim=0)[0]
            print(f"self.C.shape = {self.C.shape}") # (old classes number, new classes number)
            print(f"C_before_normed = {self.C}")

            # selective min-max normalization (hard_relation guided normalization) of C
            max_values = torch.max(self.C[:, self.C_norm_indicator], dim=0)[0]
            min_values = torch.min(self.C[:, self.C_norm_indicator], dim=0)[0]
            self.C[:, self.C_norm_indicator] = (self.C[:, self.C_norm_indicator] - min_values.unsqueeze(0) + EPSILON) / (
                        max_values.unsqueeze(0) - min_values.unsqueeze(0) + EPSILON)
            self.C[:, ~self.C_norm_indicator] /= torch.max(self.C[:, ~self.C_norm_indicator], dim=0, keepdim=True)[0]
            print(f"C_normed = {self.C}")

    # FT
    def solving_ft(self):
        with torch.no_grad():
            # CKC
            self._mu1_vec_ft = torch.sum(self.all_probs_pred_all, dim=0)
            self._mu1_vec_ft = self._mu1_vec_ft / torch.sum(self._mu1_vec_ft)
            # self._mu2_vec_ft = torch.sum(self.all_probs_pred_all, dim=1)
            # self._mu2_vec_ft = self._mu2_vec_ft / torch.sum(self._mu2_vec_ft)
            # Uniform
            offset1, offset2 = self._compute_offsets(self.cur_task_id)
            # self._mu1_vec_ft = (torch.ones(offset1) / offset1 * 1.0)
            self._mu2_vec_ft = (torch.ones(offset2-offset1) / (offset2-offset1) * 1.0)
            # print("_mu1_vec_ft =\n", self._mu1_vec_ft)
            # print("_mu2_vec_ft =\n", self._mu2_vec_ft)

            T_ft = ot.sinkhorn(self._mu1_vec_ft, self._mu2_vec_ft, (self.C).double(), self.sinkhorn_reg, numItermax = 2000)
            T_ft = torch.tensor(T_ft).float().to(self.device)
            T_ft = T_ft/((torch.sum(T_ft, dim=0).unsqueeze(0)) + EPSILON)
            
            if self.DP:
                self.aux_Psi_new_by_ft = torch.mm(T_ft.T, self.net.module.fc.weight[:offset1])
                oldnorm = (torch.norm(self.net.module.fc.weight[:offset1], p=2, dim=1))
            else:
                self.aux_Psi_new_by_ft = torch.mm(T_ft.T, self.net.fc.weight[:offset1])
                oldnorm = (torch.norm(self.net.fc.weight[:offset1], p=2, dim=1))
            newnorm = (torch.norm(self.aux_Psi_new_by_ft * offset1, p=2, dim=1))
            meannew = torch.mean(newnorm)
            meanold = torch.mean(oldnorm)
            print(f"task {self.cur_task_id}_init_fc.weight:")
            print(f"mean_norm_oldweight = {meanold}")
            calibration_term = meanold / meannew
            self.aux_Psi_new_by_ft = self.aux_Psi_new_by_ft * offset1 * calibration_term
            print(f"offset1 * calibration_term = {offset1} * {calibration_term} = {offset1*calibration_term}")
            print(f"mean_norm_newweight = {torch.mean(torch.norm(self.aux_Psi_new_by_ft, p=2, dim=1))}")

    # BT
    def solving_bt(self):
        with torch.no_grad():
            # CKC
            self._nu1_vec_bt = torch.sum(self.all_probs_pred_all, dim=0)
            self._nu1_vec_bt = self._nu1_vec_bt / torch.sum(self._nu1_vec_bt)
            self._nu2_vec_bt = torch.sum(self.all_probs_pred_all, dim=1)
            self._nu2_vec_bt = self._nu2_vec_bt / torch.sum(self._nu2_vec_bt)
            # Uniform
            # offset1, offset2 = self._compute_offsets(self.cur_task_id)
            # self._nu1_vec_bt = (torch.ones(offset1) / offset1 * 1.0)
            # self._nu2_vec_bt = (torch.ones(offset2 - offset1) / (offset2 - offset1) * 1.0)
            # print("_nu1_vec_bt =\n", self._nu1_vec_bt)
            # print("_nu2_vec_bt =\n", self._nu2_vec_bt)

            T_bt = ot.sinkhorn(self._nu2_vec_bt, self._nu1_vec_bt, (self.C).T.double(), self.sinkhorn_reg, numItermax=2000)
            T_bt = torch.tensor(T_bt).float().to(self.device)
            # print(f"T_bt =\n{T_bt}")
            max_values = torch.max(T_bt[:, self.bt_norm_indicator], dim=0)[0]
            min_values = torch.min(T_bt[:, self.bt_norm_indicator], dim=0)[0]
            T_bt[:, self.bt_norm_indicator] = (T_bt[:, self.bt_norm_indicator] - min_values.unsqueeze(0)) / (
                                                      max_values.unsqueeze(0) - min_values.unsqueeze(0))
            # print(f"T_bt_norm =\n{T_bt}")
            self.T_bt = T_bt

    def _preprocess_target(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Replaces the labels on the older classes with the distillation targets produced by the old network"""
        offset1, offset2 = self._compute_offsets(self.cur_task_id)
        y = y.clone()
        old_features = None
        if self.cur_task_id > 0:
            distill_model_output, old_features = self.old_net(x)
            distill_model_output = distill_model_output.detach()
            distill_model_output = torch.sigmoid(distill_model_output / self.temperature)
            y[:, :offset1] = distill_model_output[:, :offset1]
        return y, old_features

    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
        train: bool = True, epoch: int = 0) -> Tuple[torch.Tensor, float]:
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
        x (torch.Tensor): The batch of images
        y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
        in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
        train (bool): Whether this is training or validation/test
        epoch(int): config['task_epoch'], the current epoch

        Returns:
        Tuple[torch.Tensor, float]:
        predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        loss (float): the value of the loss
        """

        offset_1, offset_2 = self._compute_offsets(self.cur_task_id)
        target, old_features = self._preprocess_target(x, y) # for task0, old_features is None
        assert target.shape[1] == offset_2
        logits, features = self.forward_net(x)
        output = logits[:, :offset_2]
        assert torch.equal(y[:, offset_1:offset_2], target[:, offset_1:offset_2])

        lambd = offset_1 / offset_2
        cls_loss = self.bce(output[:, offset_1:offset_2] / self.temperature, y[:, offset_1:offset_2])
        ft_loss = 0
        if self.ada_ft_weight:
            self.ft_weight = 1-lambd
        distill_bt_loss = 0

        if self.cur_task_id > 0:
            # ft_loss
            # if (self.aux_Psi_new_by_ft != None) and (epoch < 0.5*self.epochs_per_task):
            current_logits_new = logits[:, offset_1:offset_2]
            # print("\ncurrent_logits_new[0]:\n", torch.sigmoid(current_logits_new / self.temperature)[0])
            # print("label_new[0]:\n", y[0, offset_1:offset_2])
            aux_logits_new_by_ft = F.linear(features, self.aux_Psi_new_by_ft)
            aux_logits_new_by_ft = torch.sigmoid(aux_logits_new_by_ft / self.temperature)
            # print("aux_logits_new_by_ft[0]:\n", aux_logits_new_by_ft[0])
            assert current_logits_new.shape == aux_logits_new_by_ft.shape
            ft_loss = self.bce(current_logits_new / self.temperature, aux_logits_new_by_ft)

            distill_model_output = target[:, :offset_1]
            # bt_loss
            # only compute on the new data(~buffer)
            if (self.T_bt != None) and (epoch >= 0.5 * self.epochs_per_task):
                if not train:  # task_in_valid
                    assert in_buffer is None
                    in_buffer = torch.zeros(len(x)).to(torch.bool)
                if torch.any(~in_buffer):
                    label_new_Dt = y[~in_buffer, offset_1:offset_2]
                    # print("label_new_Dt[0]:\n", label_new_Dt[0])

                    distill_model_output_Dt = distill_model_output[~in_buffer]
                    # print("distill_model_output_Dt[0]:\n", distill_model_output_Dt[0])

                    aux_logits_old_by_bt_Dt = torch.mm(label_new_Dt, self.T_bt)
                    # print("aux_logits_old_by_bt_Dt[0]:\n", aux_logits_old_by_bt_Dt[0])
                    distill_norm = (torch.norm(distill_model_output_Dt, p=2, dim=1))
                    bt_norm = (torch.norm(aux_logits_old_by_bt_Dt, p=2, dim=1))
                    calibration_vec = distill_norm / bt_norm
                    # print(f"distill_norm[0] : {distill_norm[0]}\nbt_norm[0] : {bt_norm[0]}\ncalibration_vec[0] : {calibration_vec[0]}")
                    aux_logits_old_by_bt_Dt = aux_logits_old_by_bt_Dt * calibration_vec.unsqueeze(1)
                    # print("aux_logits_old_by_bt_Dt[0](calibrated):\n", aux_logits_old_by_bt_Dt[0])
                    aux_logits_old_by_bt_Dt = torch.where(aux_logits_old_by_bt_Dt > 1,
                                                          torch.tensor(1.0).to(self.device), aux_logits_old_by_bt_Dt)
                    # print("aux_logits_old_by_bt_Dt[0](calibrated + <=1):\n", aux_logits_old_by_bt_Dt[0])

                    aux_logits_old_by_distill_bt_Dt = (1-self.bt_weight) * distill_model_output_Dt + self.bt_weight * aux_logits_old_by_bt_Dt
                    # print(f"aux_logits_old_by_distill_bt_Dt = {1 - self.bt_weight} * distill_model_output_Dt + {self.bt_weight} * new_logits_old_bt_Dt")
                    # print("aux_logits_old_by_distill_bt_Dt[0]:\n", aux_logits_old_by_distill_bt_Dt[0])

                    ########################## If use get_sibling_main.py to get the self.cur_sibling_dict,
                    ########################## conduct the similar Hierarchical Relation Alignment(HRA) to method UAHR on the aux_logits_old_by_distill_bt_Dt
                    if self.cur_task_id > 1 and len(self.cur_sibling_dict) > 0:
                        bs_Dt, _ = (y[~in_buffer]).size()
                        is_sibling = torch.zeros((bs_Dt, len(list(self.cur_sibling_dict.values())[0])))
                        for i in range(bs_Dt):
                            # transform the label of the current sample to tuple
                            label_tuple_i = tuple(y[~in_buffer][i].tolist())
                            value = self.cur_sibling_dict.get(label_tuple_i)
                            if value is None:
                                raise KeyError(f"Label {label_tuple_i} not found in cur_sibling_dict")
                            is_sibling[i] = torch.from_numpy(value)
                        is_sibling = is_sibling.bool()
                        aux_logits_old_by_distill_bt_Dt_ori = copy.deepcopy(aux_logits_old_by_distill_bt_Dt)
                        print("aux_logits_old_by_distill_bt_Dt_ori[0]:\n", aux_logits_old_by_distill_bt_Dt_ori[0])
                        aux_logits_old_by_distill_bt_Dt[is_sibling] = torch.where(
                            aux_logits_old_by_distill_bt_Dt[is_sibling] >= 0.5,
                            aux_logits_old_by_distill_bt_Dt[is_sibling] - 0.4,
                            aux_logits_old_by_distill_bt_Dt[is_sibling])

                        if ~torch.equal(aux_logits_old_by_distill_bt_Dt_ori, aux_logits_old_by_distill_bt_Dt):
                            print("\ncur_sibling_dict functions!")
                            print("aux_logits_old_by_distill_bt_Dt_sibling[0]:\n", aux_logits_old_by_distill_bt_Dt[0])
                    else:
                        print("\ncur_sibling_dict = {}\n")
                    ##########################

                    distill_model_output[~in_buffer] = aux_logits_old_by_distill_bt_Dt

            distill_bt_loss = self.bce(output[:, :offset_1] / self.temperature, distill_model_output)

            print(f"{lambd} * distill_bt_loss = {lambd * distill_bt_loss}\n"
                  f"{1-lambd} * {self.ft_weight} * ft_loss = {(1-lambd) * self.ft_weight * ft_loss}")

        loss = lambd * distill_bt_loss + (1-lambd) * cls_loss + (1-lambd) * self.ft_weight * ft_loss
        print(f"{1-lambd} * cls_loss = {(1-lambd) * cls_loss}\nloss = {loss}")


        if train:
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        predictions = output.ge(0.0)

        return predictions, loss.item()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions

        Args:
        x (torch.Tensor): The batch of images

        Returns:
        torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        num_seen_classes = len(self.seen_classes)
        output, _ = self.forward_net(x)
        output = output[:, :num_seen_classes]
        predictions = output.ge(0.0)
        return predictions

    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        """
        pass

    def consolidate_task_knowledge(self,**kwargs) -> None:
        """Takes place after training on each task"""


class Buffer(BufferBase):
    def __init__(self,
         config: Dict,
         buffer_dir: Optional[str] = None,
         map_size: int = 1e9,
         essential_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
         augmentation_transforms_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None):
        super(Buffer, self).__init__(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)
        self.device = config['device']

    def _reduce_exemplar_set(self, **kwargs) -> None:
        """remove extra exemplars from the buffer"""
        for label in self.seen_classes:
            if len(self.mem_class_x[label]) > self.n_mems_per_cla:
                n = len(self.mem_class_x[label]) - self.n_mems_per_cla
                self.remove_samples(label, n)

    def _construct_exemplar_set(self, task_data: Dataset, dist_args: Optional[dict] = None,
                        model: torch.nn.Module = None, batch_size=1, **kwargs):
        """
        Update the buffer with the new task samples using herding

        Args:
        task_data (Dataset): The new task data
        dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
        rank of the device) (default: None)
        model (BaseMethod): The current method object to calculate the latent variables
        batch_size (int): The minibatch size
        """
        distributed = dist_args is not None
        if distributed:
            device = torch.device(f"cuda:{dist_args['gpu']}")
            rank = dist_args['rank']
        else:
            device = self.device
            rank = 0
            new_class_labels = task_data.cur_task
            model.eval()

        with task_data.disable_augmentations(): # disable augmentations then enable them (if they were already enabled)
            with torch.no_grad():
                for class_label in new_class_labels:
                    class_data_indices = task_data.get_image_indices_by_cla(class_label, self.max_mems_pool_size)
                    if distributed:
                        device = torch.device(f"cuda:{dist_args['gpu']}")
                        class_data_indices_to_broadcast = torch.from_numpy(class_data_indices).to(device)
                        dist.broadcast(class_data_indices_to_broadcast, 0)
                        class_data_indices = class_data_indices_to_broadcast.cpu().numpy()
                    sampler = SubsetSampler(class_data_indices)
                    class_loader = DataLoader(task_data, batch_size=batch_size, sampler=sampler)
                    latent_vectors = []
                    for minibatch in class_loader:
                        images = minibatch[0].to(device)
                        output, out_latent = model.forward_net(images)
                        out_latent = out_latent.detach()
                        out_latent = F.normalize(out_latent, p=2, dim=-1)
                        latent_vectors.append(out_latent)
                    latent_vectors = torch.cat(latent_vectors, dim=0)
                    class_mean = torch.mean(latent_vectors, dim=0)

                    chosen_exemplars_ind = []
                    exemplars_mean = torch.zeros_like(class_mean)
                    while len(chosen_exemplars_ind) < min(self.n_mems_per_cla, len(class_data_indices)):
                        potential_exemplars_mean = (exemplars_mean.unsqueeze(0) * len(chosen_exemplars_ind) + latent_vectors) \
                                                   / (len(chosen_exemplars_ind) + 1)
                        distance = (class_mean.unsqueeze(0) - potential_exemplars_mean).norm(dim=-1)
                        shuffled_index = torch.argmin(distance).item()
                        exemplars_mean = potential_exemplars_mean[shuffled_index, :].clone()
                        exemplar_index = class_data_indices[shuffled_index]
                        chosen_exemplars_ind.append(exemplar_index)
                        latent_vectors[shuffled_index, :] = float("inf")

                    for image_index in chosen_exemplars_ind:
                        image, label1, label2 = task_data.get_item(image_index)
                        if label2 != NO_LABEL_PLACEHOLDER:
                            warnings.warn(f"Sample is being added to the buffer with labels {label1} and {label2}")
                        self.add_sample(class_label, image, (label1, label2), rank=rank)
