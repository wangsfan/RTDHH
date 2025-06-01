import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Iterator, Tuple

import numpy as np
import torch
import torch.nn as nn
from iirc.lifelong_dataset.torch_dataset import Dataset
from iirc.utils.utils import print_msg
from lifelong_methods.models.resnet import ResNet
from lifelong_methods.models.resnetcifar import ResNetCIFAR
from lifelong_methods.utils import get_optimizer, transform_labels_names_to_vector
from lifelong_methods.buffer.buffer import BufferBase
from lifelong_methods.utils import l_distance, SubsetSampler, copy_freeze
from torch.utils.data import DataLoader


class BaseMethod(ABC, nn.Module):
    """
    A base model for all the lifelong learning methods to inherit, which contains all common functionality

    Args:
        n_cla_per_tsk (Union[np.ndarray, List[int]]): An integer numpy array including the number of classes per each task.
        class_names_to_idx (Dict[str, int]): The index of each class name
        config (Dict): A dictionary that has the following key value pairs:
            temperature (float): the temperature to divide the logits by
            memory_strength (float): The weight to add for the samples from the buffer when computing the loss
                (not implemented yet)
            n_layers (int): The number of layers for the network used (not all values are allowed depending on the
                architecture)
            dataset (str): The name of the dataset (for ex: iirc_cifar100)
            optimizer (str): The type of optimizer ("momentum" or "adam")
            lr (float): The initial learning rate
            lr_schedule (Optional[list[int]]): The epochs for which the learning rate changes
            lr_gamma (float): The multiplier multiplied by the learning rate at the epochs specified in lr_schedule
            reduce_lr_on_plateau (bool): reduce learning rate on plateau
            weight_decay (float): the weight decay multiplier
    """

    def __init__(self, n_cla_per_tsk: Union[np.ndarray, List[int]], class_names_to_idx: Dict[str, int], config: Dict):
        super(BaseMethod, self).__init__()
        self.n_cla_per_tsk = n_cla_per_tsk
        self.increment = n_cla_per_tsk[1]
        self.class_names_to_idx = class_names_to_idx
        self.num_classes = int(sum(self.n_cla_per_tsk))
        self.cur_task_id = 0  # The current training task id
        self.temperature = config["temperature"]
        self.memory_strength = config["memory_strength"]
        self.n_layers = config["n_layers"]
        self.seen_classes = []
        self.batch_size = config["batch_size"]
        self.device = config['device']
        self.dataset = config["dataset"]
        self.metadata_root = config["metadata_root"]

        self.num_workers = config["num_workers"]
        self.threshold = config["threshold"]
        self.logging_path = config["logging_path"]

        # setup network
        if 'cifar' in config["dataset"] or config["dataset"] == "iirc_imagenet_subset":
            self.net = ResNetCIFAR(num_classes=self.num_classes, num_layers=self.n_layers, relu_last_hidden=False)
            self.latent_dim = self.net.convnet.latent_dim
        elif 'imagenet' in config["dataset"]:
            self.net = ResNet(num_classes=self.num_classes, num_layers=self.n_layers)
            self.latent_dim = self.net.latent_dim
        else:
            raise ValueError(f"Unsupported dataset {config['dataset']}")

        # setup optimizer
        self.optimizer_type = config["optimizer"]
        self.lr = config["lr"]
        self.lr_gamma = config["lr_gamma"]
        self.lr_schedule = config["lr_schedule"]
        self.reduce_lr_on_plateau = config["reduce_lr_on_plateau"]
        self.weight_decay = config["weight_decay"]
        self.opt, self.scheduler = get_optimizer(
            model_parameters=self.net.parameters(), optimizer_type=self.optimizer_type, lr=self.lr,
            lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            weight_decay=self.weight_decay
        )

        # The model variables that are not in the state_dicts of the model and that need to be saved
        self.method_variables = ['n_cla_per_tsk', 'num_classes', 'cur_task_id', 'temperature', 'memory_strength',
                                 'n_layers', 'seen_classes', 'latent_dim', 'optimizer_type', 'lr', 'lr_gamma',
                                 'lr_schedule', 'weight_decay']
        for variable in self.method_variables:
            assert variable in self.__dict__.keys()

    def method_state_dict(self) -> Dict[str, Dict]:
        """
        This function returns a dict that contains the state dictionaries of this method (including the model, the
            optimizer, the scheduler, as well as the values of the variables whose names are inside the
            self.method_variables), so that they can be used for checkpointing.

        Returns:
            Dict: a dictionary with the state dictionaries of this method, the optimizer, the scheduler, and the values
            of the variables whose names are inside the self.method_variables
        """
        state_dicts = {}
        state_dicts['model_state_dict'] = self.state_dict()
        state_dicts['optimizer_state_dict'] = self.opt.state_dict()
        state_dicts['scheduler_state_dict'] = self.scheduler.state_dict()
        state_dicts['method_variables'] = {key: self.__dict__[key] for key in self.method_variables}
        return state_dicts

    def load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This function loads the state dicts of the various parts of this method (along with the variables in
            self.method_variables)

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        self._load_method_state_dict(state_dicts)
        keys = {'model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict', 'method_variables'}
        assert keys.issubset(state_dicts.keys())
        assert set(self.method_variables) == set(state_dicts['method_variables'].keys())
        self.load_state_dict(state_dicts['model_state_dict'])
        self.opt.load_state_dict(state_dicts['optimizer_state_dict'])
        self.scheduler.load_state_dict(state_dicts['scheduler_state_dict'])
        for key, value in state_dicts['method_variables'].items():
            self.__dict__[key] = value

    @abstractmethod
    def _load_method_state_dict(self, state_dicts: Dict[str, Dict]) -> None:
        """
        This is where anything model specific needs to be done before the state_dicts are loaded
        This function needs to be defined in the inheriting method class

        Args:
            state_dicts (Dict[str, Dict]): a dictionary with the state dictionaries of this method, the optimizer, the
            scheduler, and the values of the variables whose names are inside the self.method_variables
        """
        pass

    def reset_optimizer_and_scheduler(self, optimizable_parameters: Optional[
        Iterator[nn.parameter.Parameter]] = None) -> None:
        """
        Reset the optimizer and scheduler after a task is done (with the option to specify which parameters to optimize

        Args:
            optimizable_parameters (Optional[Iterator[nn.parameter.Parameter]]: specify the parameters that should be
                optimized, in case some parameters needs to be frozen (default: None)
        """
        print_msg(f"resetting scheduler and optimizer, learning rate = {self.lr}")
        if optimizable_parameters is None:
            optimizable_parameters = self.net.parameters()
        self.opt, self.scheduler = get_optimizer(
            model_parameters=optimizable_parameters, optimizer_type=self.optimizer_type, lr=self.lr,
            lr_gamma=self.lr_gamma, lr_schedule=self.lr_schedule, reduce_lr_on_plateau=self.reduce_lr_on_plateau,
            weight_decay=self.weight_decay
        )

    def get_last_lr(self) -> List[float]:
        """Get the current learning rate"""
        lr = [group['lr'] for group in self.opt.param_groups]
        return lr

    def step_scheduler(self, val_metric: Optional = None) -> None:
        """
        Take a step with the scheduler (should be called after each epoch)

        Args:
            val_metric (Optional): a metric to compare in case of reducing the learning rate on plateau (default: None)
        """
        cur_lr = self.get_last_lr()
        if self.reduce_lr_on_plateau:
            assert val_metric is not None
            self.scheduler.step(val_metric)
        else:
            self.scheduler.step()
        new_lr = self.get_last_lr()
        if cur_lr != new_lr:
            print_msg(f"learning rate changes to {new_lr}")


    # # 无论新类旧类都用了所有样本来计算类中心，应该不对
    # def _extract_class_means(self, data_manager, low, high):
    #     self._ot_prototype_means = np.zeros((data_manager.get_total_classnum(), self._network.feature_dim))
    #     with torch.no_grad():
    #         for class_idx in range(low, high):
    #             data, targets, idx_dataset = data_manager.get_dataset(np.arange(class_idx, class_idx + 1),
    #                                                                   source='train',
    #                                                                   mode='test', ret_data=True)
    #             idx_loader = DataLoader(idx_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    #             vectors, _ = self._extract_vectors(idx_loader)
    #             vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
    #             class_mean = np.mean(vectors, axis=0)
    #             class_mean = class_mean / (np.linalg.norm(class_mean))
    #             self._ot_prototype_means[class_idx, :] = class_mean
    #     self._network.train()

    # def _update_means(self, buffer: BufferBase, device: torch.device, batch_size: int) -> torch.Tensor:
    #     nd = self.latent_dim
    #     offset1, offset2 = self._compute_offsets(self.cur_task_id)
    #     class_means = torch.ones(offset2, nd, device=device) * float('inf')
    #
    #     with buffer.disable_augmentations():
    #         with torch.no_grad():
    #             for class_label in buffer.mem_class_x.keys():
    #                 class_images_indices = buffer.get_image_indices_by_class(class_label)
    #                 sampler = SubsetSampler(class_images_indices)
    #                 class_loader = DataLoader(buffer, batch_size=batch_size, sampler=sampler)
    #                 latent_vectors = []
    #                 for minibatch in class_loader:
    #                     images = minibatch[0].to(device)
    #                     _, latent_vecs = self.forward_net(images)
    #                     latent_vecs = latent_vecs.detach()
    #                     _, flipped_latent_vecs = self.forward_net(images.flip((-1)))
    #                     flipped_latent_vecs = flipped_latent_vecs.detach()
    #                     latent_vecs = torch.cat((latent_vecs, flipped_latent_vecs), dim=0)
    #                     latent_vecs /= latent_vecs.norm(p=2, dim=-1, keepdim=True)
    #                     latent_vectors.append(latent_vecs)
    #                 latent_vectors = torch.cat(latent_vectors, dim=0)
    #                 class_mean = latent_vectors.mean(0)
    #                 class_means[class_label, :] = class_mean / class_mean.norm(2)
    #     return class_means

    def _extract_class_means_with_memory(self, task_data:Dataset, buffer:BufferBase, low:int, high:int, device: torch.device):
        idx_to_class_names = dict((y, x) for x, y in self.class_names_to_idx.items())
        self._ot_prototype_means = torch.zeros(self.num_classes, self.latent_dim)
        # with buffer.disable_augmentations(), task_data.disable_augmentations():
        #     with torch.no_grad():
        for class_idx in range(0, high):
            class_label = idx_to_class_names[class_idx]

            if class_idx < low:
                class_images_indices = buffer.get_image_indices_by_class(class_label)
                sampler = SubsetSampler(class_images_indices)
                class_loader = DataLoader(buffer, batch_size=self.batch_size, sampler=sampler)
            else:
                class_images_indices = task_data.get_image_indices_by_cla(class_label)
                sampler = SubsetSampler(class_images_indices)
                class_loader = DataLoader(task_data, batch_size=self.batch_size, sampler=sampler)

            latent_vectors = []
            for minibatch in class_loader:
                images = minibatch[0].to(device)

                _, latent_vecs = self.forward_net(images)
                latent_vecs = latent_vecs.detach()

                _, flipped_latent_vecs = self.forward_net(images.flip((-1)))
                flipped_latent_vecs = flipped_latent_vecs.detach()

                latent_vecs = torch.cat((latent_vecs, flipped_latent_vecs), dim=0)
                latent_vecs /= latent_vecs.norm(p=2, dim=-1, keepdim=True)

                latent_vectors.append(latent_vecs)

            latent_vectors = torch.cat(latent_vectors, dim=0)
            class_mean = latent_vectors.mean(0)
            self._ot_prototype_means[class_idx, :] = class_mean / class_mean.norm(2)
        # self.net.train()

    def _compute_offsets(self, task) -> Tuple[int, int]:
        offset1 = int(sum(self.n_cla_per_tsk[:task]))
        offset2 = int(sum(self.n_cla_per_tsk[:task + 1]))
        return offset1, offset2

    # 获取真正的子类与父类索引的对应字典（层次字典）
    def get_true_heirarchy_dict(self):
        if 'cifar' in self.dataset:
            heirarchy_path = os.path.join(self.metadata_root, "iirc_cifar100_hierarchy.json")
        elif 'imagenet' in self.dataset:
            heirarchy_path = os.path.join(self.metadata_root, "iirc_imagenet_hierarchy_wnids.json")
        else:
            raise ValueError(f"Unsupported dataset {self.dataset}")

        with open(heirarchy_path) as f:
            class_hierarchy = json.load(f)

        superclass_to_subclasses = class_hierarchy['super_classes']

        # 对于cifar和imagenet_full, selected_supcls_to_subclses == superclass_to_subclasses;
        # 而对于imagenet_lite/subset, 读取层次json文件所得的superclass_to_subclass中包含了数据集(class_names_to_idx)没有的类，所以要做select
        selected_supcls_to_subclses = {
            superclass: [
                subclass_i for subclass_i in superclass_to_subclasses[superclass]
                if subclass_i in self.class_names_to_idx
            ]
            for superclass in superclass_to_subclasses
            if superclass in self.class_names_to_idx
        }

        selected_subclses_to_supcls = {subclass_i: superclass for superclass, subclasses in
                                    selected_supcls_to_subclses.items()
                                    for subclass_i in subclasses}

        sub_idx_to_super_idx = {self.class_names_to_idx[subclass]: self.class_names_to_idx[superclass] for
                                subclass, superclass in selected_subclses_to_supcls.items()}
        return selected_subclses_to_supcls, sub_idx_to_super_idx


    def task_eval_superclass_in_training(self, Dt_train_loader:DataLoader):
            # assert dataset.complete_information_mode is True

            idx_to_class_names = dict((y, x) for x, y in self.class_names_to_idx.items())
            offset1, offset2 = self._compute_offsets(self.cur_task_id)
            _, sub_idx_to_sup_idx = self.get_true_heirarchy_dict()

            self.net.eval()
            with torch.no_grad():
                # 父类预测字典
                # 在遍历iters时得到，方便后续对各新类预测的父类情况进行汇总
                super_pred_dict = {}

                for minibatch in Dt_train_loader:
                    labels_names = list(zip(minibatch[1], minibatch[2]))
                    # labels的维度其实为num_seen_classes数，即截至当前任务的类别数量
                    labels = transform_labels_names_to_vector(
                        labels_names, offset2, self.class_names_to_idx
                    )
                    images = minibatch[0].to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)

                    predictions= self.forward(images)
                    # super_predictions
                    # 其令outputs>0对应的旧类类别为true（为true就是将该旧类暂定为该样本可能的超类），可能不止一个 >0！
                    super_predictions = predictions[:, :offset1]

                    # 遍历所有新类样本，统计父类预测字典super_pred_dict
                    # key为各新类单粒度标签向量（的字符串形式，即’[0, 0, 0, 1]‘），值为该新类所有样本的super_preditions构成的二维数组
                    # 因为num_class_per_task=5,所以共有5个键值对；各键值对值的行数为该键值对对应的新类的样本数
                    for index in range(len(super_predictions)):
                        complete_label = labels[index]
                        complete_label = complete_label.data.cpu().numpy()
                        value = np.array2string(complete_label.astype(np.int32))
                        if not value in super_pred_dict.keys():
                            super_pred_dict[value] = []

                        super_pred_dict[value].append(super_predictions[index].data.cpu().detach().numpy())

                self.verified_super_key = []
                # all_pos_super_key = []
                self.all_pos_super_key = [None] * self.increment
                # all_keys = []
                self.all_keys = [None] * self.increment
                # all_probs = []
                self.all_probs = [None] * self.increment
                self.all_probs_pred_all = [[0 for j in range(offset1)] for i in range (self.increment)]

                for key in super_pred_dict.keys():
                    count = np.vstack(super_pred_dict[key]).astype(np.int64).sum(axis=0)

                    # prob_pred_all即为一个新类key属于各个旧类的概率
                    probs_pred_all = count / float(len(super_pred_dict[key]))

                    key_arr = np.fromstring(key[1:-1], dtype=np.int, sep=' ')
                    key_idx = key_arr.nonzero()[-1][-1].item()

                    prob_max = float(count.max()) / float(len(super_pred_dict[key]))

                    # all_pos_super_key是未经阈值threshold验证的超类预测结果，该情况下每个新类必被预测一个超类
                    # （直接取probs.argmax作为超类，而不在乎probs.max也许很低....）
                    max_label = np.zeros(len(count), dtype=np.int64)
                    max_label[count.argmax()] = 1
                    max_string = np.fromstring(key[1:-1], dtype=np.int, sep=' ')
                    max_string = np.hstack((max_label, max_string[offset1:offset2]))
                    # all_pos_super_key.append(max_string)
                    self.all_pos_super_key[key_idx-offset1] = max_string

                    # verified_super_key则是经阈值验证的最终超类预测结果
                    if prob_max > self.threshold:# 找到父类
                        preset_label = np.zeros(len(count), dtype=np.int64)
                        preset_label[count.argmax()] = 1

                        label_string = np.fromstring(key[1:-1], dtype=np.int, sep=' ')
                        label_string = np.hstack((preset_label, label_string[offset1:offset2]))

                        self.verified_super_key.append(label_string)

                    self.all_probs_pred_all[key_idx - offset1] = probs_pred_all
                    self.all_keys[key_idx-offset1] = key
                    self.all_probs[key_idx-offset1] = prob_max

                print(f"verified_super_key:\n{self.verified_super_key}")
                print(f"all_probs: {self.all_probs}")
                # print(f"all_probs_pred_all:\n{self.all_probs_pred_all}")
                # print(f"all_pos_super_key: \n {all_pos_super_key}")
                # return verified_super_key, all_keys, all_probs, all_pos_super_key

    def forward_net(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        an alias for self.net(x)

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            output (torch.Tensor): The network output of shape (minibatch size x output size)
            latent (torch.Tensor): The network latent variable of shape (minibatch size x last hidden size)
        """
        return self.net(x)

    def prepare_model_for_new_task(self, task_data: Optional[Dataset] = None, dist_args: Optional[dict] = None,
                                   **kwargs) -> None:
        """
        Takes place before the starting epoch of each new task.

        The shared functionality among the methods is that the seen classes are updated and the optimizer and scheduler
        are reset. (see _prepare_model_for_new_task for method specific functionality)

        Args:
            task_data (Optional[Dataset]): The new task data (default: None)
            dist_args (Optional[Dict]): a dictionary of the distributed processing values in case of multiple gpu (ex:
            rank of the device) (default: None)
            **kwargs: arguments that are method specific
        """
        self.seen_classes = list(set(self.seen_classes) | set(task_data.cur_task))

        # if self.cur_task_id > 0:
        #     self.task_eval_superclass_in_training(task_data)

        self.reset_optimizer_and_scheduler()
        # if task_data.cur_task_id > self.cur_task_id:
        #     assert task_data.cur_task_id == self.cur_task_id
        #     self.cur_task_id = task_data.cur_task_id
        self._prepare_model_for_new_task(task_data=task_data, dist_args=dist_args, **kwargs)

    @abstractmethod
    def _prepare_model_for_new_task(self, **kwargs) -> None:
        """
        A method specific function that takes place before the starting epoch of each new task (runs from the
            prepare_model_for_task function)
        This function needs to be defined in the inheriting method class
        """
        pass

    def consolidate_epoch_knowledge(self, val_metric=None, **kwargs) -> None:
        """
        Takes place after training on each epoch

        The shared functionality among the methods is that the scheduler takes a step. (see _consolidate_epoch_knowledge
        for method specific functionality)

        Args:
            val_metric (Optional): a metric to compare in case of reducing the learning rate on plateau (default: None)
            **kwargs: arguments that are method specific
        """
        self.step_scheduler(val_metric)
        self._consolidate_epoch_knowledge(**kwargs)

    @abstractmethod
    def _consolidate_epoch_knowledge(self, **kwargs) -> None:
        """
        A method specific function that takes place after training on each epoch (runs from the
        consolidate_epoch_knowledge function)
        This function needs to be defined in the inheriting method class
        """
        pass

    @abstractmethod
    def observe(self, x: torch.Tensor, y: torch.Tensor, in_buffer: Optional[torch.Tensor] = None,
                train: bool = True) -> Tuple[torch.Tensor, float]:
        """
        The method used for training and validation, returns a tensor of model predictions and the loss
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images
            y (torch.Tensor): A 2-d batch indicator tensor of shape (number of samples x number of classes)
            in_buffer (Optional[torch.Tensor]): A 1-d boolean tensor which indicates which sample is from the buffer.
            train (bool): Whether this is training or validation/test

        Returns:
            Tuple[torch.Tensor, float]:
            predictions (torch.Tensor) : a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
            loss (float): the value of the loss
        """
        pass

    @abstractmethod
    def consolidate_task_knowledge(self, **kwargs) -> None:
        """
        Takes place after training each task
        This function needs to be defined in the inheriting method class

        Args:
            **kwargs: arguments that are method specific
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The method used during inference, returns a tensor of model predictions
        This function needs to be defined in the inheriting method class

        Args:
            x (torch.Tensor): The batch of images

        Returns:
            torch.Tensor: a 2-d float tensor of the model predictions of shape (number of samples x number of classes)
        """
        pass
