import argparse
import importlib
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchvision
from math import ceil
from ml_logger import logbook as ml_logbook

torch.set_printoptions(threshold=np.inf)
np.set_printoptions(threshold=np.inf)
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from iirc.datasets_loader import get_lifelong_datasets
from iirc.utils.utils import print_msg
from iirc.definitions import CIL_SETUP, IIRC_SETUP
import lifelong_methods
import lifelong_methods.utils
import utils
from prepare_config import prepare_config
from train import task_train, tasks_eval


def get_transforms(dataset_name):
    essential_transforms_fn = None
    augmentation_transforms_fn = None
    if "cifar100" in dataset_name:
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
        ])
    elif "imagenet" in dataset_name:
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        essential_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
        augmentation_transforms_fn = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])
    return essential_transforms_fn, augmentation_transforms_fn


def print_task_info(lifelong_dataset):
    class_names_samples = {class_: 0 for class_ in lifelong_dataset.cur_task}
    for idx in range(len(lifelong_dataset)):
        labels = lifelong_dataset.get_labels(idx)
        for label in labels:
            if label in class_names_samples.keys():
                class_names_samples[label] += 1
    print_msg(f"Task {lifelong_dataset.cur_task_id} number of samples: {len(lifelong_dataset)}")
    for class_name, num_samples in class_names_samples.items():
        print_msg(f"{class_name} is present in {num_samples} samples")


def main_worker(gpu, config: dict, dist_args: dict = None):
    # DP
    config["DP"] = (config["DP"] and config["ngpus_per_node"] > 1)
    DP = config["DP"]
    # distributed training DDP
    distributed = dist_args is not None
    assert not (DP and dist_args is not None), "DP and DDP cannot be used together"

    if distributed:
        dist_args["gpu"] = gpu
        device = torch.device(f"cuda:{gpu}")
        dist_args["rank"] = dist_args["node_rank"] * dist_args["ngpus_per_node"] + gpu
        rank = dist_args["rank"]
        print_msg(f"Using GPU {gpu} with rank {rank}")
        dist.init_process_group(backend="nccl", init_method=dist_args["dist_url"],
                                world_size=dist_args["world_size"], rank=dist_args["rank"])
    elif gpu is not None:
        device = torch.device(f"cuda:{gpu}")
        rank = 0
        print_msg(f"Using GPU: {gpu}")
    elif DP:
        device = config["device"]
        main_device_id = int((str(device).split(":")[1]))
        device_ids = [main_device_id] + [i for i in range(config['ngpus_per_node']) if i != main_device_id]
        config["device_ids"] = device_ids
        rank = 0
        print_msg(f"using GPU: {device_ids}\n")
    else:
        device = config["device"]
        rank = 0
        print_msg(f"using GPU: {device}")

    checkpoint = None

    non_loadable_attributes = ["logging_path", "dataset_path", "batch_size"]

    if DP:
        non_loadable_attributes.append("device_ids")

    temp = {key: val for key, val in config.items() if key in non_loadable_attributes}

    checkpoint_path = os.path.join(config['logging_path'], 'latest_model')

    json_logs_file_name = 'jsonlogs.jsonl'

    # load ckpt
    if os.path.isfile(checkpoint_path):
        logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                            filemode='a+',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)
        if distributed:
            print_msg(f"\n\nLoading checkpoint {checkpoint_path} on gpu {dist_args['rank']}")
            checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{dist_args['gpu']}")
        elif DP:
            print_msg(f"Loaded the checkpoint successfully on gpu {device}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
        else:
            print_msg(f"\n\nLoading checkpoint {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)

        config = checkpoint['config']

        for key in non_loadable_attributes:
            config[key] = temp[key]

        if distributed:
            print_msg(f"Loaded the checkpoint successfully on gpu {dist_args['rank']}")
        elif DP:
            print_msg(f"Loaded the checkpoint successfully on gpu {device}")
        else:
            print_msg(f"Loaded the checkpoint successfully")

        if rank == 0:
            print_msg(f"Resuming from task {config['cur_task_id']} epoch {config['task_epoch']}")
            # Remove logs related to training after the checkpoint was saved
            utils.remove_extra_logs(config['cur_task_id'], config['task_epoch'],
                                    os.path.join(config['logging_path'], json_logs_file_name))
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(config['logging_path'], exist_ok=True)
            if os.path.isfile(os.path.join(config['logging_path'], json_logs_file_name)):
                os.remove(os.path.join(config['logging_path'], json_logs_file_name))
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
            logging.basicConfig(filename=os.path.join(config['logging_path'], "logs.txt"),
                                filemode='a',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.INFO)

    random.seed(config["seed"])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config["wandb_log"]:
        wandb_config = dict(project=config["wandb_project"], config=config, allow_val_change=True,
                            name=config["id"], id=config["id"])
    else:
        wandb_config = None

    logbook_config = ml_logbook.make_config(
        logger_dir=config['logging_path'],
        filename=json_logs_file_name,
        create_multiple_log_files=False,
        wandb_config=wandb_config,
    )
    logbook = ml_logbook.LogBook(config=logbook_config)
    if checkpoint is None and (not distributed or dist_args['rank'] == 0):
        config_to_write = {str(key): str(value) for key, value in config.items()}
        logbook.write_config(config_to_write)

    essential_transforms_fn, augmentation_transforms_fn = get_transforms(config['dataset'])
    #      lifelong_datasets (Dict[str, BaseDataset]): a dictionary with the keys corresponding to the four splits (train,
    #      intask_validation, posttask_validation, test), and the values containing the dataset object inheriting from
    #      BaseDataset for that split.
    lifelong_datasets, tasks, class_names_to_idx = \
        get_lifelong_datasets(config['dataset'], dataset_root=config['dataset_path'],
                              tasks_configuration_id=config["tasks_configuration_id"],
                              essential_transforms_fn=essential_transforms_fn,
                              augmentation_transforms_fn=augmentation_transforms_fn, cache_images=False,
                              joint=config["joint"])
    n_cla_per_tsk = [len(task) for task in tasks]
    total_num_classes = len(class_names_to_idx)
    idx_to_class_names = dict((y, x) for x, y in class_names_to_idx.items())

    if config["complete_info"]:
        for lifelong_dataset in lifelong_datasets.values():
            lifelong_dataset.enable_complete_information_mode()

    if checkpoint is None:
        metadata = {}
        metadata['n_tasks'] = len(tasks)
        metadata["total_num_classes"] = total_num_classes
        metadata["tasks"] = tasks
        metadata["class_names_to_idx"] = class_names_to_idx
        metadata["n_cla_per_tsk"] = n_cla_per_tsk
        if rank == 0:
            metadata_to_write = {str(key): str(value) for key, value in metadata.items()}
            logbook.write_metadata(metadata_to_write)
    else:
        metadata = checkpoint['metadata']

    # Assert that methods files lie in the folder "methods"
    method = importlib.import_module('lifelong_methods.methods.' + config["method"])
    model = method.Model(metadata["n_cla_per_tsk"], metadata["class_names_to_idx"], config)
    buffer_dir = None
    map_size = None

    if "imagenet" in config["dataset"]:
        if config['n_memories_per_class'] > 0:
            n_classes = sum(metadata["n_cla_per_tsk"])
            buffer_dir = config['logging_path']
            map_size = int(config['n_memories_per_class'] * n_classes * 1.4e6)
        elif config['total_n_memories'] > 0:
            buffer_dir = config['logging_path']
            map_size = int(config['total_n_memories'] * 1.4e6)
    buffer = method.Buffer(config, buffer_dir, map_size, essential_transforms_fn, augmentation_transforms_fn)

    if gpu is not None:
        torch.cuda.set_device(gpu)
        model.to(device)
        model.net = torch.nn.parallel.DistributedDataParallel(model.net, device_ids=[gpu])
    else:
        model.to(config["device"])

    if DP:
        model.net = torch.nn.DataParallel(model.net, device_ids=device_ids)

    # If loading a checkpoint, load the corresponding state_dicts
    if checkpoint is not None:
        lifelong_methods.utils.load_model(checkpoint, model, buffer, lifelong_datasets)
        print_msg(f"Loaded the state dicts successfully")
        starting_task = config["cur_task_id"]
    else:
        starting_task = 0

    if "ave_test_m_jaccard_dict" in config and "ave_test_strict_acc_dict" in config:
        ave_test_m_jaccard_dict = config["ave_test_m_jaccard_dict"]
        ave_test_strict_acc_dict = config["ave_test_strict_acc_dict"]
    else:
        ave_test_m_jaccard_dict = {"metric_name":"ave_test_m_jaccard"}
        ave_test_strict_acc_dict = {"metric_name":"ave_test_strict_acc"}

    for cur_task_id in range(starting_task, len(tasks)):
        if checkpoint is not None and cur_task_id == starting_task and config["task_epoch"] > 0:
            new_task_starting = False
        else:
            new_task_starting = True

        if cur_task_id > model.cur_task_id:
            model.cur_task_id = cur_task_id

        if config["incremental_joint"]:
            for lifelong_dataset in lifelong_datasets.values():
                lifelong_dataset.load_tasks_up_to(cur_task_id)
        else:
            for lifelong_dataset in lifelong_datasets.values():
                lifelong_dataset.choose_task(cur_task_id) # choose the data of the current task (train, intask_valid, posttask_valid, test)

        if rank == 0:
            print_task_info(lifelong_datasets["train"])

        if cur_task_id > 0:
            Dt_train_loader = DataLoader(lifelong_datasets['train'], batch_size=config['batch_size'], shuffle=False, num_workers= config['num_workers'], pin_memory=True)
            if new_task_starting:
                model.task_eval_superclass_in_training(Dt_train_loader)
                if config["method"] == 'RTDHH':
                    model.update_C(lifelong_datasets['train'], buffer)
                    model.solving_ft()
                    model.solving_bt()

        model.net.eval()
        if new_task_starting:
            model.prepare_model_for_new_task(task_data=lifelong_datasets["train"], dist_args=dist_args, buffer=buffer,
                                             num_workers=config["num_workers"])

        if distributed:
            dist.barrier()

        # train mode
        # train and intask_valid
        model.net.train()
        start_time = time.time()

        task_train(model, buffer, lifelong_datasets, config, metadata, logbook=logbook, dist_args=dist_args)

        end_time = time.time()
        print_msg(f"Time taken on device {rank} for training on task {cur_task_id}: "
                  f"{round((end_time - start_time) / 60, 2)} mins")

        # eval mode
        model.net.eval()

        # expand buffer
        buffer.update_buffer_new_task(
            lifelong_datasets["train"], dist_args=dist_args, model=model, batch_size=config["batch_size"]
        )

        if distributed:
            dist.barrier()
        # Assert that all gpus share the same buffer (check the last image)
        if distributed and len(buffer) > 0:
            with buffer.disable_augmentations():
                image_1, _, _ = buffer[-1]
                image_1 = image_1.to(device)
                if rank == 0:
                    image_2, _, _ = buffer[-1]
                    image_2 = image_2.to(device)
                else:
                    image_2 = torch.empty_like(image_1)
                torch.distributed.broadcast(image_2, 0)
                if torch.all(image_1.eq(image_2)):
                    print_msg(f"buffers are similar between rank {dist_args['rank']} and rank 0")

        model.consolidate_task_knowledge(
            buffer=buffer, device=device, batch_size=config["batch_size"]
        )

        if distributed:
            dist.barrier()

        # posttask_valid and test
        tasks_eval(
            model, lifelong_datasets["posttask_valid"], cur_task_id, config, metadata, logbook=logbook,
            dataset_type="valid", dist_args=dist_args
        )
        average_test_modified_jaccard, average_test_strict_acc = tasks_eval(
            model, lifelong_datasets["test"], cur_task_id, config, metadata, logbook=logbook, dataset_type="test",
            dist_args=dist_args
        )

        ave_test_m_jaccard_dict.update({f"task{cur_task_id}": average_test_modified_jaccard})
        ave_test_strict_acc_dict.update({f"task{cur_task_id}": average_test_strict_acc})

        if rank == 0:
            utils.log_task(cur_task_id, deepcopy(ave_test_m_jaccard_dict), logbook)
            utils.log_task(cur_task_id, deepcopy(ave_test_strict_acc_dict), logbook)

        # update config
        config["cur_task_id"] = cur_task_id + 1
        config["task_epoch"] = 0
        config["ave_test_m_jaccard_dict"] = ave_test_m_jaccard_dict
        config["ave_test_strict_acc_dict"] = ave_test_strict_acc_dict

        # save model
        if config["save_each_task_model"] and rank == 0:
            save_file = os.path.join(config['logging_path'], f"task_{cur_task_id}_model")
            lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)
        if rank == 0:
            print_msg("Saving checkpoint")
            save_file = os.path.join(config['logging_path'], "latest_model")
            lifelong_methods.utils.save_model(save_file, config, metadata, model, buffer, lifelong_datasets)

    ave_test_m_jaccard_dict.update({"Ave": sum(list(ave_test_m_jaccard_dict.values())[1:])/(len(ave_test_m_jaccard_dict)-1)})
    ave_test_strict_acc_dict.update({"Ave": sum(list(ave_test_strict_acc_dict.values())[1:])/(len(ave_test_strict_acc_dict)-1)})
    if rank == 0:
        logbook.write_metric(ave_test_m_jaccard_dict)
        logbook.write_metric(ave_test_strict_acc_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="iirc_cifar100",
                        choices=["incremental_cifar100", "iirc_cifar100", "incremental_imagenet_full",
                                 "incremental_imagenet_lite", "iirc_imagenet_full", "iirc_imagenet_lite", "iirc_imagenet_subset"])
    parser.add_argument('--epochs_per_task', type=int, default=140,
                        help="The number of epochs per task. This number is multiplied by 2 for the first task.")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--dataset_path', type=str, default="/home/yqn/data")
    parser.add_argument('--metadata_root', type=str, default="/home/yqn/RTDHH/iirc/metadata")
    parser.add_argument('--group', type=str, default="experiments_cifar",
                        choices=["experiments_cifar", "experiments_imagenet_subset", "experiments_imagenet_lite", "experiments_imagenet_full"],
                        help="The parent folder of the experiment results, so as to group related experiments easily")
    parser.add_argument('--logging_path_root', type=str, default="results",
                        help="The directory where the logs and results will be saved")
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--n_layers', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6,
                        help="Number of workers used to fetch the data for the dataloader")

    # Parameters for creating the tasks
    parser.add_argument('--tasks_configuration_id', type=int, default=0, choices=range(0, 10),
                        help="The task configuration id. Ignore for joint training")
    # The training method
    parser.add_argument('--method', type=str, default="RTDHH",
                        choices=["finetune", "mask_seen_classes", "lucir", "agem", "icarl_cnn",
                                 "icarl_norm", "RTDHH"])
    parser.add_argument('--complete_info', action='store_true',
                        help='use the complete information during training (a multi-label setting)')
    parser.add_argument('--incremental_joint', action='store_true',
                        help="keep all data from previous tasks, while updating their labels as per the observed "
                             "classes (use only with complete_info and without buffer)")
    parser.add_argument('--joint', action='store_true',
                        help="load all classes during the first task. This option ignores the tasks_configuration_id "
                             "(use only with complete_info and without buffer)")

    # RTDHH hyperparameters
    parser.add_argument('--sinkhorn', type=float, default=0.464,
                        help="The regularization term for sinkhorn")
    parser.add_argument('--ft_weight', type=float, default=0.0001, help="Only functions when the ada_ft_weight is False.")
    parser.add_argument('--ada_ft_weight', action='store_true', help="Whether to use the adaptive ft_weight which changes with the task number.")
    parser.add_argument('--bt_weight', type=float, default=0.5, help="Weight for the bt_output when caculating the weighted sum of distill_output and bt_output.")
    parser.add_argument('--period', type=int, default=10, help="The updating (epochs) interval of C.")
    parser.add_argument('--threshold', type=float, default=0.6, help="The threshold to decide hard relation (the superclass for the new classes).")
    parser.add_argument('--C_norm_bdr', type=float, default=0.6, help="The When prob_max > C_norm_bdr， the new class corresponding to the prob_max will be"
                                                                    "normalized when calculating C.")
    parser.add_argument('--c', type=float, default=0.05, help="The (negative) curvature of the Poincare ball when using hyperbolic space to calculate C.")


    # The optimizer parameters
    parser.add_argument('--optimizer', type=str, default="momentum", choices=["adam", "momentum"])
    parser.add_argument('--lr', type=float, default=1.0, help="The initial learning rate for each task")
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help="The multiplicative factor for learning rate decay at the epochs specified")
    parser.add_argument('--lr_schedule', nargs='+', type=int, default=None,
                        help="the epochs per task at which to multiply the current learning rate by lr_gamma "
                             "(resets after each task). This setting is ignored if reduce_lr_on_plateau is specified")
    parser.add_argument('--reduce_lr_on_plateau', action='store_true',
                        help='reduce the lr on plateau based on the validation performance metric.If set to True,'
                             'the lr_schedule is ignored')
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Model selection and validation set
    parser.add_argument('--checkpoint_interval', type=int, default=5,
                        help="The number of epochs within each task after which the checkpoint is updated. When a task "
                             "is finished, the checkpoint is updated anyways, so set to 0 for checkpointing only after "
                             "each task")
    parser.add_argument('--use_best_model', action='store_true',
                        help='use the best model after training each task based on the best task validation accuracy')
    parser.add_argument('--save_each_task_model', action='store_true',
                        help='save the model after each task')

    # The buffer parameters
    parser.add_argument('--total_n_memories', type=int, default=2300,
                        help="The total replay buffer size, which is divided by the observed number of classes to get "
                             "the number of memories kept per task, note that the number of memories per task here is "
                             "not fixed but rather decreases as the number of tasks observed increases (with a minimum "
                             "of 1). If n_memories_per_class is set to a value greater than -1, the "
                             "n_memories_per_class is used instead.")
    parser.add_argument('--n_memories_per_class', type=int, default=20,
                        help="The number of samples to keep from each class, if set to -1, the total_n_memories "
                             "argument is used instead")
    parser.add_argument('--buffer_sampling_multiplier', type=float, default=1.0,
                        help="A multiplier for sampling from the buffer more/less times than the size of the buffer "
                             "(for example a multiplier of 2 samples from the buffer (with replacement) twice its size "
                             "per epoch)")
    parser.add_argument('--memory_strength', type=float, default=1.0,
                        help="a weight to be multiplied by the loss from the buffer")
    parser.add_argument('--max_mems_pool_per_class', type=int, default=1e5,
                        help="Maximum size of the samples pool per class from which the buffer chooses the exemplars, "
                             "use -1 for choosing from the whole class samples.")

    # LUCIR Hyperparameters
    parser.add_argument('--lucir_lambda', type=float, default=5.0,
                        help="a weight to be multiplied by the distillation loss (only for the LUCIR method)")
    parser.add_argument('--lucir_margin_1', type=float, default=0.5,
                        help="The 1st margin used with the margin ranking loss for in the LUCIR method")
    parser.add_argument('--lucir_margin_2', type=float, default=0.5,
                        help="The 2nd margin used with the margin ranking loss for in the LUCIR method")

    # Distributed arguments
    parser.add_argument('--DP', action='store_true', help='Use DataParallel, DP is not compatible with DDP(multiprocessing_distributed)!')
    # parser.add_argument('--device', type=int, default=0, help="gpu device id")
    parser.add_argument('--num_nodes', type=int, default=1,
                        help="num of nodes to use")
    parser.add_argument('--node_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url', default="tcp://172.19.27.221:10000", type=str,
                        help='node rank for distributed training')

    # select hyperparameter configurations in .json folder
    parser.add_argument('--config_path', default="../experiments/experiments_configurations/configurations_cifar.json",
                        type=str, choices=["../experiments/experiments_configurations/configurations_cifar.json",
                        "../experiments/experiments_configurations/configurations_imagenet_full.json",
                        "../experiments/experiments_configurations/configurations_imagenet_lite.json",
                        "../experiments/experiments_configurations/configurations_imagenet_subset.json"],
                        help='Path to the JSON configuration file')
    parser.add_argument('--config_group', type=str, default="RTDHH_20",
                        help='The name of config_group to be selected')
    args = parser.parse_args()

    if args.config_path :
        with open(args.config_path, 'r') as config_file:
            config_data = json.load(config_file)
        # Add each key-value pair from the config file as a command-line argument
        for key, value in config_data.get(args.config_group).items():
            if type(value) == dict and key != "iterable_arguments":
                for key1, value1 in value.items():
                    if hasattr(args, key1):
                       setattr(args, key1, value1)
            elif hasattr(args, key):
                setattr(args, key, value)

    config = prepare_config(args)

    if "iirc" in config["dataset"]:
        config["setup"] = IIRC_SETUP
    else:
        config["setup"] = CIL_SETUP

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # config['device'] = torch.device(f"cuda:{config['device']}" if torch.cuda.is_available() else 'cpu')
    config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config["ngpus_per_node"] = torch.cuda.device_count()
    print_msg(f"number of gpus per node: {config['ngpus_per_node']}")
    if args.multiprocessing_distributed or args.num_nodes > 1:
        dist_args = {"num_nodes": args.num_nodes, "node_rank": args.node_rank}
        dist_args["ngpus_per_node"] = config["ngpus_per_node"]
        dist_args["world_size"] = dist_args["ngpus_per_node"] * dist_args["num_nodes"]
        config["batch_size"] = int(ceil(config["batch_size"] / dist_args["world_size"]))
        dist_args["dist_url"] = args.dist_url
        mp.spawn(main_worker, nprocs=dist_args["ngpus_per_node"], args=(config, dist_args))
    else:
        main_worker(None, config, None)
