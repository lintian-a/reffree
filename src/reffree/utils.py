import torch
import numpy as np

def set_seed_for_demo():
    """reproduce the training demo"""
    import random

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def path_import(absolute_path):
    """implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly"""
    import importlib

    spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def make_dir(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)

def slice_training(model, optimizer, data, ite, path):
    """This function is used to get the status of the training.
    """
    torch.save({
            "optimizer_state_dict": optimizer.state_dict(),
            "ite": ite
            },
            path + "/optimizer_weight_" + str(ite)
        )

    torch.save(
        model.state_dict(),
        path + "/net_weight_" + str(ite)
    )

    torch.save(data, path + "/data_" + str(ite))