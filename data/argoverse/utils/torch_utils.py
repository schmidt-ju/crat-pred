import numpy as np
import torch


def recursive_numpy_to_tensor(np_data):
    """
    Recursively convert lists of numpy arrays, tuples of numpy arrays
    or dictionary of numpy arrays to torch tensors

    Args:
        np_data: Numpy data to convert

    Returns:
        Converted torch tensor
    """
    if isinstance(np_data, np.ndarray):
        np_data = torch.from_numpy(np_data)
    elif isinstance(np_data, tuple):
        np_data = [recursive_numpy_to_tensor(x) for x in np_data]
    elif isinstance(np_data, list):
        np_data = [recursive_numpy_to_tensor(x) for x in np_data]
    elif isinstance(np_data, dict):
        for key in np_data:
            np_data[key] = recursive_numpy_to_tensor(np_data[key])
    return np_data


def collate_fn_dict(in_batch):
    """Custom collate_fn that returns a dictionary of lists

    Args:
        in_batch: Batch containing a list of dictionaries

    Returns:
        Batch containing a dictionary of lists
    """
    in_batch = recursive_numpy_to_tensor(in_batch)
    out_batch = dict()
    for key in in_batch[0]:
        out_batch[key] = [x[key] for x in in_batch]
    return out_batch
