import numpy as np
import random
import os
import pickle
from typing import Any
import sys
import argparse
import json
import torch


# import torch
def save_pkl(obj: Any, fname: str) -> None:
    r"""
    save a python object as a pickle file
    """
    with open(fname, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pkl(fname: str) -> Any:
    r"""
    load a python object from a pickle file
    """
    with open(fname, "rb") as handle:
        return pickle.load(handle)


def set_random_seed(seed: int):
    r"""
    setting random seed for reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_args():
    parser = argparse.ArgumentParser('*** TNCN ***')
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-wiki')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--num_epoch', type=int, help='Number of epochs', default=50)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_dim', type=int, help='Memory dimension', default=100)
    parser.add_argument('--time_dim', type=int, help='Time dimension', default=100)
    parser.add_argument('--emb_dim', type=int, help='Embedding dimension', default=100)
    parser.add_argument('--tolerance', type=float, help='Early stopper tolerance', default=1e-6)
    parser.add_argument('--patience', type=float, help='Early stopper patience', default=0)
    parser.add_argument('--num_run', type=int, help='Number of iteration runs', default=1)
    parser.add_argument('--num_neighbors', type=int, help='Number of stored recent neighbors', default=10)
    parser.add_argument('--nei_sampler', type=str, help='Neighbor sampler, r for random, fr for fast random with bias, l for last', default='l')
    parser.add_argument('--hop_num', type=int, help='hop number of neighbors', default=1)
    parser.add_argument(
        '--NCN_mode', type=int, help='NCN hop kind, 0 for 0&1 hop, 1 for 1 hop, 2 for 0~2 hop', default=1
        )
    parser.add_argument('--per_val_epoch', type=int, help='val per k epoch', default=1)
    parser.add_argument('--patch_num', type=int, help='patch number', default=1)
    parser.add_argument('--msg_func', type=str, help='message function', default='identity')
    parser.add_argument('--emb_func', type=str, help='embedding function', default='GraphAttention')
    parser.add_argument('--agg_func', type=str, help='aggregator', default='last')
    parser.add_argument('--device', type=str, help='device', default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv


def save_results(new_results: dict, filename: str):
    r"""
    save (new) results into a json file
    :param: new_results (dictionary): a dictionary of new results to be saved
    :filename: the name of the file to save the (new) results
    """
    if os.path.isfile(filename):
        # append to the file
        with open(filename, 'r+') as json_file:
            file_data = json.load(json_file)
            # convert file_data to list if not
            if type(file_data) is dict:
                file_data = [file_data]
            file_data.append(new_results)
            json_file.seek(0)
            json.dump(file_data, json_file, indent=4)
    else:
        # dump the results
        with open(filename, 'w') as json_file:
            json.dump(new_results, json_file, indent=4)
