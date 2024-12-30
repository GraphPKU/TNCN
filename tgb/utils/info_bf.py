import os.path as osp
import os

r"""
General space to store global information used elsewhere such as url links, evaluation metrics etc.
"""
PROJ_DIR = osp.dirname(osp.abspath(os.path.join(__file__, os.pardir))) + "/"

class BColors:
    """
    A class to change the colors of the strings.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

DATA_URL_DICT = {
    "reddit": "http://snap.stanford.edu/jodie/reddit.csv",
    "wikipedia": "http://snap.stanford.edu/jodie/wikipedia.csv",
    "mooc": "http://snap.stanford.edu/jodie/mooc.csv",
    "lastfm": "http://snap.stanford.edu/jodie/lastfm.csv",
    # "enron": "", https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing 
    # "uci": "", https://drive.google.com/drive/folders/1umS1m1YbOM10QOyVbGwtXrsiK3uTD7xQ?usp=sharing
}

DATA_EVAL_METRIC_DICT = {
    "reddit": "ap",
    "wikipedia": "ap",
    "mooc": "ap",
    "lastfm": "ap",
    "enron": "ap",
    "uci": "ap",
}

"""
ap, auc, 
mrr, recall@10, hits@5
"""
