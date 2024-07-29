import math
import timeit

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor, MergeLayer
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from tgb.linkproppred.data_bf import get_data, RandEdgeSampler

# ==========
# ========== Define helper function...
# ==========

def train():
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_src, neg_dst = train_neg_sampler.sample(src.size(0))

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        # n_id = torch.cat([src, neg_src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        raw_n_feat = node_features[n_id]
        z = (z + raw_n_feat)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler: RandEdgeSampler, split_mode):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        loader: an object containing positive attributes of the positive edges of the evaluation set
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluaiton
    """
    model['memory'].eval()
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = {"acc": [], "ap": [], "auc": []}

    for pos_batch in loader:
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        
        neg_src, neg_dst = neg_sampler.sample(pos_src.size(0))

        n_id = torch.cat([pos_src, pos_dst, neg_dst]).unique()
        # n_id = torch.cat([pos_src, neg_src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        raw_n_feat = node_features[n_id]
        z = (z + raw_n_feat) 
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )
        
        src_re = assoc[pos_src]
        nsrc_re = assoc[neg_src]
        pos_re = assoc[pos_dst]
        neg_re = assoc[neg_dst]

        pos_pred = model['link_pred'](z[src_re], z[pos_re])
        neg_pred = model['link_pred'](z[src_re], z[neg_re])

        pred_score = torch.cat([pos_pred, neg_pred])
        pred_label = pred_score > 0.0
        true_label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        acc = pred_label.eq(true_label).sum().item() / true_label.size(0)
        ap = average_precision_score(true_label.cpu().numpy(), pred_score.cpu().numpy())
        auc = roc_auc_score(true_label.cpu().numpy(), pred_score.cpu().numpy())

        perf_list["acc"].append(acc)
        perf_list["ap"].append(ap)
        perf_list["auc"].append(auc)

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = {
        "acc": np.mean(perf_list["acc"]),
        "ap": np.mean(perf_list["ap"]),
        "auc": np.mean(perf_list["auc"]),
    }

    return perf_metrics

# ==========
# ==========
# ==========


# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = args.data
DATA_DIR = f"./tgb/data_bf/{DATA}"
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = args.num_neighbors
HOP_NUM = args.hop_num
NCN_MODE = args.NCN_mode
PER_VAL_EPOCH = args.per_val_epoch


MODEL_NAME = 'TGN'
# ==========

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # data loading
# dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
# train_mask = dataset.train_mask
# val_mask = dataset.val_mask
# test_mask = dataset.test_mask
# data = dataset.get_TemporalData()
# data = data.to(device)
# metric = dataset.eval_metric

# train_data = data[train_mask]
# val_data = data[val_mask]
# test_data = data[test_mask]

# train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
# val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
# test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# get data
data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, metric = get_data(DATA_DIR, DATA, include_padding=True)
data = data.to(device)
node_features = data.node_features
train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)
new_node_val_data = new_node_val_data.to(device)
new_node_test_data = new_node_test_data.to(device)
train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)
new_val_loader = TemporalDataLoader(new_node_val_data, batch_size=BATCH_SIZE)
new_test_loader = TemporalDataLoader(new_node_test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

# link_pred = LinkPredictor(in_channels=EMB_DIM).to(device)
link_pred = MergeLayer(EMB_DIM, EMB_DIM, EMB_DIM, 1).to(device)

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

# evaluator = Evaluator(name=DATA)
# neg_sampler = dataset.negative_sampler
train_neg_sampler = RandEdgeSampler((train_data.src, ), (train_data.dst, ), 2024)
val_neg_sampler = RandEdgeSampler((train_data.src, val_data.src), (train_data.dst, val_data.dst), 2024)
test_neg_sampler = RandEdgeSampler((train_data.src, val_data.src, test_data.src), (train_data.dst, val_data.dst, test_data.dst), 2024)

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # # loading the validation negative samples
    # dataset.load_val_ns()

    val_perf_list = {
        "acc": [],
        "ap": [],
        "auc": []
    }
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss = train()
        print(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

        # validation
        start_val = timeit.default_timer()
        perf_metric_val = test(val_loader, val_neg_sampler, split_mode="val")
        print(f"\tValidation results: {perf_metric_val}")
        print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
        val_perf_list["acc"].append(perf_metric_val["acc"])
        val_perf_list["ap"].append(perf_metric_val["ap"])
        val_perf_list["auc"].append(perf_metric_val["auc"])

        # check for early stopping
        if early_stopper.step_check(perf_metric_val['auc'], model):
            break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # # loading the test negative samples
    # dataset.load_test_ns()

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test = test(test_loader, test_neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest results: {perf_metric_test}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  f'val_perf': val_perf_list,
                  f'test_perf': perf_metric_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
