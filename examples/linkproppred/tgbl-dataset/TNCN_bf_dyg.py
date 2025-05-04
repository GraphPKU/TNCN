import math
import timeit
from tqdm import tqdm

import os
import os.path as osp
from pathlib import Path
import numpy as np

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_sparse import SparseTensor

from torch_geometric.loader import TemporalDataLoader

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results
from tgb.linkproppred.evaluate import Evaluator
from modules.decoder import LinkPredictor
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from modules.NCNDecoder.NCNPred import NCNPredictor
from tgb.linkproppred.data_bf import get_data#, RandEdgeSampler
from modules.neighbor_sampler_ns import get_neighbor_sampler, NeighborSampler, NegativeEdgeSampler


# ==========
# ========== Define helper function...
# ==========

def train():
    r"""
    Training procedure for TNCN model
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
    # neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_src, neg_dst = train_neg_edge_sampler.sample(src.size(0))
        n_t = torch.cat([t, t])

        pos_n_id, pos_unique_idx = np.unique(torch.cat([src, pos_dst]).cpu().numpy(), return_index=True)
        
        pos_n_t = n_t[pos_unique_idx]     
        # assert len(n_id) == len(n_t)
        # n_id = torch.cat([src, pos_dst, neg_dst])

        pos_n_id, pos_edge_index, pos_e_id = train_find_neighbor(train_neighbor_sampler, pos_n_id, pos_n_t, HOP_NUM)

        # Get updated memory of all nodes involved in the computation.
        pos_z, pos_last_update = model['memory'](pos_n_id)
        # pos_raw_n_feat = node_features[pos_n_id]
        # pos_z = pos_z + pos_raw_n_feat

        pos_z1 = model['gnn'](
            pos_z,
            pos_last_update,
            pos_edge_index,
            train_data.t[pos_e_id].to(device),
            train_data.msg[pos_e_id].to(device),
        )
        
        ################################################################

        # z2 = ord_func(z, edge_index, data.msg[e_id].to(device))
        # pos_z2 = ord_func(pos_z, pos_edge_index, train_data.msg[pos_e_id].to(device))
        # pos_z1 += pos_z2
        
        src_re = assoc[src]
        pos_re = assoc[pos_dst]

        pos_time_info = (pos_last_update, t)
        pos_out = model['link_pred'](pos_z1, pos_edge_index, torch.stack([src_re,pos_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=pos_time_info)#, z=z2)

        neg_n_id, neg_unique_idx = np.unique(torch.cat([neg_src, neg_dst]).cpu().numpy(), return_index=True)
        neg_n_t = n_t[neg_unique_idx]
        neg_n_id, neg_edge_index, neg_e_id = train_find_neighbor(train_neighbor_sampler, neg_n_id, neg_n_t, HOP_NUM)

        neg_z, neg_last_update = model['memory'](neg_n_id)
        # neg_raw_n_feat = node_features[neg_n_id]
        # neg_z = neg_z + neg_raw_n_feat

        neg_z1 = model['gnn'](
            neg_z,
            neg_last_update,
            neg_edge_index,
            train_data.t[neg_e_id].to(device),
            train_data.msg[neg_e_id].to(device),
        )

        # neg_z2 = ord_func(neg_z, neg_edge_index, train_data.msg[neg_e_id].to(device))
        # neg_z1 += neg_z2

        nsrc_re = assoc[neg_src]
        neg_re = assoc[neg_dst]

        neg_time_info = (neg_last_update, t)
        neg_out = model['link_pred'](neg_z1, neg_edge_index, torch.stack([nsrc_re,neg_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=neg_time_info)#, z=z2)

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        # neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

        # input("Press Enter to continue...")

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler: NegativeEdgeSampler, split_mode):
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
    model['memory'].eval(split_mode)
    model['gnn'].eval()
    model['link_pred'].eval()

    perf_list = {"acc": [], "ap": [], "auc": []}

    for idx, pos_batch in enumerate(tqdm(loader)):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )
        
        #########################
        neg_src, neg_dst = neg_sampler.sample(pos_src.size(0))
        n_t = torch.cat([pos_t, pos_t])

        pos_n_id, pos_unique_idx = np.unique(torch.cat([pos_src, pos_dst]).cpu().numpy(), return_index=True)
        pos_n_t = n_t[pos_unique_idx]
        # n_id = torch.cat([pos_src, neg_src, pos_dst, neg_dst]).unique()
        pos_n_id, pos_edge_index, pos_e_id = test_find_neighbor(full_neighbor_sampler, pos_n_id, pos_n_t, HOP_NUM)
        
        # id_num = n_id.size(0)
        # Get updated memory of all nodes involved in the computation.
        pos_z, pos_last_update = model['memory'](pos_n_id)
        # pos_raw_n_feat = node_features[pos_n_id]
        # pos_z = pos_z + pos_raw_n_feat
        pos_z1 = model['gnn'](
            pos_z,
            pos_last_update,
            pos_edge_index,
            data.t[pos_e_id].to(device),
            data.msg[pos_e_id].to(device),
        )

        # z2 = ord_func(z, edge_index, data.msg[e_id].to(device))
        # pos_z2 = ord_func(pos_z, pos_edge_index, data.msg[pos_e_id].to(device))
        # pos_z1 += pos_z2

        src_re = assoc[pos_src]
        pos_re = assoc[pos_dst]

        pos_time_info = (pos_last_update, pos_t)
        pos_pred = model['link_pred'](pos_z1, pos_edge_index, torch.stack([src_re, pos_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=pos_time_info)#, z=z2)

        neg_n_id, neg_unique_idx = np.unique(torch.cat([neg_src, neg_dst]).cpu().numpy(), return_index=True)
        neg_n_t = n_t[neg_unique_idx]
        neg_n_id, neg_edge_index, neg_e_id = test_find_neighbor(full_neighbor_sampler, neg_n_id, neg_n_t, HOP_NUM)

        neg_z, neg_last_update = model['memory'](neg_n_id)
        # neg_raw_n_feat = node_features[neg_n_id]
        # neg_z = neg_z + neg_raw_n_feat

        neg_z1 = model['gnn'](
            neg_z,
            neg_last_update,
            neg_edge_index,
            data.t[neg_e_id].to(device),
            data.msg[neg_e_id].to(device),
        )

        # neg_z2 = ord_func(neg_z, neg_edge_index, data.msg[neg_e_id].to(device))
        # neg_z1 += neg_z2

        neg_re = assoc[neg_dst]
        nsrc_re = assoc[neg_src]

        neg_time_info = (neg_last_update, pos_t)
        neg_pred = model['link_pred'](neg_z1, neg_edge_index, torch.stack([nsrc_re, neg_re]), NCN_MODE, cn_time_decay=CN_TIME_DECAY, time_info=neg_time_info)#, z=z2)

        pred_score = torch.cat([pos_pred, neg_pred])
        pred_label = pred_score > 0.0
        true_label = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        # compute the performance metrics
        acc = pred_label.eq(true_label).sum().item() / true_label.size(0)
        ap = average_precision_score(true_label.cpu().numpy(), pred_score.cpu().numpy())
        auc = roc_auc_score(true_label.cpu().numpy(), pred_score.cpu().numpy())

        perf_list["acc"].append(acc)
        perf_list["ap"].append(ap)
        perf_list["auc"].append(auc)

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        # neighbor_loader.insert(pos_src, pos_dst)

    perf_metrics = {
        "acc": np.mean(perf_list["acc"]),
        "ap": np.mean(perf_list["ap"]),
        "auc": np.mean(perf_list["auc"]),
    }

    return perf_metrics

# def find_neighbor(neighbor_loader:LastNeighborLoader, n_id, k=1):
#     for i in range(k-1):
#         n_id, _, _ = neighbor_loader(n_id)
#     neighbor_info = neighbor_loader(n_id)
#     return neighbor_info
def train_find_neighbor(neighbor_sampler: NeighborSampler, n_id, times, k=1):
    # np_n_id = n_id.cpu().numpy()
    np_n_id = n_id
    np_times = times.cpu().numpy()
    # print("np_n_id", np_n_id)
    for i in range(k-1):
        n_node_ids, _, n_ts = neighbor_sampler.get_historical_neighbors(np_n_id, np_times, NUM_NEIGHBORS)
        np_n_id = np.concatenate([np_n_id, n_node_ids])
        np_times = np.concatenate([np_times, n_ts])
        # get the unique nodes and mask
        np_n_id, unique_idx = np.unique(np_n_id, return_index=True)
        np_times = np_times[unique_idx]

    n_node_ids, n_edge_ids, n_ts = neighbor_sampler.get_historical_neighbors(np_n_id, np_times, NUM_NEIGHBORS)
    # print("n_edge_ids", n_edge_ids-1)
    # for i in range(len(n_node_ids)):
    #     print(n_node_ids[i], train_data.src[n_edge_ids[i]-1], train_data.dst[n_edge_ids[i]-1])
    #     assert (n_node_ids[i] == train_data.src[n_edge_ids[i]-1] or n_node_ids[i] == train_data.dst[n_edge_ids[i]-1])
    # assert len(n_node_ids) == len(n_ts)
    # assert len(n_node_ids) == len(n_edge_ids)
    np_n_id = np.concatenate([np_n_id, n_node_ids])
    np_times = np.concatenate([np_times, n_ts])
    # get the unique nodes and mask
    np_n_id, unique_idx = np.unique(np_n_id, return_index=True)
    np_times = np_times[unique_idx]
    np_edge_id = n_edge_ids - 1
    # print("np_edge_id", np_edge_id)
    # print("np_n_id", np_n_id)
    torch_n_id = torch.from_numpy(np_n_id).long().to(device)
    torch_edge_index = torch.stack([train_data.src[np_edge_id], train_data.dst[np_edge_id]]).long().to(device)
    torch_e_id = torch.from_numpy(np_edge_id).long().to(device)

    assoc[torch_n_id] = torch.arange(torch_n_id.size(0), device=device)
    torch_edge_index[0] = assoc[torch_edge_index[0]]
    torch_edge_index[1] = assoc[torch_edge_index[1]]

    neighbor_info = (torch_n_id, torch_edge_index, torch_e_id)
    return neighbor_info

def test_find_neighbor(neighbor_sampler: NeighborSampler, n_id, times, k=1):
    # np_n_id = n_id.cpu().numpy()
    np_n_id = n_id
    np_times = times.cpu().numpy()
    # print("np_n_id", np_n_id)
    for i in range(k-1):
        n_node_ids, _, n_ts = neighbor_sampler.get_historical_neighbors(np_n_id, np_times, NUM_NEIGHBORS)
        np_n_id = np.concatenate([np_n_id, n_node_ids])
        np_times = np.concatenate([np_times, n_ts])
        # get the unique nodes and mask
        np_n_id, unique_idx = np.unique(np_n_id, return_index=True)
        np_times = np_times[unique_idx]

    n_node_ids, n_edge_ids, n_ts = neighbor_sampler.get_historical_neighbors(np_n_id, np_times, NUM_NEIGHBORS)
    # print("n_edge_ids", n_edge_ids-1)
    # for i in range(len(n_node_ids)):
    #     print(n_node_ids[i], train_data.src[n_edge_ids[i]-1], train_data.dst[n_edge_ids[i]-1])
    #     assert (n_node_ids[i] == train_data.src[n_edge_ids[i]-1] or n_node_ids[i] == train_data.dst[n_edge_ids[i]-1])
    # assert len(n_node_ids) == len(n_ts)
    # assert len(n_node_ids) == len(n_edge_ids)
    np_n_id = np.concatenate([np_n_id, n_node_ids])
    np_times = np.concatenate([np_times, n_ts])
    # get the unique nodes and mask
    np_n_id, unique_idx = np.unique(np_n_id, return_index=True)
    np_times = np_times[unique_idx]
    np_edge_id = n_edge_ids - 1
    # print("np_edge_id", np_edge_id)
    # print("np_n_id", np_n_id)
    torch_n_id = torch.from_numpy(np_n_id).long().to(device)
    torch_edge_index = torch.stack([data.src[np_edge_id], data.dst[np_edge_id]]).long().to(device)
    torch_e_id = torch.from_numpy(np_edge_id).long().to(device)

    assoc[torch_n_id] = torch.arange(torch_n_id.size(0), device=device)
    torch_edge_index[0] = assoc[torch_edge_index[0]]
    torch_edge_index[1] = assoc[torch_edge_index[1]]

    neighbor_info = (torch_n_id, torch_edge_index, torch_e_id)
    return neighbor_info

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
CN_TIME_DECAY = False
inductive = True

MODEL_NAME = 'TNCN'
# ==========

# set the device
device = args.device

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
# neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
train_neighbor_sampler = get_neighbor_sampler(train_data, seed=0)
full_neighbor_sampler = get_neighbor_sampler(data, seed=1)

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

link_pred = NCNPredictor(in_channels=EMB_DIM, hidden_channels=EMB_DIM, 
                         out_channels=1, NCN_mode=NCN_MODE).to(device)

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
# train_neg_sampler = RandEdgeSampler((train_data.src, ), (train_data.dst, ), 2024)
# val_neg_sampler = RandEdgeSampler((train_data.src, val_data.src), (train_data.dst, val_data.dst), 2024)
# test_neg_sampler = RandEdgeSampler((train_data.src, val_data.src, test_data.src), (train_data.dst, val_data.dst, test_data.dst), 2024)
train_neg_edge_sampler = NegativeEdgeSampler(train_data.src.detach().cpu().numpy(), train_data.dst.detach().cpu().numpy(), seed=0, device=device)
val_neg_edge_sampler = NegativeEdgeSampler(data.src.detach().cpu().numpy(), data.dst.detach().cpu().numpy(), seed=0, device=device)
new_node_val_neg_edge_sampler = NegativeEdgeSampler(new_node_val_data.src.detach().cpu().numpy(), new_node_val_data.dst.detach().cpu().numpy(), seed=1, device=device)
test_neg_edge_sampler = NegativeEdgeSampler(data.src.detach().cpu().numpy(), data.dst.detach().cpu().numpy(), seed=2, device=device)
new_node_test_neg_edge_sampler = NegativeEdgeSampler(new_node_test_data.src.detach().cpu().numpy(), new_node_test_data.dst.detach().cpu().numpy(), seed=3, device=device)

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_{NCN_MODE}_results_dyg.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}_NCN_{NCN_MODE}_nei_{NUM_NEIGHBORS}_dyg'
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
    nn_val_perf_list = {
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
        if epoch % PER_VAL_EPOCH == 0:
            start_val = timeit.default_timer()

            if inductive:
                train_mem_backup = model["memory"].backup_memory()
                nn_perf_metric_val = test(new_val_loader, new_node_val_neg_edge_sampler, split_mode="val")
                model["memory"].restore_memory(train_mem_backup)

            # perf_metric_val = test(val_loader, neg_sampler, split_mode="val")
            perf_metric_val = test(val_loader, val_neg_edge_sampler, split_mode="val")
            print(f"\tTransductive Validation results: {perf_metric_val}")
            if inductive:
                print(f"\tInductive Validation results: {nn_perf_metric_val}")
            print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
            val_perf_list["acc"].append(perf_metric_val["acc"])
            val_perf_list["ap"].append(perf_metric_val["ap"])
            val_perf_list["auc"].append(perf_metric_val["auc"])
            if inductive:
                nn_val_perf_list["acc"].append(nn_perf_metric_val["acc"])
                nn_val_perf_list["ap"].append(nn_perf_metric_val["ap"])
                nn_val_perf_list["auc"].append(nn_perf_metric_val["auc"])

            # check for early stopping
            if early_stopper.step_check(perf_metric_val["ap"], model):
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
    if inductive:
        val_mem_backup = model["memory"].backup_memory()
        nn_perf_metric_test = test(new_test_loader, new_node_test_neg_edge_sampler, split_mode="test")
        model["memory"].restore_memory(val_mem_backup)
    # perf_metric_test = test(test_loader, neg_sampler, split_mode="test")
    perf_metric_test = test(test_loader, test_neg_edge_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> Transductive <<< ")
    print(f"\tTest results: {perf_metric_test}")
    if inductive:
        print(f"INFO: Test: Evaluation Setting: >>> Inductive <<< ")
        print(f"\tTest results: {nn_perf_metric_test}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  'NCN_mode': NCN_MODE,
                  'emb_dim': EMB_DIM,
                  'lr': LR,
                  'bs': BATCH_SIZE,
                  'num_neighbors': NUM_NEIGHBORS,
                  f'val_perf': val_perf_list,
                  f'test_perf': {"Transductive": perf_metric_test, 
                                 "Inductive": nn_perf_metric_test} if inductive else perf_metric_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
