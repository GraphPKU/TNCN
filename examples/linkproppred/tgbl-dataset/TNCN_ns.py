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
from modules.emb_module import GraphAttentionEmbedding, TimeEmbedding, IdentityEmbedding
from modules.msg_func import IdentityMessage, MLPMessage
from modules.msg_agg import LastAggregator, MeanAggregator
from modules.neighbor_loader import LastNeighborLoader, RandomNeighborLoader, FastBiasRandomNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import  EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

from modules.NCNDecoder.NCNPred import NCNPredictor


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
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    # clear the upd_time_table
    upd_time_table.zero_()

    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        bsrc, bpos_dst, bt, bmsg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        bneg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (bsrc.size(0),),
            dtype=torch.long,
            device=device,
        )

        bn_id = torch.cat([bsrc, bpos_dst, bneg_dst]).unique()
        bz, blast_update = model['memory'](bn_id)
        assert blast_update.allclose(upd_time_table[bn_id])

        loss = 0.0

        patch_size = math.ceil(bsrc.size(0) / K_PATCH)
        for k in range(K_PATCH):
            start_idx = k * patch_size
            end_idx = min((k + 1) * patch_size, bsrc.size(0))
            exact_patch_size = end_idx - start_idx
            if exact_patch_size <= 0:
                break

            src, pos_dst, t, msg, neg_dst = (
                bsrc[start_idx:end_idx],
                bpos_dst[start_idx:end_idx],
                bt[start_idx:end_idx],
                bmsg[start_idx:end_idx],
                bneg_dst[start_idx:end_idx],
            )
            n_id = torch.cat([src, pos_dst, neg_dst]).unique()
            # n_id, edge_index, e_id = neighbor_loader(n_id)
            n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            id_num = n_id.size(0)

            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)

            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )
            
            ################################################################
            
            src_re = assoc[src]
            pos_re = assoc[pos_dst]
            neg_re = assoc[neg_dst]

            def generate_adj_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                mask = ~ torch.isin(loop_edge, edge_index)
                loop_edge = loop_edge[mask]
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                    # adj = SparseTensor.from_edge_index(edge_index).to_device(device)
                return adj
            
            def generate_adj_0_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                return adj
            
            def generate_adj_0_1_2_hop(adj):
                # adj = SparseTensor.to_dense(adj)
                # adj = torch.mm(adj, adj)
                # adj = SparseTensor.from_dense(adj)
                adj = adj.matmul(adj)
                return adj

            if NCN_MODE == 0:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adjs = (adj_0_1, adj_1)
            elif NCN_MODE == 1:
                adj_1 = generate_adj_1_hop()
                adjs = (adj_1)
            elif NCN_MODE == 2:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adj_0_1_2 = generate_adj_0_1_2_hop(adj_1)
                adjs = (adj_0_1, adj_1, adj_0_1_2)
            else: 
                raise ValueError('Invalid NCN Mode! Mode must be 0, 1, or 2.')

            pos_out = model['link_pred'](z, adjs, torch.stack([src_re,pos_re]), NCN_MODE)
            neg_out = model['link_pred'](z, adjs, torch.stack([src_re,neg_re]), NCN_MODE)
            loss += criterion(pos_out, torch.ones_like(pos_out)) * exact_patch_size
            loss += criterion(neg_out, torch.zeros_like(neg_out)) * exact_patch_size

            neighbor_loader.insert(src, pos_dst)
            for s, d, t in zip(src, pos_dst, t):
                upd_time_table[s] = t
                upd_time_table[d] = t

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(bsrc, bpos_dst, bt, bmsg)

        loss /= batch.num_events
        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader, neg_sampler, split_mode):
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

    perf_list = []

    for pos_batch in tqdm(loader):
        pos_src, pos_dst, pos_t, pos_msg = (
            pos_batch.src,
            pos_batch.dst,
            pos_batch.t,
            pos_batch.msg,
        )

        bn_id = torch.cat([pos_src, pos_dst]).unique()
        bz, blast_update = model['memory'](bn_id)
        assert blast_update.allclose(upd_time_table[bn_id])

        #########################
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)

        for idx, neg_batch in enumerate(neg_batch_list):
            src = torch.full((1 + len(neg_batch),), pos_src[idx], device=device)
            dst = torch.tensor(
                np.concatenate(
                    ([np.array([pos_dst.cpu().numpy()[idx]]), np.array(neg_batch)]),
                    axis=0,
                ),
                device=device,
            )

            n_id = torch.cat([src, dst]).unique()
            # n_id, edge_index, e_id = neighbor_loader(n_id)
            n_id, edge_index, e_id = find_neighbor(neighbor_loader, n_id, HOP_NUM)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)
            
            id_num = n_id.size(0)
            # Get updated memory of all nodes involved in the computation.
            z, last_update = model['memory'](n_id)
            z = model['gnn'](
                z,
                last_update,
                edge_index,
                data.t[e_id].to(device),
                data.msg[e_id].to(device),
            )

            def generate_adj_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                mask = ~ torch.isin(loop_edge, edge_index)
                loop_edge = loop_edge[mask]
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                    # adj = SparseTensor.from_edge_index(edge_index).to_device(device)
                return adj
            
            def generate_adj_0_1_hop():
                loop_edge = torch.arange(id_num, dtype=torch.int64, device=device)
                loop_edge = torch.stack([loop_edge,loop_edge])
                if edge_index.size(1) == 0:
                    adj = SparseTensor.from_edge_index(loop_edge).to_device(device)
                else:
                    adj = SparseTensor.from_edge_index(torch.cat((loop_edge, edge_index, torch.stack([edge_index[1], edge_index[0]])),dim=-1)).to_device(device)
                return adj
            
            def generate_adj_0_1_2_hop(adj):
                # adj = SparseTensor.to_dense(adj)
                # adj = torch.mm(adj, adj)
                # adj = SparseTensor.from_dense(adj)
                adj = adj.matmul(adj)
                return adj

            if NCN_MODE == 0:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adjs = (adj_0_1, adj_1)
            elif NCN_MODE == 1:
                adj_1 = generate_adj_1_hop()
                adjs = (adj_1)
            elif NCN_MODE == 2:
                adj_0_1 = generate_adj_0_1_hop()
                adj_1 = generate_adj_1_hop()
                adj_0_1_2 = generate_adj_0_1_2_hop(adj_1)
                adjs = (adj_0_1, adj_1, adj_0_1_2)
            else:
                raise ValueError('Invalid NCN Mode! Mode must be 0, 1, or 2.')

            y_pred = model['link_pred'](z, adjs, torch.stack([assoc[src], assoc[dst]]), NCN_MODE)

            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0, :].squeeze(dim=-1).cpu()]),
                "y_pred_neg": np.array(y_pred[1:, :].squeeze(dim=-1).cpu()),
                "eval_metric": [metric],
            }
            perf = evaluator.eval(input_dict)[metric]
            perf_list.append(perf)

            neighbor_loader.insert(src[0:1], dst[0:1])
            upd_time_table[src[0:1]] = pos_t[idx]
            upd_time_table[dst[0:1]] = pos_t[idx]

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(pos_src, pos_dst, pos_t, pos_msg)
        # neighbor_loader.insert(pos_src, pos_dst)

    perf_tensor = torch.tensor(perf_list)
    transductive_perf = perf_tensor[transductive_mask[split_mode]]
    inductive_perf = perf_tensor[inductive_mask[split_mode]]

    perf_metrics = float(torch.tensor(perf_list).mean())
    transductive_perf = float(transductive_perf.mean())
    inductive_perf = float(inductive_perf.mean())

    return perf_metrics, transductive_perf, inductive_perf

def find_neighbor(neighbor_loader, n_id, k=1):
    for i in range(k-1):
        n_id, _, _ = neighbor_loader(n_id)
    neighbor_info = neighbor_loader(n_id)
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
device = args.device

K_PATCH = args.patch_num

MODEL_NAME = 'TNCN'
# ==========

# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]
observed_nodes = torch.cat([train_data.src, train_data.dst]).unique()
transductive_mask = {
    "val": (torch.isin(val_data.src, observed_nodes) & torch.isin(val_data.dst, observed_nodes)).cpu().numpy(),
    "test": (torch.isin(test_data.src, observed_nodes) & torch.isin(test_data.dst, observed_nodes)).cpu().numpy(),
}
inductive_mask = {
    "val": ~ (torch.isin(val_data.src, observed_nodes) & torch.isin(val_data.dst, observed_nodes)).cpu().numpy(),
    "test": ~ (torch.isin(test_data.src, observed_nodes) & torch.isin(test_data.dst, observed_nodes)).cpu().numpy(),
}

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighhorhood sampler
if args.nei_sampler == 'l':
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
elif args.nei_sampler == 'r':
    neighbor_loader = RandomNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
elif args.nei_sampler == 'fr':
    neighbor_loader = FastBiasRandomNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)
else:
    raise ValueError('Invalid neighbor sampler! Sampler must be "l", "r", or "fr".')

def get_msg_func(msg_func):
    if msg_func == 'identity':
        return IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM)
    elif msg_func == 'mlp':
        return MLPMessage(data.msg.size(-1), MEM_DIM, TIME_DIM)
    else:
        raise ValueError('Invalid message function! Function must be "identity" or "mlp".')

def get_agg_module(agg_func):
    if agg_func == 'last':
        return LastAggregator()
    elif agg_func == 'mean':
        return MeanAggregator()
    else:
        raise ValueError('Invalid aggregator function! Function must be "last" or "mean".')
    
def get_emb_module(emb_func):
    if emb_func == 'GraphAttention':
        return GraphAttentionEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    elif emb_func == 'Time':
        return TimeEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    elif emb_func == 'Identity':
        return IdentityEmbedding(
            in_channels=MEM_DIM,
            out_channels=EMB_DIM,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)
    else:
        raise ValueError('Invalid embedding function! Function must be "GraphAttentionEmbedding", "TimeEmbedding", or "IdentityEmbedding".')

msg_module = get_msg_func(args.msg_func)
agg_module = get_agg_module(args.agg_func)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=msg_module,
    aggregator_module=agg_module,
).to(device)

gnn = get_emb_module(args.emb_func)

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
upd_time_table = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

print("==========================================================")
print(f"=================*** {MODEL_NAME}: LinkPropPred: {DATA} ***=============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_{NCN_MODE}_{args.msg_func}_{args.emb_func}_{args.agg_func}_{args.nei_sampler}_{args.patch_num}_results.json'

for run_idx in range(NUM_RUNS):
    print('-------------------------------------------------------------------------------')
    print(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}_NCN_{NCN_MODE}_{args.msg_func}_{args.emb_func}_{args.agg_func}_{args.nei_sampler}_{args.patch_num}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    val_perf_list = []
    trans_val_list = []
    ind_val_list = []
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
            perf_metric_val, trans_val, ind_val = test(val_loader, neg_sampler, split_mode="val")
            print(f"\tValidation {metric}: {perf_metric_val: .4f}, Transductive {metric}: {trans_val: .4f}, Inductive {metric}: {ind_val: .4f}")
            print(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
            val_perf_list.append(perf_metric_val)
            trans_val_list.append(trans_val)
            ind_val_list.append(ind_val)

            # check for early stopping
            if early_stopper.step_check(perf_metric_val, model):
                break

    train_val_time = timeit.default_timer() - start_train_val
    print(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")

    # ==================================================== Test
    # first, load the best model
    early_stopper.load_checkpoint(model)

    # loading the test negative samples
    dataset.load_test_ns()

    # final testing
    start_test = timeit.default_timer()
    perf_metric_test, trans_test, ind_test = test(test_loader, neg_sampler, split_mode="test")

    print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
    print(f"\tTest: {metric}: {perf_metric_test: .4f}, Transductive {metric}: {trans_test: .4f}, Inductive {metric}: {ind_test: .4f}")
    test_time = timeit.default_timer() - start_test
    print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

    save_results({'model': MODEL_NAME,
                  'data': DATA,
                  'run': run_idx,
                  'seed': SEED,
                  'NCN_mode': NCN_MODE,
                  'msg_func': args.msg_func,
                  'emb_func': args.emb_func,
                  'agg_func': args.agg_func,
                  'nei_sampler': args.nei_sampler,
                  'patch_num': args.patch_num,
                  f'val {metric}': val_perf_list,
                  f'trans_val {metric}': trans_val_list,
                  f'ind_val {metric}': ind_val_list,
                  f'test {metric}': perf_metric_test,
                  f'trans_test {metric}': trans_test,
                  f'ind_test {metric}': ind_test,
                  'test_time': test_time,
                  'tot_train_val_time': train_val_time
                  }, 
    results_filename)

    print(f"INFO: >>>>> Run: {run_idx}, elapsed time: {timeit.default_timer() - start_run: .4f} <<<<<")
    print('-------------------------------------------------------------------------------')

print(f"Overall Elapsed Time (s): {timeit.default_timer() - start_overall: .4f}")
print("==============================================================")
