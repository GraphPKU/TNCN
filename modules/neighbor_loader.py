"""
Neighbor Loader

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""

import copy
from typing import Callable, Dict, Tuple

import torch
from torch import Tensor


class LastNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        # Filter invalid neighbors (identified by `e_id < 0`).
        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        # Relabel node indices.
        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]

        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor):
        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, : self.size], dense_neighbors], dim=-1
        )

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)

    def reset_state(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)

class RandomNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size
        self.src = torch.ones((0,), dtype=torch.long, device=device)
        self.dst = torch.ones((0,), dtype=torch.long, device=device)
        self.e_id = torch.ones((0,), dtype=torch.long, device=device)
        self.device = device
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        e_ids = []
        for idx in n_id.unique():
            mask = torch.logical_or(self.src == idx, self.dst == idx)
            e_id = self.e_id[mask]
            # randomly sample size neighbors
            e_id = e_id[torch.randperm(e_id.size(0))[:self.size]]
            e_ids.append(e_id)
        e_ids = torch.cat(e_ids)
        src = self.src[e_ids - 1]
        dst = self.dst[e_ids - 1]
        n_ids = torch.cat([n_id, src, dst]).unique()
        # reindex
        self._assoc[n_ids] = torch.arange(n_ids.size(0), device=n_ids.device)
        src, dst = self._assoc[src], self._assoc[dst]

        return n_ids, torch.stack([src, dst]), e_ids
        
    def insert(self, src: Tensor, dst: Tensor):
        self.src = torch.cat([self.src, src])
        self.dst = torch.cat([self.dst, dst])
        self.e_id = torch.cat([self.e_id, torch.arange(self.e_id.size(0) + 1, self.e_id.size(0) + 1 + src.size(0), dtype=torch.long, device=src.device)])

    def reset_state(self):
        self.src = torch.ones((0,), dtype=torch.long, device=self.device)
        self.dst = torch.ones((0,), dtype=torch.long, device=self.device)
        self.e_id = torch.ones((0,), dtype=torch.long, device=self.device)

class FastBiasRandomNeighborLoader:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size
        self.src = torch.ones((0,), dtype=torch.long, device=device)
        self.dst = torch.ones((0,), dtype=torch.long, device=device)
        self.e_id = torch.ones((0,), dtype=torch.long, device=device)
        self.device = device
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)

    def __call__(self, n_id: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mask = torch.logical_or(torch.isin(self.src, n_id), torch.isin(self.dst, n_id))
        e_id = self.e_id[mask]
        # randomly sample size neighbors
        e_id = e_id[torch.randperm(e_id.size(0))[:self.size * n_id.unique().size(0)]] # sample size * num_nodes neighbors
        src = self.src[e_id - 1]
        dst = self.dst[e_id - 1]
        n_id = torch.cat([n_id, src, dst]).unique()
        # reindex
        self._assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        src, dst = self._assoc[src], self._assoc[dst]

        return n_id, torch.stack([src, dst]), e_id
        
    def insert(self, src: Tensor, dst: Tensor):
        self.src = torch.cat([self.src, src])
        self.dst = torch.cat([self.dst, dst])
        self.e_id = torch.cat([self.e_id, torch.arange(self.e_id.size(0) + 1, self.e_id.size(0) + 1 + src.size(0), dtype=torch.long, device=src.device)])

    def reset_state(self):
        self.src = torch.ones((0,), dtype=torch.long, device=self.device)
        self.dst = torch.ones((0,), dtype=torch.long, device=self.device)
        self.e_id = torch.ones((0,), dtype=torch.long, device=self.device)