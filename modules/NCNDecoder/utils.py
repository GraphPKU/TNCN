import torch
from torch import nn
from torch_sparse import SparseTensor
from torch import Tensor
import torch_sparse
from typing import List, Tuple, Final


class PermIterator:
    '''
    Iterator of a permutation
    '''
    def __init__(self, device, size, bs, training=True) -> None:
        self.bs = bs
        self.training = training
        self.idx = torch.randperm(
            size, device=device) if training else torch.arange(size,
                                                               device=device)

    def __len__(self):
        return (self.idx.shape[0] + (self.bs - 1) *
                (not self.training)) // self.bs

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr + self.bs * self.training > self.idx.shape[0]:
            raise StopIteration
        ret = self.idx[self.ptr:self.ptr + self.bs]
        self.ptr += self.bs
        return ret


def sparsesample(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > 0
    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand]

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask]

    ret = SparseTensor(row=samplerow.reshape(-1, 1).expand(-1, deg).flatten(),
                       col=samplecol.flatten(),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce().fill_value_(1.0)
    #print(ret.storage.value())
    return ret


def sparsesample2(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(
        row=torch.cat((samplerow, nosamplerow)),
        col=torch.cat((samplecol, nosamplecol)),
        sparse_sizes=adj.sparse_sizes()).to_device(
            adj.device()).fill_value_(1.0).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    #assert (ret.sum(dim=-1) == torch.clip(adj.sum(dim=-1), 0, deg)).all()
    return ret


def elem2spm(element: Tensor, sizes: List[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem


def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmnotoverlap_(adj1: SparseTensor,
                   adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    idx = torch.searchsorted(element1[:-1], element2)
    matchedmask = (element1[idx] == element2)

    maskelem1 = torch.ones_like(element1, dtype=torch.bool)
    maskelem1[idx[matchedmask]] = 0
    retelem1 = element1[maskelem1]

    retelem2 = element2[torch.logical_not(matchedmask)]
    return elem2spm(retelem1, adj1.sizes()), elem2spm(retelem2, adj2.sizes())


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> Tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)


def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               tarei: Tensor,
               filled1: bool = False,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    # a wrapper for functions above.
    adj1 = adj1[tarei[0]]
    adj2 = adj2[tarei[1]]
    # if calresadj:
    #     adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
    #     if cnsampledeg > 0:
    #         adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    #     if ressampledeg > 0:
    #         adjres1 = sparsesample_reweight(adjres1, ressampledeg)
    #         adjres2 = sparsesample_reweight(adjres2, ressampledeg)
    #     return adjoverlap, adjres1, adjres2
    # else:
    #     adjoverlap = spmoverlap_(adj1, adj2)
    #     if cnsampledeg > 0:
    #         adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    # return adjoverlap
    return spmoverlap_(adj1, adj2)


# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: Final[bool] # whether to rescale edge weight
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj

def sparse_diff(spm_x, spm_y):
    """
    Given 2 sparse tensor spm_x and spm_y, do the diff x - y.
    x: bs * nidx, y: bs * nidy
    require nidx >= nidy
    """
    # return spmoverlap_notoverlap_(spm_x, spm_y)[1]
    
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(spm_x)
    element2 = spm2elem(spm_y)

    if element1.shape[0] == 0:
        retelem1 = element1
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

    sizes = spm_x.sizes()
    return elem2spm(retelem1, sizes)

def sparse_intersect(spm_x, spm_y):
    """
    Given 2 sparse tensor spm_x and spm_y, do the intersect x and y.
    x: bs * nidx, y: bs * nidy
    """
    return spmoverlap_(spm_x, spm_y)

if __name__ == "__main__":
    # adj1 = SparseTensor.from_edge_index(
    #     torch.LongTensor([[0, 0, 1, 2, 3, 3, 4], [0, 1, 1, 2, 3, 4, 4]]))
    # adj2 = SparseTensor.from_edge_index(
    #     torch.LongTensor([[0, 3, 1, 2, 2, 2, 3], [0, 1, 1, 2, 2, 3, 3]]))
    # adj3 = SparseTensor.from_edge_index(
    #     torch.LongTensor([[0, 1,  2, 2, 2, 2, 3, 3, 3], [1, 0,  2, 3, 4, 5, 4, 5, 6]]))
    # # print(spmnotoverlap_(adj1, adj2))
    # # print(spmoverlap_(adj1, adj2))
    # print(spmoverlap_notoverlap_(adj1, adj2))
    # # print(sparsesample2(adj3, 3))
    # # print(sparsesample_reweight(adj3, 3))
    # tarei = torch.LongTensor([[2,2],
    #                           [3,3]])
    # print(adjoverlap(adj1=adj2, adj2=adj2, tarei=tarei))
    # adj1 = SparseTensor.from_edge_index(
    #     torch.LongTensor([[0, 0, 0, 1, 1], [1, 1, 2, 0, 3]]), sparse_sizes=(2,4)).fill_value_(1.0).coalesce()
    # print(adj1)
    # x = torch.tensor([[1,0,0,0],
    #                   [0,1,0,0],
    #                   [0,0,1,0],
    #                   [0,0,0,1],
    #                   ],
    #                   dtype=torch.float)
    # from torch_sparse.matmul import spmm_add
    # print(spmm_add(adj1, x))

    adj1 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 0, 1, 2, 3, 3, 4], 
                          [0, 1, 1, 2, 3, 4, 4]])).fill_value_(2.0).coalesce()
    adj2 = SparseTensor.from_edge_index(
        torch.LongTensor([[0, 3, 1, 2, 2, 2, 3], 
                          [0, 1, 1, 2, 2, 3, 3]]))
    # print(sparse_diff(adj2, adj1))
    # print(adj1 * adj1)
    time = torch.rand((2,3))
    cn = torch.rand((5, 3))
    spcn = SparseTensor.from_dense(cn)
    sptime = SparseTensor.from_dense(time)
    # cn*time
    # spcn*time
    print(spcn)
    print(sptime)
    print(spcn * sptime)
    
