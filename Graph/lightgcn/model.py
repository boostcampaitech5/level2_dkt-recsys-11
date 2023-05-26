from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Embedding, ModuleList
from torch.nn.modules.loss import _Loss

from torch_geometric.nn.conv import LGConv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index


class LightGCN(torch.nn.Module):
    """
        LightGCN은 그래프를 기반으로 임베딩을 선형적으로 전파함으로서 임베딩을 학습한다.
        그리고 Weighted Sum을 사용하고, 모든 레이어에서 학습한 임베딩의 Weighted Sum을 최종 임베딩으로 사용한다.
        
        prediction 종류로는, link prediction과 recommendation으로 나뉘는데, 우리는 link prediction을 사용한다.
        (연결됐으면 정답, 연결되지 않았으면 오답)
        
        임베딩은 'edge_index'에 의해 지정되는 그래프 연결성에 따라 전파되고, 
        rankings나 link probabilities는 'edge_label_index'에 따라 전파됩니다.
        
        Args:
            num_nodes (int): 그래프 안에 있는 노드의 개수
            embedding_dim (int): 노드 임베딩들의 차원
            num_layers (int): "torch_geometric.nn.conv.LGConv" 레이어의 개수 (강의에서 n-hop으로 말한 것, 일반적으로 1-4 사이)
            alpha (float or torch.Tensor, optional): 최종 임베딩을 aggregating 할 때 re-weighting을 하는 특정한 스칼라 혹은 벡터
            -> None으로 설정하면 논문에 나온 대로 1 / (num_layers + 1)로 설정한다.
            **kwargs (optional): "torch_geometric.nn.conv.LGConv" 레이어의 추가적인 args
    """
    def __init__(
        self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)

        self.embedding = Embedding(num_nodes, embedding_dim)
        self.convs = ModuleList([LGConv(**kwargs) for _ in range(num_layers)])

        self.reset_parameters()
        
        self.out = []
    def reset_parameters(self):
        """모듈의 모든 learnable parameters를 리셋"""
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        for conv in self.convs:
            conv.reset_parameters()


    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """그래프 안에 있는 노드들의 임베딩을 반환"""
        x = self.embedding.weight
        # alpha 기본값: 1. / (num_layers + 1)
        # alpha = torch.tensor([alpha] * (num_layers + 1))
        out = x * self.alpha[0] 
        
        # 지정한 hop 만큼 반복하면서 업데이트하기
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out


    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        """노드들의 쌍에 대해 랭킹을 연산

        Args:
            edge_index (torch.Tensor or SparseTensor): Edge tensor 그래프 연결성을 지정하는 것
            edge_label_index (torch.Tensor, optional): Edge tensor rankings 또는 probabilities를 계산할 노드 쌍을 지정하는 것
                만약에 'edge_label_index'가 None이면, edge_index가 대신 사용됨
            edge_weight (torch.Tensor, optional): 'edge_index'에 있는 각각의 edge의 weight
        """
        if edge_label_index is None:
            # is_sparse: Returns True if the input src is of type torch.sparse.Tensor (in any sparse layout) or of type torch_sparse.SparseTensor.
            if is_sparse(edge_index):
                # to_edge_index: Converts a torch.sparse.Tensor or a torch_sparse.SparseTensor to edge indices and edge attributes.
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index

        out = self.get_embedding(edge_index, edge_weight)
        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]
        
        return (out_src * out_dst).sum(dim=-1)


    def predict_link(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        prob: bool = False,
    ) -> Tensor:
        """ 
        edge_label_index에서 노드들 사이의 link predction

        Args:
            prob (bool, optional): 반올림을 안 할지 할지 여부
                (default: :obj:`False`)
                (현재 DKT Task 상관없긴 한데 기본적으로 Regression처럼 제출했음)
        """
        pred = self(edge_index, edge_label_index, edge_weight).sigmoid()
        return pred if prob else pred.round()

    # 현재 Task에서는 활용할 일 없어보임
    def recommend(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        src_index: OptTensor = None,
        dst_index: OptTensor = None,
        k: int = 1,
    ) -> Tensor:
        r"""Get top-:math:`k` recommendations for nodes in :obj:`src_index`.

        Args:
            src_index (torch.Tensor, optional): Node indices for which
                recommendations should be generated.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            dst_index (torch.Tensor, optional): Node indices which represent
                the possible recommendation choices.
                If set to :obj:`None`, all nodes will be used.
                (default: :obj:`None`)
            k (int, optional): Number of recommendations. (default: :obj:`1`)
        """
        out_src = out_dst = self.get_embedding(edge_index, edge_weight)

        if src_index is not None:
            out_src = out_src[src_index]

        if dst_index is not None:
            out_dst = out_dst[dst_index]

        pred = out_src @ out_dst.t()
        top_index = pred.topk(k, dim=-1).indices

        if dst_index is not None:  # Map local top-indices to original indices.
            top_index = dst_index[top_index.view(-1)].view(*top_index.size())

        return top_index


    def link_pred_loss(self, pred: Tensor, edge_label: Tensor,
                       **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (torch.Tensor): The predictions.
            edge_label (torch.Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    ### Not used ###
    def recommendation_loss(self, pos_edge_rank: Tensor, neg_edge_rank: Tensor,
                            lambda_reg: float = 1e-4, **kwargs) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).

        Args:
            pos_edge_rank (torch.Tensor): Positive edge rankings.
            neg_edge_rank (torch.Tensor): Negative edge rankings.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: :obj:`1e-4`)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        return loss_fn(pos_edge_rank, neg_edge_rank, self.embedding.weight)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.num_nodes}, '
                f'{self.embedding_dim}, num_layers={self.num_layers})')


### Not used ###
class BPRLoss(_Loss):
    r"""The Bayesian Personalized Ranking (BPR) loss.

    The BPR loss is a pairwise loss that encourages the prediction of an
    observed entry to be higher than its unobserved counterparts
    (see `here <https://arxiv.org/abs/2002.02126>`__).

    .. math::
        L_{\text{BPR}} = - \sum_{u=1}^{M} \sum_{i \in \mathcal{N}_u}
        \sum_{j \not\in \mathcal{N}_u} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj})
        + \lambda \vert\vert \textbf{x}^{(0)} \vert\vert^2

    where :math:`lambda` controls the :math:`L_2` regularization strength.
    We compute the mean BPR loss for simplicity.

    Args:
        lambda_reg (float, optional): The :math:`L_2` regularization strength
            (default: 0).
        **kwargs (optional): Additional arguments of the underlying
            :class:`torch.nn.modules.loss._Loss` class.
    """
    __constants__ = ['lambda_reg']
    lambda_reg: float

    def __init__(self, lambda_reg: float = 0, **kwargs):
        super().__init__(None, None, "sum", **kwargs)
        self.lambda_reg = lambda_reg

    def forward(self, positives: Tensor, negatives: Tensor,
                parameters: Tensor = None) -> Tensor:
        r"""Compute the mean Bayesian Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`positives` vector and i-th entry
            in the :obj:`negatives` entry should correspond to the same
            entity (*.e.g*, user), as the BPR is a personalized ranking loss.

        Args:
            positives (Tensor): The vector of positive-pair rankings.
            negatives (Tensor): The vector of negative-pair rankings.
            parameters (Tensor, optional): The tensor of parameters which
                should be used for :math:`L_2` regularization
                (default: :obj:`None`).
        """
        n_pairs = positives.size(0)
        log_prob = F.logsigmoid(positives - negatives).mean()
        regularization = 0

        if self.lambda_reg != 0:
            regularization = self.lambda_reg * parameters.norm(p=2).pow(2)

        return (-log_prob + regularization) / n_pairs