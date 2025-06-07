import torch
import torch.nn.functional
from functional import bin_op_s_matrix, conv_cal

class DiffLogicLayer(torch.nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            layer: int,
            ops: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            connections: str = 'random',
    ):
        super().__init__()
        self.layer = layer
        self.ops = ops
        self.weights = torch.nn.parameter.Parameter(torch.randn(out_dim, self.ops, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor
        self.connections = connections

        # 注册缓冲区来保存 indices，这样它们会被包含在 state_dict 中
        self.register_buffer('indices_a', None)
        self.register_buffer('indices_b', None)
        
        # 初始化连接
        self._init_connections()

    def _init_connections(self):
        """初始化连接索引，并确保可重现性"""
        if self.connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(self.device), b.to(self.device)
            
            # 保存到缓冲区
            self.indices_a = a
            self.indices_b = b
        else:
            raise NotImplementedError(f'Unknown connections type: {self.connections}')

    @property
    def indices(self):
        """返回连接索引的元组 (a, b)"""
        return self.indices_a, self.indices_b

    def forward(self, x):
        assert x.shape[-1] == self.in_dim, (x.shape, self.in_dim)
        a, b = x[..., self.indices_a], x[..., self.indices_b]
        if self.training:
            x = bin_op_s_matrix(a, b, torch.nn.functional.softmax(self.weights, dim=-1), self.layer)
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), self.ops).to(torch.float32)
            x = bin_op_s_matrix(a, b, weights, self.layer)
        return x
    
class GroupSum(torch.nn.Module):
    def __init__(self, k: int, tau: float = 1., device='cuda'):
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device

    def forward(self, x):
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau


