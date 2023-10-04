import torch.nn as nn
import torch
from models.encoder import DescriptorEncoder, KeypointEncoder
from models.transformer_encoder import TransformerEncoderLayer

#TODO why remove bias for conv?

### OT
def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)


    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z
###

# ### cnn
# def conv(in_channels, out_channels, relu, norm, dropout, kernel_size, stride, padding=1):
#     layers = []

#     conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
#                            kernel_size=kernel_size, stride=stride, 
#                            padding=padding, bias=False)
    
#     layers.append(conv_layer)

#     if norm == "instance":
#         layers.append(nn.InstanceNorm2d(out_channels))
#     elif norm == "batch":
#         layers.append(nn.BatchNorm2d(out_channels))
#     elif norm is None:
#         pass
#     else:
#         raise RuntimeError('Illagel norm passed: ' + norm)
    
#     if relu:
#         layers.append(nn.ReLU())
#         #layers.append(nn.LeakyReLU(0.2))
#         #layers.append(nn.LeakyReLU(0.01))

#     if dropout:
#         layers.append(nn.Dropout(0.5))

#     return nn.Sequential(*layers)

# class CNN(nn.Module):
#     def __init__(self, params):
#         super(CNN, self).__init__()

#         input_dim = params['input_dim']
#         output_dims = params['output_dims']
#         dropouts = params['dropouts']
#         kernel_sizes = params['kernel_sizes']
#         strides = params['strides']
#         paddings = params['paddings']
#         norm = params['norm']

#         conv_layers = []
#         prev_dim = input_dim

#         num_convs = len(output_dims)
#         for i in range(num_convs):
#             output_dim = output_dims[i]
#             dropout = dropouts[i]
#             kernel_size = kernel_sizes[i]
#             stride = strides[i]
#             padding = paddings[i]

#             if i == num_convs - 1:
#                 relu = False
#             else:
#                 relu = True

#             if i == num_convs - 1:
#                 norm = None

#             conv_layer = conv(prev_dim, output_dim, relu=relu, norm=norm,
#                               dropout=dropout, 
#                               kernel_size=kernel_size, stride=stride, padding=padding)
#             conv_layers.append(conv_layer)

#             prev_dim = output_dim

#         self.network = nn.Sequential(*conv_layers)
    
#     def forward(self, x):
#         x = self.network(x)
#         return x
    
# ###

### mlp
def fc(in_dim, out_dims, norm):
    layers = []
    prev_dim = in_dim
    for i in range(len(out_dims) - 1):
        fc = nn.Linear(prev_dim, out_dims[i])
        relu = nn.ReLU()
        #relu = nn.LeakyReLU(0.01)

        layers.append(fc)
        if norm == "instance":
            layers.append(nn.InstanceNorm1d(out_dims[i]))
        elif norm == "batch":
            layers.append(nn.BatchNorm1d(out_dims[i]))
        elif norm is None:
            pass
        else:
            raise RuntimeError('Illagel norm passed: ' + norm)
        layers.append(relu)

        prev_dim = out_dims[i]

    final_fc = nn.Linear(prev_dim, out_dims[-1])
    layers.append(final_fc)

    return nn.Sequential(*layers)

class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()

        input_dim = params['input_dim']
        output_dims = params['output_dims']
        norm = params['norm']

        self.network = fc(input_dim, output_dims, norm)

    def forward(self, x):
        x = self.network(x)
        return x
###

### gnn
class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
def gnn(layers, d_model, dim_feedforward, num_heads, batch_first=True):
    network = []
    for layer in layers:
        transformer = TransformerEncoderLayer(name=layer, d_model=d_model, 
                                              nhead=num_heads,
                                              dim_feedforward=dim_feedforward,
                                              batch_first=batch_first)
        network.append(transformer)
    
    return mySequential(*network)

class GNN(nn.Module):
    def __init__(self, params):
        super(GNN, self).__init__()

        layers = params['layers']
        d_model = params['d_model']
        dim_feedforward = params['dim_feedforward']
        num_heads = params['num_heads']

        self.network = gnn(layers, d_model, dim_feedforward, num_heads)

    def forward(self, src0, src1):
        src0, src1 = self.network(src0, src1)

        return src0, src1

def attention(query, key, value):
    dim = query.shape[3]
    scores = torch.einsum('bnhd,bmhd->bhnm', query, key) / dim**.5

    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bmhd->bnhd', prob, value), prob

#I disagreed with how this was originally written
#so I re-wrote it
class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads #dk and dv
        self.num_heads = num_heads

        self.merge = MLP({'input_dim': d_model, 'output_dims': [d_model], 'norm': None})
        self.projQ = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdk
        self.projK = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdk
        self.projV = nn.ModuleList([MLP({'input_dim': d_model, 'output_dims': [self.dim], 'norm': None}) for _ in range(num_heads)]) #dmodelxdv

        # self.merge = nn.Conv1d(d_model, d_model, kernel_size=1) #h*dvxdmodel
        # #self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.projQ = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        # self.projK = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdk
        # self.projV = nn.ModuleList([nn.Conv1d(d_model, self.dim, kernel_size=1) for _ in range(num_heads)]) #dmodelxdv

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        # query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
        #                      for l, x in zip(self.proj, (query, key, value))]
        queries = [q(query) for q in self.projQ]
        keys = [k(key)for k in self.projK]
        values = [v(value) for v in self.projV]

        query = torch.stack(queries, dim=2)
        key = torch.stack(keys, dim=2)
        value = torch.stack(values, dim=2)

        x, _ = attention(query, key, value)

        return self.merge(x.contiguous().view(batch_dim, -1, self.dim*self.num_heads))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, norm: str):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP({'input_dim': feature_dim*2, 
                        'output_dims': [feature_dim*2, feature_dim],
                        'norm': norm})
        #removed
        #nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=2))
    
class AttentionalGNN(nn.Module):
    def __init__(self, params):
        super().__init__()

        layers = params['layers']
        num_layer_reps = params['num_layer_reps']
        d_model = params['d_model']
        num_heads = params['num_heads']
        norm = params['norm']

        layers = layers*num_layer_reps
        
        self.layers = nn.ModuleList([
            AttentionalPropagation(d_model, num_heads, norm)
            for _ in range(len(layers))])
        self.names = layers

    def forward(self, descs_0, descs_1):
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src_0 = descs_1
                src_1 = descs_0
            elif name == 'self':
                src_0 = descs_0
                src_1 = descs_1
            else:
                raise RuntimeError('Illegal name: ' + name)

            delta_0 = layer(descs_0, src_0)
            delta_1 = layer(descs_1, src_1)

            descs_0 = (descs_0 + delta_0)
            descs_1 = (descs_1 + delta_1)

        return descs_0, descs_1
###

###descriptor encoder
# class DescriptorEncoder(nn.Module):
#     def __init__(self, params):
#         super(DescriptorEncoder, self).__init__()

#         cnn_params = params['cnn_params']
#         #mlp_params = params['mlp_params']
        
#         self.cnn = CNN(cnn_params)
#         #self.mlp = MLP(mlp_params)
        
#     def forward(self, x):
#         x = self.cnn(x)
#         x = x.reshape((x.shape[0], -1))
#         #x = self.mlp(x)
#         return x
###
    
class Associator(nn.Module):
    def __init__(self, params):
        super(Associator, self).__init__()
        
        gnn_params = params['gnn_params']
        final_mlp_params = {
            'input_dim': gnn_params['d_model'],
            'output_dims': [gnn_params['d_model']],
            'norm': gnn_params['norm']
        }
        
        self.denc = DescriptorEncoder()
        self.kenc = KeypointEncoder()

        self.gnn = AttentionalGNN(gnn_params)
        #self.gnn = GNN(gnn_params)
        self.final_mlp = MLP(final_mlp_params)
        self.feature_scale = gnn_params['d_model']**0.5
        self.sinkhorn_iterations = params['sinkhorn_iterations']

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, box_features, keypoint_vecs, is_tags, scores):
        descs_0, descs_1 = box_features
        kpts_0, kpts_1 = keypoint_vecs
        is_tags_0, is_tags_1 = is_tags
        scores_0, scores_1 = scores

        descs_0 = self.denc(descs_0).squeeze(-1).squeeze(-1)
        descs_1 = self.denc(descs_1).squeeze(-1).squeeze(-1)

        kpts_0 = self.kenc(kpts_0).squeeze(-1).squeeze(-1)
        kpts_1 = self.kenc(kpts_1).squeeze(-1).squeeze(-1)

        descs_0 = descs_0 + kpts_0
        descs_1 = descs_1 + kpts_1

        descs_0 = torch.concatenate([descs_0, is_tags_0.unsqueeze(-1), scores_0.unsqueeze(-1)], dim=1)
        descs_1 = torch.concatenate([descs_1, is_tags_1.unsqueeze(-1), scores_1.unsqueeze(-1)], dim=1)

        descs_0, descs_1 = self.gnn(descs_0.unsqueeze(0), descs_1.unsqueeze(0))

        descs_0 = self.final_mlp(descs_0)
        descs_1 = self.final_mlp(descs_1)

        scores = torch.einsum("bnd,bmd->bnm", descs_0, descs_1)
        scores = scores / self.feature_scale

        scores = log_optimal_transport(scores, self.bin_score, iters=self.sinkhorn_iterations)
        
        return scores.squeeze(0)

