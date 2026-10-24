import torch
import torch.nn.functional as F
import dgl.function as fn
import torch.nn as nn
import dgl
import numpy as np
import math


"""
	Util functions
"""

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-10, 10))}

    return func


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func

def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-10, 10))}

    return func

# Improving implicit attention scores with explicit edge features, if available
def scaling(field, scale_constant):
    def func(edges):
        return {field: (((edges.data[field])) / scale_constant)}

    return func






class GT(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim, num_heads, I_dim, num_gtlayers):
        super().__init__()
        self.h = None
        self.embedding_h = nn.Linear(in_dim, hidden_dim, bias=False)  # node feat is an integer
        self.D_dim = I_dim
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.layergt = GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim)

        # self.layers = nn.ModuleList(
        #     [GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim) for _ in range(num_gtlayers)])
        #
        # self.layers.append(GraphTransformerLayer(hidden_dim, out_dim, num_heads, I_dim))

        self.MLP_layer_x = Reconstruct_X(out_dim, in_dim)

    def extract_features(self, g):

        X = g.ndata['x'].to(torch.float64) # (779,758)
        I = g.ndata['I'].to(torch.float64)  # (779,5)


        h = self.embedding_h(X)

        # for layer in self.layers:
        #     h = layer(h, g, I)
        #     h = F.dropout(h, 0.3, training=self.training)

        h1 = self.layergt(h, g, I)
        h1 = F.dropout(h1, 0.5, training=self.training)
        h1 = self.layergt(h1, g, I)
        h1 = F.dropout(h1, 0.5, training=self.training)

        h2 = self.layergt(h1, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)

        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)
        h2 = self.layergt(h2, g, I)
        h2 = F.dropout(h2, 0.3, training=self.training)


        # h2 = self.layergt(h2, g, I)
        # h2 = F.dropout(h2, 0.3, training=self.training)
        # h2 = self.layergt(h2, g, I)
        # h2 = F.dropout(h2, 0.3, training=self.training)


        return h2


    def forward(self, g):

        h = self.extract_features(g)

        # compute X
        x_hat = self.MLP_layer_x(h)
        self.h = h

        return h, x_hat


class GraphTransformerLayer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads, I_dim):
        super().__init__()

        self.in_channels = in_dim # 16
        self.out_channels = out_dim # 16
        self.num_heads = num_heads # 4

        self.attention = MultiHeadAttentionKernelLayer(in_dim, out_dim // num_heads, num_heads, I_dim)
        
        self.O = nn.Linear(out_dim, out_dim)

        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self.FFN_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_layer2 = nn.Linear(out_dim * 2, out_dim)

        

    def forward(self, h, g, I):
        h_in1 = h  # for first residual connection

        attn_out = self.attention(h, g) # (779,4,32)


        h = attn_out.view(-1, self.out_channels)

        h = F.dropout(h, 0.5, training=self.training) # 779*128




        h = self.O(h)

        h = h_in1 + h  # residual connection

        h = self.layer_norm1(h)

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)

        h = F.dropout(h, 0.5, training=self.training)
        h = self.FFN_layer2(h)
        h = h_in2 + h  # residual connection
        # h = F.dropout(h, 0.3, training=self.training)
        h = self.layer_norm2(h)

        return h

class MultiHeadAttentionKernelLayer(nn.Module):
    def __init__(self, in_dim, head_dim, num_heads, proj_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_out_dim = num_heads * head_dim
        self.Q = nn.Linear(in_dim, self.total_out_dim)
        self.K = nn.Linear(in_dim, self.total_out_dim)
        self.V = nn.Linear(in_dim, self.total_out_dim)
        self.register_buffer('projection_matrix', 
                            nn.init.orthogonal_(torch.randn(proj_dim, self.head_dim)))

    def softmax_kernel_transformation(self, data, is_query):
        data_normalizer = 1.0 / (data.shape[-1] ** 0.25)
        data = data_normalizer * data
        ratio = 1.0 / (self.projection_matrix.shape[0] ** 0.5)
        
        data_dash = torch.einsum("bnht,mt->bnhm", data, self.projection_matrix)
        
        diag_data = torch.sum(data**2, dim=-1, keepdim=True) / 2.0
        if is_query:
            data_dash = ratio * torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True)[0])
        else:
            max_vals = torch.max(torch.max(data_dash, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
            data_dash = ratio * torch.exp(data_dash - diag_data - max_vals)
        return data_dash

    def propagate_attention(self, g):
        g.apply_edges(fn.u_dot_v('Q_hat', 'K_hat', 'score'))
        g.edata['score'] = dgl.ops.edge_softmax(g, g.edata['score'])
        g.update_all(fn.u_mul_e('V_h', 'score', 'm'), fn.sum('m', 'h_out'))

    def forward(self, h, g): 
        
        Q = self.Q(h).view(1, -1, self.num_heads, self.head_dim)  
        K = self.K(h).view(1, -1, self.num_heads, self.head_dim)
        V = self.V(h).view(1, -1, self.num_heads, self.head_dim)
        

        Q_hat = self.softmax_kernel_transformation(Q, is_query=True)  
        K_hat = self.softmax_kernel_transformation(K, is_query=False) 
        
        g.ndata['Q_hat'] = Q_hat.squeeze(0)  
        g.ndata['K_hat'] = K_hat.squeeze(0)
        g.ndata['V_h'] = V.squeeze(0)
        

        self.propagate_attention(g)
        

        h_out = g.ndata['h_out'].view(-1, self.total_out_dim)  # [779, 64]
        return h_out
    


class Reconstruct_X(torch.nn.Module):
    def __init__(self, inp, outp, dims=128):
        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(inp, dims),
            torch.nn.SELU(),
            torch.nn.Linear(dims, outp))

    def forward(self, x):
        x = self.mlp(x)
        return x



class VAE(nn.Module):
    def __init__(self, input_dim, d_model=200, num_classes=2, latent_dim=64, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
        )
        
        self.mu_layer = nn.Linear(d_model // 2, latent_dim)
        self.logvar_layer = nn.Linear(d_model // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim),
        )
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, exp_data):

        batch_size, seq_len, feat_dim = exp_data.shape
        
        # Flatten sequence and encode each time step
        exp_data = exp_data.reshape(batch_size * seq_len, feat_dim)
        #exp_data = exp_data.mean(dim=1) 






        encoded = self.encoder(exp_data)
        
        # Reshape back to sequence and aggregate over time steps
        encoded = encoded.view(batch_size, seq_len, -1)
        encoded = encoded.mean(dim=1)  # Aggregate over time steps
        
        # Get mu and logvar for the aggregated representation
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).unsqueeze(1).expand(-1, seq_len, -1)
        
        return z, recon, mu, logvar




class scCSGRN(nn.Module):
    def __init__(self, in_dim_GT, out_dim_GT, hidden_dim, num_heads, I_dim, in_dim_exp, num_gtlayers, d_model=200, num_classes=2):
        super().__init__()
        self.topo_net = GT(in_dim_GT, out_dim_GT, hidden_dim, num_heads, I_dim, num_gtlayers)
        self.exp_net = VAE(in_dim_exp, d_model=d_model, num_classes=2, dropout=0.1)
        self.pred_layer = nn.Sequential(

            nn.Linear(64+ out_dim_GT + out_dim_GT, 64),
            #nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, g, exp_data, all_tf, all_target, device, return_loss_terms=False):
        
         
        A, A_hat = self.topo_net(g) # A 779*64
        B, exp_recon, mu, logvar = self.exp_net(exp_data)  # B 32*64

        AA = []
        for i in range(len(all_tf)):
            tf = int(all_tf[i])
            target = int(all_target[i])
            feature = torch.cat((A[tf], A[target]), dim=0)
            AA.append(feature)
        AA = torch.stack(AA, dim=0).to(device)            
        
        C = torch.cat((AA, B), dim = 1, out = None)
        
        out = self.pred_layer(C)    
     
        if return_loss_terms:
            return out, {
                'exp_recon': exp_recon,
                'mu': mu,
                'logvar': logvar,
                'gt_recon': A_hat,
                'gt_target': g.ndata['x'].to(device=A_hat.device, dtype=A_hat.dtype),
            }

        return out
















