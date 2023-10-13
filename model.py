import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np

import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.special import softmax
from tqdm.notebook import tqdm
from sklearn.covariance import shrunk_covariance, ledoit_wolf
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

def compute_partial_correlation(precision):
    """Estimates partial correlation from precision.

    Args:
        Y: target responses, a float numpy array of shape (T, N).
        f, forecast responses,  a float numpy array of shape (T, N).
        sample_size: number of firms in the financial networks dictates sample_size, str.
            - 'small', it may be reasonable to select small when num_time_samples < num_firms.
            - 'large'
    Returns:
        rho: pre-estimatore for rho, a float numpy array of shape (N, N).
    """
    rho = np.zeros_like(precision)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            rho[i,j] = -precision[i, j] / ((precision[i, i] * precision[j, j])**0.5)
    return rho


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            device = x.get_device()
            device = torch.device(device if device>=0 else "cpu")
            a = a.to(device)
#             print("x.get_device():", x.get_device())
#             print("a.get_device():", a.get_device())
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
#                 print("x1.get_device():", x1.get_device())
#                 print("a.get_device():", a.get_device())
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class dynamic_nconv(nn.Module):
    def __init__(self):
        super(dynamic_nconv, self).__init__()

    def forward(self, x, A):
        
        # x.shape = [batch_size, c_in, n, n, timesteps]
        # Batched tensor multiplication x_dot = x * ker: [batch_size, timesteps, n, n, c_in] x [batch_size, timesteps, n, c_in] -> [batch_size, timesteps, n, c_in)
#         x_dot = tf.einsum('bijkl,bikl->bijl', graph_kernel, x)
#         x.shape: [batch_size, c_in, n, timesteps], A.shape: [n, n] 
#         x = torch.einsum('ncvl,vw->ncwl', (x, A))
        
        #  x.shape: [batch_size, c_in, n, timesteps] 
        #  A.shape: [batch_size, c_in, n, n, timesteps]
        x = torch.einsum('bijl,bijkl->bikl', (x, A))
        return x.contiguous()
    
class dynamic_gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, num_nodes, dynamic_supports_len=1, order=2):
        super(dynamic_gcn, self).__init__()
        self.dynamic_nconv = dynamic_nconv()
        c_in = (order*dynamic_supports_len)*c_in
#         print("c_in:", c_in, "c_out:", c_out)
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.dynamic_supports_len = dynamic_supports_len
        self.num_nodes = num_nodes
        self.order = order

    def forward(self, x, support):
        out = []
        for a in support:
            x1 = self.dynamic_nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.dynamic_nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

        # support.shape: (batch_size, c_out, num_nodes*num_nodes, time-steps)
#         support = support.view(support.size(0), support.size(1), self.num_nodes, self.num_nodes, support.size(3))
#         support = torch.permute(support, (0, 4, 2, 3, 1)) # (batch_size, time-steps, num_nodes*num_nodes, c_out)
#         support = support.view(-1, self.num_nodes, self.num_nodes, graph_filter.shape[-1])  
#         print("support.shape:", support.shape)
#         supports = [torch.nn.ReLU()(support)]
#         supports = [support]
#         supports = [torch.nn.Softmax(dim=2)(support), torch.nn.Softmax(dim=2)(torch.permute(support, (0, 1, 3, 2, 4)))]
#         out = [self.dynamic_nconv(x, torch.nn.ReLU()(support))]
#         for a in supports:
#             x1 = self.dynamic_nconv(x, a)
#             out.append(x1)

#         h = torch.cat(out, dim=1)
#         print("h.shape:", h.shape)
#         support = torch.nn.Softmax(dim=2)(support)
#         h = self.dynamic_nconv(x, support)
    

class dyngwn(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, dynamic_gcn_bool=False, dynamic_supports_len=1, in_dim=2, input_sequence_dim=12, out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2, graph_method='correlation', transformation='absolute', apply_first_order_approx=False):
        super(dyngwn, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        self.input_sequence_dim = input_sequence_dim


        self.start_conv = nn.Conv2d(
            in_channels=in_dim,
            out_channels=residual_channels,
            kernel_size=(1,1)
        )
        self.supports = supports
        self.dynamic_gcn_bool = dynamic_gcn_bool

        
        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)
        
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        if gcn_bool and addaptadj:
#             print("gcn_bool:", gcn_bool)
#             print("addaptadj:", addaptadj)
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
                self.supports_len += 1

        if self.gcn_bool and self.supports is not None:
            self.gconv = nn.ModuleList()
        else:
            self.residual_convs = nn.ModuleList()

        if self.dynamic_gcn_bool:
            self.dynamic_graph_filter_convs = nn.ModuleList()
            self.dynamic_graph_gate_convs = nn.ModuleList()
            self.dynamic_gconv = nn.ModuleList()
            
            self.dynamic_supports_len = dynamic_supports_len            
            self.start_dynamic_graph_conv = nn.Conv2d(
                in_channels=dynamic_supports_len,
                out_channels=residual_channels,
                kernel_size=(1,1)
            )
            
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation
                    )
                )

                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=residual_channels,
                        out_channels=dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation
                    )
                ) # changed from Conv1d to Conv 2d

                # 1x1 convolution for skip connection
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=dilation_channels,
                        out_channels=skip_channels,
                        kernel_size=(1, 1)
                    )
                ) # changed from Conv1d to Conv 2d
                
                if (i+1)*(b+1)-1 < (blocks*layers - 1):
                    self.bn.append(nn.BatchNorm2d(residual_channels))
                    if self.gcn_bool:
                        self.gconv.append(
                            gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len)
                        )
                    else:
                        # 1x1 convolution for residual connection
                        self.residual_convs.append(
                            nn.Conv1d(
                                in_channels=dilation_channels,
                                out_channels=residual_channels,
                                kernel_size=(1, 1)
                            )
                        )

                    if self.dynamic_gcn_bool:
                        self.dynamic_graph_filter_convs.append(
                            nn.Conv2d(
                                in_channels=residual_channels,
                                out_channels=dilation_channels,
                                kernel_size=(1, kernel_size),
                                dilation=new_dilation
                            )
                        )
                        self.dynamic_graph_gate_convs.append(
                            nn.Conv2d(
                                in_channels=residual_channels,
                                out_channels=dilation_channels,
                                kernel_size=(1, kernel_size),
                                dilation=new_dilation
                            )
                        ) # changed from Conv1d to Conv 2d
                        self.dynamic_gconv.append(
                            dynamic_gcn(dilation_channels, residual_channels, dropout, num_nodes, dynamic_supports_len=1)
                        )

                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                print("new_dilation:", new_dilation)
                print("receptive_field:", receptive_field)
                print("additional_scope:", additional_scope)


        self.receptive_field = receptive_field
        print("self.input_sequence_dim-self.receptive_field+1:", self.input_sequence_dim+1-self.receptive_field+1)
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1,self.input_sequence_dim+1-self.receptive_field+1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1), # +1 because of input zero-padding in trainer
                                    bias=True)

        self.graph_method = graph_method
        self.transformation = transformation
        self.apply_first_order_approx = apply_first_order_approx
        
    def graph_input_transform(self, graph_input_batch):
        graph_input_batch= graph_input_batch.transpose(1, 3)
        dtype = graph_input_batch.dtype
        device = graph_input_batch.get_device()
        device = torch.device(device if device>=0 else "cpu")
        graph_input_batch = graph_input_batch.cpu().detach().numpy()
        batch_size, num_past_steps_x, graph_window, n_nodes = graph_input_batch.shape
        x_r = graph_input_batch.reshape(
            batch_size*num_past_steps_x,
            graph_window,
            n_nodes
        )
        if self.graph_method=='correlation':
            dynamic_corr = np.array([np.corrcoef(arr.T) for arr in x_r])  
            dynamic_corr = np.nan_to_num(dynamic_corr, nan=0.0)
            dynamic_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_graph = dynamic_corr
        elif self.graph_method=='shrunk-correlation':
            dynamic_corr = np.array([np.corrcoef(arr.T) for arr in x_r])  
            dynamic_corr = np.nan_to_num(dynamic_corr, nan=0.0)
            dynamic_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_shrunk_corr = np.array([shrunk_covariance(corr, shrinkage=0.1) for corr in dynamic_corr])
            dynamic_graph = dynamic_shrunk_corr
        elif self.graph_method=='ledoitwolf-correlation':
            x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(arr) for arr in x_r])          
            dynamic_lf_corr = np.array([ledoit_wolf(arr)[0] for arr in x_r_transformed])  
    #         dynamic_lf_corr = np.nan_to_num(dynamic_lf_corr, nan=0.0)
            dynamic_lf_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_graph = dynamic_lf_corr
        elif self.graph_method=='precision':
            x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(arr) for arr in x_r])          
            dynamic_lf_corr = np.array([ledoit_wolf(arr)[0] for arr in x_r_transformed])  
    #         dynamic_lf_corr = np.nan_to_num(dynamic_lf_corr, nan=0.0)
            dynamic_lf_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_prec = np.array([np.linalg.inv(corr) for corr in tqdm(dynamic_lf_corr)]) # precision matrices
            dynamic_graph = dynamic_prec
        elif self.graph_method=='partial-correlation':
            x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(arr) for arr in x_r])          
            dynamic_lf_corr = np.array([ledoit_wolf(arr)[0] for arr in x_r_transformed])  
    #         dynamic_lf_corr = np.nan_to_num(dynamic_lf_corr, nan=0.0)
            dynamic_lf_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_prec = np.array([np.linalg.inv(corr) for corr in dynamic_lf_corr]) # precision matrices
            dynamic_partial_corr = np.array([compute_partial_correlation(prec) for prec in dynamic_prec]) # partial correlation matrices
            dynamic_graph = dynamic_partial_corr        
        elif self.graph_method=='cosine-similarity':
            x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=False, with_std=True).fit_transform(arr) for arr in x_r])          
            dynamic_lf_cosine = np.array([cosine_similarity(arr.T) for arr in x_r_transformed])
            dynamic_lf_cosine = np.nan_to_num(dynamic_lf_cosine, nan=0.0)
            dynamic_lf_cosine[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
            dynamic_graph = dynamic_lf_cosine

        # diagonal of corr has to be 1. no variation in column gives NaN. Set it it 0 for adjacency use.
        dynamic_graph[:,np.arange(n_nodes),np.arange(n_nodes)] = 0   
        if self.transformation=='absolute':
            dynamic_graph = np.array([np.abs(gr) for gr in dynamic_graph])
        elif self.transformation=='relu':
            dynamic_graph = np.array([np.maximum(gr, 0) for gr in dynamic_graph])
#                 elif transformation=='relu-softmax':
#                     dynamic_graph = np.array([softmax(np.maximum(gr, 0), axis=1) for gr in tqdm(dynamic_graph)])
        else:
            raise ValueError("Transformation: {} is not supported".format(transformation))

        if self.apply_first_order_approx:
            dynamic_graph = np.array([sym_adj(gr) for gr in dynamic_graph])
        
#         print("dynamic_graph.shape:", dynamic_graph.shape)
        A_batch = dynamic_graph.reshape(
            batch_size,
            num_past_steps_x,
            n_nodes*n_nodes,
            -1
        ) # n_sample, 12, n_nodes*n_nodes, 1
#                     print(x_i.shape, y_i.shape, A_i.shape)
        A_batch = torch.from_numpy(A_batch).to(dtype).to(device)
        A_batch= A_batch.transpose(1, 3)
        A_batch = nn.functional.pad(A_batch,(1,0,0,0))

        return A_batch

    def forward(self, input, graph_input=None):
#         print("input.get_device():", input.get_device())
#         print("input.shape:", input.shape)
        input = nn.functional.pad(input,(1,0,0,0))
        in_len = input.size(3)
        if self.dynamic_gcn_bool:
            g_x = self.graph_input_transform(graph_input)
        if in_len<self.receptive_field:
#             print("=================padded input=================")
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
            g_x = nn.functional.pad(g_x,(self.receptive_field-in_len,0,0,0))
        else:
#             print("=================NOT padded input=================")
            x = input
#             if self.dynamic_gcn_bool:
#                 print("graph_input.get_device():", graph_input.get_device())
#                 print("graph_input.shape:", graph_input.shape)
#                 g_x = self.graph_input_transform(graph_input)
#                 print("graph_input.get_device():", g_x.get_device())
#                 print("graph_input.shape:", g_x.shape)
#         print("x.shape:", x.shape)
        x = self.start_conv(x)
#         print("x.shape:", x.shape)
        
        if self.dynamic_gcn_bool:
#             print("g_x.shape:", g_x.shape)
            g_x = self.start_dynamic_graph_conv(g_x)
#             print("g_x.shape:", g_x.shape)
        
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):
            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
#             print("filter.shape", filter.shape)
            gate = self.gate_convs[i](residual)
#             print("gate.shape", gate.shape)
            gate = torch.sigmoid(gate)
            x = filter * gate
#             print("============i:", i, "x.shape:", x.shape)
            
            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
#                 print("============i:", i, "try")                
                skip = skip[:, :, :,  -s.size(3):]
            except:
#                 print("============i:", i, "except")                
                skip = 0
            skip = s + skip
            
            if i < (self.blocks * self.layers - 1):
                if self.dynamic_gcn_bool:
                    g_filter = self.dynamic_graph_filter_convs[i](g_x) # (batch_size, c_out, num_nodes*num_nodes, time-steps) 
                    g_filter = torch.tanh(g_filter)
    #                 print("x.shape:", x.shape, "g_x.shape:", g_x.shape)
                    g_gate = self.dynamic_graph_gate_convs[i](g_x)
                    g_gate = torch.sigmoid(g_gate)
                    g_x = (g_filter + g_x[:, :, :, -g_filter.size(3):]) * g_gate 
    #                 print("g_x NaNs:", torch.sum(torch.isnan(g_x)))
#                     print("i:", i, "After dyn conv=======x.shape:", x.shape)
#                     print("i:", i, "After dyn conv=======g_x.shape:", g_x.shape)
                    d_g_x = self.dynamic_gconv[i](
                        x,
                        [torch.nn.Softmax(dim=2)(g_x.view(g_x.size(0), g_x.size(1), self.num_nodes, self.num_nodes, g_x.size(3)))]
                    )        

                if self.gcn_bool and self.supports is not None:
                    if self.addaptadj:
                        x = self.gconv[i](x, new_supports)
                    else:
                        x = self.gconv[i](x, self.supports)
#                     print("i:", i, "After gconv ==== x.shape:", x.shape)
                else:
                    x = self.residual_convs[i](x)
#                     print("i:", i, "After residual conv ==== x.shape:", x.shape)
                        
                if self.dynamic_gcn_bool:
                    x = x + d_g_x
            

                x = x + residual[:, :, :, -x.size(3):]


                x = self.bn[i](x)
    #             print("block-shapes:", x.shape)

        x = F.relu(skip)
#         print("output-shapes:", x.shape)
        x = F.relu(self.end_conv_1(x))
#         print("output-shapes:", x.shape)
        x = self.end_conv_2(x)
#         print("output-shapes:", x.shape)
        return x





