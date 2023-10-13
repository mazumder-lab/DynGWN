import pickle
import numpy as np
import pandas as pd
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from scipy.special import softmax
from tqdm.notebook import tqdm
from sklearn.covariance import shrunk_covariance, ledoit_wolf
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity
from numpy.lib.stride_tricks import sliding_window_view

class DataLoader(object):
    def __init__(self, xs, ys, x_As=None, batch_size=64, pad_with_last_sample=True, graph_method='correlation', transformation='absolute', apply_first_order_approx=False):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            if x_As is not None:
                x_A_padding = np.repeat(x_As[-1:], num_padding, axis=0)
                x_As = np.concatenate([x_As, x_A_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.x_As = x_As
        
        self.graph_method = graph_method
        self.transformation = transformation
        self.apply_first_order_approx = apply_first_order_approx

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys
        if self.x_As is not None:
            x_As = self.x_As[permutation]
            self.x_As = x_As

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                if self.x_As is not None:
                    x_r = self.x_As[start_ind: end_ind, ...] # batch_size, 12, window, num_nodes
#                     print(x_r.shape)
                    _, num_past_steps_x, graph_window, n_nodes = x_r.shape
                    x_r = x_r.reshape(
                        self.batch_size*num_past_steps_x,
                        graph_window,
                        n_nodes
                    )
#                     print(x_r.shape)
                    
                    if self.graph_method=='partial-correlation':
                        x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(arr) for arr in x_r])          
                        dynamic_lf_corr = np.array([ledoit_wolf(arr)[0] for arr in x_r_transformed])  
                #         dynamic_lf_corr = np.nan_to_num(dynamic_lf_corr, nan=0.0)
                        dynamic_lf_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
                        dynamic_prec = np.array([np.linalg.inv(corr) for corr in dynamic_lf_corr]) # precision matrices
                        dynamic_partial_corr = np.array([compute_partial_correlation(prec) for prec in dynamic_prec]) # partial correlation matrices
                        dynamic_graph = dynamic_partial_corr
                    else:
                        raise ValueError("Graph Method: {} is not supported".format(self.graph_method))

                    # diagonal of corr has to be 1. no variation in column gives NaN. Set it it 0 for adjacency use.
                    dynamic_graph[:,np.arange(n_nodes),np.arange(n_nodes)] = 0   
                    if self.transformation=='absolute':
                        dynamic_graph = np.array([np.abs(gr) for gr in dynamic_graph])
                    else:
                        raise ValueError("Transformation: {} is not supported".format(transformation))

                    if self.apply_first_order_approx:
                        dynamic_graph = np.array([sym_adj(gr) for gr in dynamic_graph])

#                     dynamic_graph = prepare_graph_sequences(dynamic_graph, 12) # n_sample, 12, n_nodes, n_nodes
#                     dynamic_graphs = np.expand_dims(dynamic_graph, 4) # n_sample, 12, n_nodes, n_nodes, 1  
#                     dynamic_graphs.append(dynamic_graph)
#                     dynamic_graphs = np.concatenate(dynamic_graphs, axis=-1) # n_sample, 12, n_nodes, n_nodes, 1  
#                     print(dynamic_graphs.shape)
#                     dynamic_graphs = dynamic_graphs[-(n_sample-window_size):,:,:,:]            
                    A_i = dynamic_graph.reshape(
                            self.batch_size,
                            num_past_steps_x,
                            n_nodes*n_nodes,
                            1
                        ) # n_sample, 12, n_nodes*n_nodes, 1
#                     print(x_i.shape, y_i.shape, A_i.shape)
                    
                else:
                    A_i = None
                yield (x_i, y_i, A_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype, num_nodes=50):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    
    nodes = np.arange(num_nodes)
    sensor_id_to_ind = {sid: ind for sid, ind in sensor_id_to_ind.items() if ind in nodes}
    sensor_ids = [sid for sid in sensor_ids if sid in sensor_id_to_ind]
    adj_mx = adj_mx[np.ix_(nodes,nodes)]
    
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj

def prepare_graph_sequences(sequences, n_steps):
    """Prepares past-times samples as sequences.
    
    Args:
        sequences: a numpy array of shape (T, N, N)
        n_steps: num of past time steps to use as features/sequences, int scalar.
    
    Returns:
        A: past-time samples as features, a float numpy array of shape (T, n_steps, N, N).
    """
    A = list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_A = sequences[i:end_ix, ...]
        A.append(seq_A)
    return np.array(A)

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


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, num_nodes=50, load_dynamic_graphs=False, window_size=48, n_train=5000, graph_method='correlation', transformation='absolute', apply_first_order_approx=False, domain='traffic'):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_'+category] = cat_data['x'][:,:,:num_nodes,:].astype(np.float32)
        data['y_'+category] = cat_data['y'][:,:,:num_nodes,:].astype(np.float32)
        n_sample, n_steps, n_nodes, n_features = data['x_'+category].shape
        
        if load_dynamic_graphs: 
            x = np.vstack([
                data['x_'+category][0,:-1,:,:].reshape(-1, 1, n_nodes, n_features),
                data['x_'+category][:,-2:-1,:,:]
            ])
            x = np.transpose(x, (0, 3, 2, 1)) # # n_sample, n_features, n_nodes, 1
            x = x[:,:,:,0] # n_sample, n_features, n_nodes
            x_r = sliding_window_view(x[:,0,:], window_shape=[window_size,1]).transpose((0,2,1,3))[:,:,:,0]
            if domain=='traffic':
                x_r = prepare_graph_sequences(x_r, 12) # n_sample, 12, window, n_nodes
            elif domain=='stocks':
                x_r = prepare_graph_sequences(x_r, 50) # n_sample, 12, window, n_nodes
            x_r = x_r[-(n_sample-window_size):,:,:] # n_sample, 12, window, n_nodes
            data['x_A_' + category] = x_r.astype(np.float32)
            
#             sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(os.path.join(dataset_dir, 'adj_mx.pkl'))
#             nodes = np.arange(num_nodes)
#             sensor_id_to_ind = {sid: ind for sid, ind in sensor_id_to_ind.items() if ind in nodes}
#             sensor_ids = [sid for sid in sensor_ids if sid in sensor_id_to_ind]
#             adj_mx = adj_mx[np.ix_(nodes,nodes)]
#             data['A_' + category] = np.tile(
#                 adj_mx.reshape(1,1,-1,1),
#                 (data['x_' + category].shape[0], data['x_' + category].shape[1], 1, 2)
#             )
        else:
            data['x_A_' + category] = None
        data['x_'+category] = data['x_'+category][-(n_sample-window_size):,:,:,:] # to match dynamic setting
        data['y_'+category] = data['y_'+category][-(n_sample-window_size):,:,:,:] # to match dynamic setting
        if category == 'train':
            n_train_total = data['x_'+category].shape[0]
            if n_train > n_train_total:
                n_train = n_train_total
            data['x_'+category] = data['x_'+category][-n_train:,:,:,:] # to reduce training set
            data['y_'+category] = data['y_'+category][-n_train:,:,:,:] # to reduce training set
            if load_dynamic_graphs: 
                data['x_A_'+category] = data['x_A_'+category][-n_train:,:,:] # to reduce training set
            
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    data['x_train'] = torch.tensor(data['x_train'].astype(np.float32))
    data['y_train'] = torch.tensor(data['y_train'].astype(np.float32))
    data['x_A_train'] = torch.tensor(data['x_A_train'].astype(np.float32))
    data['x_val'] = torch.tensor(data['x_val'].astype(np.float32))
    data['y_val'] = torch.tensor(data['y_val'].astype(np.float32))
    data['x_A_val'] = torch.tensor(data['x_A_val'].astype(np.float32))
    data['x_test'] = torch.tensor(data['x_test'].astype(np.float32))
    data['y_test'] = torch.tensor(data['y_test'].astype(np.float32))
    data['x_A_test'] = torch.tensor(data['x_A_test'].astype(np.float32))
    
    data['train_dataset'] = torch.utils.data.TensorDataset(data['x_train'], data['y_train'], data['x_A_train'])
    data['val_dataset'] = torch.utils.data.TensorDataset(data['x_val'], data['y_val'], data['x_A_val'])
    data['test_dataset'] = torch.utils.data.TensorDataset(data['x_test'], data['y_test'], data['x_A_test'])
    
#     data['train_loader'] = torch.utils.data.DataLoader(
#         data['train_dataset'],
#         batch_size=batch_size,
#         shuffle=True
#     )
#     data['val_loader'] = torch.utils.data.DataLoader(
#         data['val_dataset'],
#         batch_size=batch_size,
#         shuffle=True
#     )
#     data['test_loader'] = torch.utils.data.DataLoader(
#         data['test_dataset'],
#         batch_size=batch_size,
#         shuffle=True
#     )        
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], x_As=data['x_A_train'], batch_size=batch_size, graph_method=graph_method, transformation=transformation, apply_first_order_approx=apply_first_order_approx)
#     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], x_As=data['x_A_val'], batch_size=valid_batch_size, graph_method=graph_method, transformation=transformation, apply_first_order_approx=apply_first_order_approx)
#     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], x_As=data['x_A_test'], batch_size=test_batch_size, graph_method=graph_method, transformation=transformation, apply_first_order_approx=apply_first_order_approx)
    data['scaler'] = scaler
    return data


def load_dataset_old(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, num_nodes=50, load_dynamic_graphs=False, window_size=48, graph_method='correlation', transformation='absolute', n_train=5000, apply_first_order_approx=False):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_'+category] = cat_data['x'][:,:,:num_nodes,:].astype(np.float32)
        data['y_'+category] = cat_data['y'][:,:,:num_nodes,:].astype(np.float32)
        n_sample, n_steps, n_nodes, n_features = data['x_'+category].shape
        if load_dynamic_graphs: 
            x = np.vstack([
                data['x_'+category][0,:-1,:,:].reshape(-1, 1, n_nodes, n_features),
                data['x_'+category][:,-2:-1,:,:]
            ])
            x = np.transpose(x, (0, 3, 2, 1)) # # n_sample, n_features, n_nodes, 1
            x = x[:,:,:,0] # n_sample, n_features, n_nodes

            dynamic_graphs = []
            for i in [0]:
                x_r = sliding_window_view(x[:,i,:], window_shape=[window_size,1]).transpose((0,2,1,3))[:,:,:,0]
                if graph_method=='partial-correlation':
                    x_r_transformed = np.array([preprocessing.StandardScaler(with_mean=True, with_std=True).fit_transform(arr) for arr in tqdm(x_r)])          
                    dynamic_lf_corr = np.array([ledoit_wolf(arr)[0] for arr in tqdm(x_r_transformed)])  
            #         dynamic_lf_corr = np.nan_to_num(dynamic_lf_corr, nan=0.0)
                    dynamic_lf_corr[:,np.arange(n_nodes),np.arange(n_nodes)] = 1
                    dynamic_prec = np.array([np.linalg.inv(corr) for corr in tqdm(dynamic_lf_corr)]) # precision matrices
                    dynamic_partial_corr = np.array([compute_partial_correlation(prec) for prec in tqdm(dynamic_prec)]) # partial correlation matrices
                    dynamic_graph = dynamic_partial_corr        
                else:
                    ValueError("Graph Method: {} is not supported".format(graph_method))
        
                # diagonal of corr has to be 1. no variation in column gives NaN. Set it it 0 for adjacency use.
                dynamic_graph[:,np.arange(n_nodes),np.arange(n_nodes)] = 0   
                if transformation=='absolute':
                    dynamic_graph = np.array([np.abs(gr) for gr in tqdm(dynamic_graph)])
                else:
                    raise ValueError("Transformation: {} is not supported".format(transformation))
                
                if apply_first_order_approx:
                    dynamic_graph = np.array([sym_adj(gr) for gr in tqdm(dynamic_graph)])
                
                dynamic_graph = prepare_graph_sequences(dynamic_graph, 12) # n_sample, 12, n_nodes, n_nodes
                dynamic_graph = np.expand_dims(dynamic_graph, 4)
                dynamic_graphs.append(dynamic_graph)
            dynamic_graphs = np.concatenate(dynamic_graphs, axis=-1) # n_sample, 12, n_nodes, n_nodes, 2    
            dynamic_graphs = dynamic_graphs[-(n_sample-window_size):,:,:,:]            
            dynamic_graphs = dynamic_graphs.reshape(
                    dynamic_graphs.shape[0],
                    dynamic_graphs.shape[1],
                    dynamic_graphs.shape[2]*dynamic_graphs.shape[3],
                    dynamic_graphs.shape[4]
                ) # n_sample, 12, n_nodes*n_nodes, 2
            data['A_' + category] = dynamic_graphs
            
#             sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(os.path.join(dataset_dir, 'adj_mx.pkl'))
#             nodes = np.arange(num_nodes)
#             sensor_id_to_ind = {sid: ind for sid, ind in sensor_id_to_ind.items() if ind in nodes}
#             sensor_ids = [sid for sid in sensor_ids if sid in sensor_id_to_ind]
#             adj_mx = adj_mx[np.ix_(nodes,nodes)]
#             data['A_' + category] = np.tile(
#                 adj_mx.reshape(1,1,-1,1),
#                 (data['x_' + category].shape[0], data['x_' + category].shape[1], 1, 2)
#             )
        else:
            data['A_' + category] = None
        data['x_'+category] = data['x_'+category][-(n_sample-window_size):,:,:,:] # to match dynamic setting
        data['y_'+category] = data['y_'+category][-(n_sample-window_size):,:,:,:] # to match dynamic setting
        if category == 'train':
            n_train_total = data['x_'+category].shape[0]
            if n_train > n_train_total:
                n_train = n_train_total
            data['x_'+category] = data['x_'+category][-n_train:,:,:,:] # to reduce training set
            data['y_'+category] = data['y_'+category][-n_train:,:,:,:] # to reduce training set
            if load_dynamic_graphs: 
                data['A_'+category] = data['A_'+category][-n_train:,:,:,:] # to reduce training set
            
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    
    print(data['x_train'].shape, data['y_train'].shape, data['A_train'].shape)
    print(data['x_val'].shape, data['y_val'].shape, data['A_val'].shape)
    print(data['x_test'].shape, data['y_test'].shape, data['A_test'].shape)
    data['x_train'] = torch.tensor(data['x_train'].astype(np.float32))
    data['y_train'] = torch.tensor(data['y_train'].astype(np.float32))
    data['A_train'] = torch.tensor(data['A_train'].astype(np.float32))
    data['x_val'] = torch.tensor(data['x_val'].astype(np.float32))
    data['y_val'] = torch.tensor(data['y_val'].astype(np.float32))
    data['A_val'] = torch.tensor(data['A_val'].astype(np.float32))
    data['x_test'] = torch.tensor(data['x_test'].astype(np.float32))
    data['y_test'] = torch.tensor(data['y_test'].astype(np.float32))
    data['A_test'] = torch.tensor(data['A_test'].astype(np.float32))
    
    data['train_loader'] = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data['x_train'], data['y_train'], data['A_train']),
        batch_size=batch_size,
        shuffle=True
    )
    data['val_loader'] = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data['x_val'], data['y_val'], data['A_val']),
        batch_size=batch_size,
        shuffle=True
    )
    data['test_loader'] = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data['x_test'], data['y_test'], data['A_test']),
        batch_size=batch_size,
        shuffle=True
    )        
#     data['train_loader'] = DataLoader(data['x_train'], data['y_train'], As=data['A_train'], batch_size=batch_size)
#     data['val_loader'] = DataLoader(data['x_val'], data['y_val'], As=data['A_val'], batch_size=valid_batch_size)
#     data['test_loader'] = DataLoader(data['x_test'], data['y_test'], As=data['A_test'], batch_size=test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real, null_val=0.0):
    mae = masked_mae(pred,real,null_val).item()
    mape = masked_mape(pred,real,null_val).item()
    rmse = masked_rmse(pred,real,null_val).item()
    return mae,mape,rmse


