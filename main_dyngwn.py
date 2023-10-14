import torch
import numpy as np
import argparse
import time
import util
import matplotlib.pyplot as plt

import builtins
import random

import os
import gc
import json
from tqdm import tqdm
import optuna
from optuna.samplers import RandomSampler
import timeit

import torch.distributed as dist

from engine import trainer

def parse_args():
    parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type=str, default='cuda', help='')
    parser.add_argument('--data', type=str, default='/home/gridsan/shibal/traffic-data/data/METR-LA', help='data path')
    parser.add_argument('--adjdata', type=str,default='/home/gridsan/shibal/traffic-data/data/METR-LA/adj_mx.pkl', help='adj data path')
    parser.add_argument('--domain', type=str, default='traffic', help='domain')
    parser.add_argument('--n_train', type=int,default=5000, help='number of training samples')
    parser.add_argument('--adjtype', type=str, default='doubletransition', help='adj type')
    parser.add_argument('--gcn_bool', action='store_true', help='whether to add graph convolution layer')
    parser.add_argument('--dynamic_gcn_bool', action='store_true', help='whether to add dynamic graph convolution layer')
    parser.add_argument('--dynamic_supports_len', type=int, default=1, help='num_features')
    parser.add_argument('--dynamic_graph', type=str, default='correlation', help='which graph method to use')
    parser.add_argument('--dynamic_graph_window', type=int, default=48, help='dynamic graphs window')
    parser.add_argument('--dynamic_graph_transform', type=str, default='absolute', help='graph transformation type')
    parser.add_argument('--aptonly', action='store_true', help='whether only adaptive adj')
    parser.add_argument('--addaptadj', action='store_true', help='whether add adaptive adj')
    parser.add_argument('--randomadj', action='store_true', help='whether random initialize adaptive adj')
    parser.add_argument('--input_seq_length', type=int, default=12,help='')
    parser.add_argument('--output_seq_length', type=int, default=12,help='')
    parser.add_argument('--nhid', type=int, default=32,help='')
    parser.add_argument('--kernel_size', type=int, default=2,help='')
    parser.add_argument('--blocks', type=int, default=4,help='')
    parser.add_argument('--layers', type=int, default=2,help='')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--num_nodes', type=int, default=80, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='')
    parser.add_argument('--print_every', type=int, default=50, help='')
    #parser.add_argument('--seed',type=int,default=99,help='random seed')
    parser.add_argument('--save_directory', type=str, default='./logs/dyngwn', help='save path')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--tuning_seed', type=int, default=0, help='random seed for tuning')
    parser.add_argument('--n_trials', dest='n_trials',  type=int, default=1)

    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--gpu', default=None, type=int)
    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')

    args = parser.parse_args()
    return args

def main(args):
    ################### Distributed Training #######################
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    else:
        args.world_size = 1
    print("args.world_size:", args.world_size)
    args.distributed = args.world_size > 1
#     ngpus_per_node = torch.cuda.device_count()


    if args.distributed:
        if args.local_rank != -1: # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
            print("Running torch.distributed.run")
        elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
            args.rank = int(os.environ['SLURM_PROCID'])
            args.gpu = args.rank % torch.cuda.device_count()
            print("Running via Slurm")
            print("args.rank:", args.rank)
            print("args.gpu:", args.gpu)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        torch.backends.cudnn.benchmark = True
        print("Distributed Process Initialized")
        
#         local_rank = args.local_rank
#         local_world_size = os.environ["LOCAL_WORLD_SIZE"]
#         print(f"local_rank={local_rank}, local_world_size={local_world_size}")
#         print(f"rank={args.rank}, gpu={args.gpu}")
        
    else:
        args.rank = 0
        args.gpu=0
        args.batch_size = args.batch_size*4

    # Load Time series and Adjacency Graphs
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype, num_nodes=args.num_nodes)
    data = util.load_dataset(
        args.data,
        args.batch_size,
        args.batch_size,
        args.batch_size,
        num_nodes=args.num_nodes,
        load_dynamic_graphs=args.dynamic_gcn_bool,
        window_size=args.dynamic_graph_window,
        n_train=args.n_train,
        domain=args.domain
    )
    scaler = data['scaler']
    print("===========Dataset Loaded==========")
    supports = [torch.tensor(i, requires_grad=False).to('cpu') for i in adj_mx]

    print(args)

    print(data['x_train'].shape, data['x_val'].shape, data['x_test'].shape)
    print(data['y_train'].shape, data['y_val'].shape, data['y_test'].shape)
    if data['x_A_train'] is not None:
        print(data['x_A_train'].shape, data['x_A_val'].shape, data['x_A_test'].shape)

    print(data.keys())

    # Create logging directory
    path = os.path.join(args.save_directory, args.data.split('/')[-1])
    if args.dynamic_gcn_bool:
        path = os.path.join(path, "Nodes{}".format(args.num_nodes), "Dynamic:True", "N{}".format(args.n_train), "Blocks{}".format(args.blocks), args.dynamic_graph, args.dynamic_graph_transform, "LR{}".format(args.learning_rate))
    else:
        path = os.path.join(path, "Nodes{}".format(args.num_nodes), "Dynamic:False", "N{}".format(args.n_train), "Blocks{}".format(args.blocks), "LR{}".format(args.learning_rate))
    path = os.path.join(path, "{}.{}".format(args.expid, args.tuning_seed))    
    if args.rank == 0:
        os.makedirs(path, exist_ok=True)
        print("====================path created:", path)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None
    
    
    
    ### Graph WaveNet parameters 
    dropout = 0.3
        
    ### Optimization parameters
    learning_rate = args.learning_rate # trial.suggest_loguniform('learning_rate', 1e-4, 1e-2) # 0.001
    weight_decay = 0.0001 # trial.suggest_loguniform('weight_decay', 1e-4, 1e-4) #0.0001
    
    engine = trainer(
        scaler,
        args.in_dim,
        args.input_seq_length,
        args.output_seq_length,
        args.num_nodes,
        args.nhid,
        args.kernel_size,
        args.blocks,
        args.layers,
        dropout,
        learning_rate,
        weight_decay,
        supports,
        args.gcn_bool,
        args.addaptadj,
        adjinit,
        args.dynamic_gcn_bool,
        args.dynamic_supports_len,
        graph_method=args.dynamic_graph,
        transformation=args.dynamic_graph_transform, 
        domain=args.domain
    )
        
    if args.distributed:
#         dist.init_process_group(backend='nccl', init_method='env://',
#                                 world_size=args.world_size, rank=args.rank)
        torch.backends.cudnn.benchmark = True

        if args.rank!=0:
            def print_pass(*args):
                pass
            builtins.print = print_pass

        print("==========Creating Distributed Data Parallel Model=============")
        torch.cuda.set_device(args.gpu)
        engine.model.cuda(args.gpu)
        engine.model = torch.nn.parallel.DistributedDataParallel(engine.model, device_ids=[args.gpu])
        engine.model_without_ddp = engine.model.module
        device = args.gpu
        print("==========Distributed Model Data Parallel Created=============")
    else:
        print("==========Creating Data Parallel Model=============")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         engine.model = torch.nn.DataParallel(engine.model)
        engine.model.to(device)
        print("==========Model Data Parallel Created=============")
#         engine.model = torch.nn.parallel.DistributedDataParallel(engine.model)
#         engine.model_without_ddp = engine.model.module
    
    print('Using device ', device)
    
    print("==========Set Optimizer=============")
    engine.set_optimizer()
    
    # initialize the early_stopping object
#     path = os.path.join(path, "trials", "{}".format(trial.number))
    if args.rank == 0: # only val and save on master node
        print("==========Creating Path=============")
        os.makedirs(path, exist_ok=True)
    checkpoint_file = os.path.join(path, "checkpoint.pt")

    early_stopping=True
    patience=25
        
    if early_stopping:
        early_stopping = EarlyStopping(patience=patience, path=checkpoint_file, verbose=False)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(data['train_dataset'], shuffle=True)
        data['train_loader'] = torch.utils.data.DataLoader(
            data['train_dataset'],
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=train_sampler,
            drop_last=False
        )

        val_sampler = None
        data['val_loader'] = torch.utils.data.DataLoader(
            data['val_dataset'],
            batch_size=8,
            shuffle=(val_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=val_sampler,
            drop_last=False
        )
        test_sampler = None
        data['test_loader'] = torch.utils.data.DataLoader(
            data['test_dataset'],
            batch_size=8,
            shuffle=(test_sampler is None),
            num_workers=args.workers,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False
        )
        torch.backends.cudnn.benchmark = True
    else:
        data['train_loader'] = torch.utils.data.DataLoader(
            data['train_dataset'],
            batch_size=args.batch_size,
            shuffle=True,
        )

        val_sampler = None
        data['val_loader'] = torch.utils.data.DataLoader(
            data['val_dataset'],
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_sampler = None
        data['test_loader'] = torch.utils.data.DataLoader(
            data['test_dataset'],
            batch_size=args.batch_size,
            shuffle=False,
        )
        

    if args.rank == 0: # only val and save on master node
        print("start training...", flush=True)
        his_loss =[]
#     val_time = []
#     train_time = []
    for epoch in range(1,args.epochs+1):
        np.random.seed(epoch)
        random.seed(epoch)
        # fix sampling seed such that each gpu gets different part of dataset
        if args.distributed: 
            data['train_loader'].sampler.set_epoch(epoch)
                
#         t1 = time.time()
        mtrain_loss, mtrain_mape, mtrain_rmse = train_one_epoch(engine, data['train_loader'], epoch, args, device)
#         t2 = time.time()
        
        if engine.scheduler is not None:
            engine.scheduler.step()
            
        if args.rank == 0: # only val and save on master node

#             train_time.append(t2-t1)
            print('Epoch: {:03d}, Train Loss: {:.8f}, Train MAPE: {:.8f}, Train RMSE: {:.8f}'.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse), flush=True)
                
#             s1 = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse = validate(engine, data['val_loader'], device)
#             s2 = time.time()
#             print('Epoch: {:03d}, Inference Time: {:.4f} secs'.format(epoch,(s2-s1)))
#             val_time.append(s2-s1)
            
            his_loss.append(mvalid_loss)

#             print('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)), flush=True)
            print('Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse), flush=True)

            early_stopping(mvalid_loss, engine.model)

#             print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
            number_of_epochs_it_ran = np.argmin(his_loss)
#             print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
            
            #testing
            engine.model.load_state_dict(torch.load(checkpoint_file)['model'])
        #     bestid = np.argmin(his_loss)
        #     engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))

            metrics = {}
    #         # Train metrics
    #         train_metrics = compute_metrics(engine, data['train_loader'], data['y_train'], scaler, name='train')
    #         metrics.update(train_metrics)

            # Valid metrics
            val_metrics = compute_metrics(engine, data['val_loader'], scaler, device, name='val')
            metrics.update(val_metrics)

            # Test metrics
            test_metrics = compute_metrics(engine, data['test_loader'], scaler, device, name='test')
            metrics.update(test_metrics)

#             [trial.set_user_attr(key, met) for key, met in metrics.items()]
#             trial.set_user_attr("epochs", number_of_epochs_it_ran+1)

        if early_stopping.early_stop:
            print("Early stopping")
            break
#         torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
            
    del engine
    gc.collect()
    
#     return np.min(his_loss)
#     torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")


def train_one_epoch(engine, train_loader, epoch, args, device):
    train_loss = []
    train_mape = []
    train_rmse = []
#     t1 = time.time()
    for iter, (x, y, A) in enumerate(train_loader):
        trainx = x.to(device)
        trainx= trainx.transpose(1, 3)
        trainy = y.to(device)
        trainy = trainy.transpose(1, 3)
        if A is not None:
            trainA = A.to(device)
            trainA= trainA.transpose(1, 3)
        else:
            trainA = None
        metrics = engine.train(trainx, trainy[:,0,:,:], graph_input=trainA)
        del trainx
        del trainy
        if A is not None:
            del trainA
        torch.cuda.empty_cache()
        print("Cuda memory after forward:", torch.cuda.memory_reserved("cuda"))
        train_loss.append(metrics[0])
        train_mape.append(metrics[1])
        train_rmse.append(metrics[2])
        if args.rank == 0: 
            if iter % args.print_every == 0 :
                print('Iter: {:03d}, Train Loss: {:.8f}, Train MAPE: {:.8f}, Train RMSE: {:.8f}'.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)

#     t2 = time.time()
    mtrain_loss = np.mean(train_loss)
    mtrain_mape = np.mean(train_mape)
    mtrain_rmse = np.mean(train_rmse)
#     train_time.append(t2-t1)
        
    return mtrain_loss, mtrain_mape, mtrain_rmse
    

def validate(engine, val_loader, device):
    valid_loss = []
    valid_mape = []
    valid_rmse = []
    for iter, (x, y, A) in enumerate(val_loader):
        testx = x.to(device)
        testx = testx.transpose(1, 3)
        testy = y.to(device)
        testy = testy.transpose(1, 3)
        if A is not None:
            testA = A.to(device)
            testA= testA.transpose(1, 3)
        else:
            testA = None
        metrics = engine.eval(testx, testy[:,0,:,:], graph_input=testA)
        del testx
        del testy
        if A is not None:
            del testA
        torch.cuda.empty_cache()
        print("Cuda memory after forward:", torch.cuda.memory_reserved("cuda"))
        valid_loss.append(metrics[0])
        valid_mape.append(metrics[1])
        valid_rmse.append(metrics[2])
        if iter % args.print_every == 0 :
            print('Iter: {:03d}, Valid Loss: {:.8f}, Valid MAPE: {:.8f}, Valid RMSE: {:.8f}'.format(iter, valid_loss[-1], valid_mape[-1], valid_rmse[-1]), flush=True)
    mvalid_loss = np.mean(valid_loss)
    mvalid_mape = np.mean(valid_mape)
    mvalid_rmse = np.mean(valid_rmse)
    return mvalid_loss, mvalid_mape, mvalid_rmse

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         torch.save(backbone.state_dict(), self.path)
        torch.save(
            {
                'model': model.state_dict(),
                'best_val_loss': self.val_loss_min
            },
            self.path
        )
        self.val_loss_min = val_loss
        
def compute_metrics(engine, test_data, scaler, device, name='test'):
    outputs = []
    realoutputs = []

#     for iter, (x, y, A) in enumerate(test_data.get_iterator()):
    for iter, (x, y, A) in enumerate(test_data):
        testx = x.to(device)
        testx = testx.transpose(1,3)
        testy = y.to(device)
        testy = testy.transpose(1,3)
        if A is not None:
            testA = A.to(device)
            testA = testA.transpose(1,3)
        else:
            testA = None
            
        with torch.no_grad():
#             testx = nn.functional.pad(testx,(1,0,0,0))
#             if A is not None:
#                 testA = nn.functional.pad(testA,(1,0,0,0))
            if engine.model_without_ddp is not None:
                preds = engine.model_without_ddp(testx, graph_input=testA).transpose(1,3)
            else:
                preds = engine.model(testx, graph_input=testA).transpose(1,3)
#             print("preds.shape:", preds.size())
        outputs.append(preds.squeeze(dim=1))
        realoutputs.append(testy[:,0,:,:])
        del testx
        del testy
        if A is not None:
            del testA
        torch.cuda.empty_cache()
        print("Cuda memory after forward:", torch.cuda.memory_reserved("cuda"))

    realy = torch.cat(realoutputs, dim=0)
    yhat = torch.cat(outputs, dim=0)
    print("realy.shape:", realy.size())
    print("yhat.shape:", yhat.size())
    yhat = yhat[:realy.size(0),...]
    print("yhat.shape:", yhat.size())

    amae = {}
    amape = {}
    armse = {}
    all_metrics = {}
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:,:,i])
        real = realy[:,:,i]
        if engine.domain in ['traffic']:
            metrics = util.metric(pred, real, 0.0)
        elif engine.domain in ['stocks', 'exchange']:
            metrics = util.metric(pred, real, np.nan)
        log = 'Evaluate best model on {} data for horizon {:d}, {} MAE: {:.4f}, {} MAPE: {:.4f}, {} RMSE: {:.4f}'
        print(log.format(name, i+1, name, metrics[0], name, metrics[1], name, metrics[2]))
        amae[name+'-mae-{}'.format(i)] = metrics[0]
        amape[name+'-mape-{}'.format(i)] = metrics[1]
        armse[name+'-rmse-{}'.format(i)] = metrics[2]
    all_metrics.update(amae)
    all_metrics.update(amape)
    all_metrics.update(armse)
    log = 'On average over 12 horizons, {} MAE: {:.4f}, {} MAPE: {:.4f}, {} RMSE: {:.4f}'
    print(log.format(name, np.mean(list(amae.values())),name, np.mean(list(amape.values())),name, np.mean(list(armse.values()))))
    return all_metrics

if __name__ == '__main__':
    args = parse_args()
    main(args)