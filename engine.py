import torch.optim as optim
from model import *
import util
class trainer():
    def __init__(self, scaler, in_dim, input_seq_length, output_seq_length, num_nodes, nhid, kernel_size, blocks, layers, dropout, lrate, wdecay, supports, gcn_bool, addaptadj, aptinit, dynamic_gcn_bool, dynamic_supports_len=1, graph_method='correlation', transformation='absolute', apply_first_order_approx=False, domain='traffic'):
        self.model = dyngwn(num_nodes, dropout, supports=supports, gcn_bool=gcn_bool, addaptadj=addaptadj, aptinit=aptinit, dynamic_gcn_bool=dynamic_gcn_bool, dynamic_supports_len=dynamic_supports_len, in_dim=in_dim, input_sequence_dim=input_seq_length, out_dim=output_seq_length, residual_channels=nhid, dilation_channels=nhid, skip_channels=nhid * 8, end_channels=nhid * 16, kernel_size=kernel_size, blocks=blocks, layers=layers, graph_method=graph_method, transformation=transformation, apply_first_order_approx=apply_first_order_approx)
        self.model_without_ddp = None
#         self.model = torch.nn.DataParallel(self.model)
#         self.model.to(device)
        self.scaler = scaler
        self.lrate = lrate
        self.wdecay = wdecay
        self.domain = domain
        
    def set_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lrate, weight_decay=self.wdecay)
        self.scheduler = None
#         if dynamic_gcn_bool:
#             self.scheduler = None
#         else:
#             self.scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
        self.optimizer = optimizer
        
        self.loss = util.masked_mae
        self.clip = 5
        
        
    def train(self, input, real_val, graph_input=None):
        self.model.train()
        self.optimizer.zero_grad()
#         print("input.shape:", input.shape)
#         input = nn.functional.pad(input,(1,0,0,0))
#         print("input.shape:", input.shape)
#         if graph_input is not None:
#             print("graph_input.shape:", graph_input.shape)
#             graph_input = nn.functional.pad(graph_input,(1,0,0,0))
#             print("graph_input.shape:", graph_input.shape)
            
        
        output = self.model(input, graph_input)
        
#         # use computation graph to find all contributing tensors
#         def get_contributing_params(y, top_level=True):
#             nf = y.grad_fn.next_functions if top_level else y.next_functions
#             for f, _ in nf:
#                 try:
#                     yield f.variable
#                 except AttributeError:
#                     pass  # node has no tensor
#                 if f is not None:
#                     yield from get_contributing_params(f, top_level=False)

#         contributing_parameters = set(get_contributing_params(output))
#         all_parameters = set(net.parameters())
#         non_contributing = all_parameters - contributing_parameters
#         print("All parameters:", len(all_parameters))
#         print("Contributing parameters:", len(contributing_parameters))
#         print("Non-contributing parameters:", len(non_contributing))
#         print("Non-contributing parameters:", non_contributing)

#         print("output.size:", output.size)
        output = output.transpose(1,3)
#         print("output NaNs:", torch.sum(torch.isnan(output)))
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
#         print("predict NaNs:", torch.sum(torch.isnan(predict)))
        
        if self.domain in ['traffic']:
            loss = self.loss(predict, real, 0.0)
        elif self.domain in ['stocks', 'exchange']:
            loss = self.loss(predict, real, np.nan)
        loss.backward()
        
#         for n, p in self.model.named_parameters():
#             if p.grad is None:
#                 print(f'{n} has no grad')
#             else:
#                 print(f'{n} has grad')
        
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        if self.domain in ['traffic']:
            mape = util.masked_mape(predict,real,0.0).item()
            rmse = util.masked_rmse(predict,real,0.0).item()
        elif self.domain in ['stocks', 'exchange']:
            mape = util.masked_mape(predict,real,np.nan).item()
            rmse = util.masked_rmse(predict,real,np.nan).item()
            
        return loss.item(),mape,rmse

    def eval(self, input, real_val, graph_input=None):
        self.model.eval()
#         print("input.shape:", input.shape)
#         input = nn.functional.pad(input,(1,0,0,0))
#         print("input.size:", input.size)
#         if graph_input is not None:
# #             print("graph_input.size:", graph_input)
#             graph_input = nn.functional.pad(graph_input,(1,0,0,0))
# #             print("graph_input.size:", graph_input)
        if self.model_without_ddp is not None:
            output = self.model_without_ddp(input, graph_input)
        else:
            output = self.model(input, graph_input)
#         print("output.size:", output.size)
        output = output.transpose(1,3)
        #output = [batch_size,12,num_nodes,1]
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        if self.domain in ['traffic']:
            loss = self.loss(predict, real, 0.0)
            mape = util.masked_mape(predict,real,0.0).item()
            rmse = util.masked_rmse(predict,real,0.0).item()
        elif self.domain in ['stocks', 'exchange']:
            loss = self.loss(predict, real, np.nan)
            mape = util.masked_mape(predict,real,np.nan).item()
            rmse = util.masked_rmse(predict,real,np.nan).item()
        return loss.item(),mape,rmse
