import numpy as np
import sys
import os
import random
from utils.dataset_generator import generate_data,generate_numpy_data
from sklearn.metrics import roc_auc_score
from networks.robod_net import ROBOD_LinearNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
import time
from tqdm import tqdm
from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data
from utils.subsampling import subsampled_indices



class LinearROBODSub(nn.Module):
    def __init__(self, 
                 lr = 0.001,
                 epochs = 250,
                 num_layer = 3,
                 weight_decay = 0.0,
                 dropout = 0.0,
                 input_dim = 784,
                 input_decay_list = [1.5,1.75,2.0,2.25],
                 optimizer = torch.optim.Adam,
                 batch_size = 256,
                 num_model = 4,
                 device = "cuda",
                 is_masked = False,
                 threshold = 1,
                 subsample_rate = 0.1):
        super(LinearROBODSub, self).__init__()
        
        self.input_dim_list, self.masks = self.create_mask(num_layer = num_layer,
                                                           num_model = num_model,
                                                           input_dim = input_dim,
                                                           input_decay_list = input_decay_list,
                                                           threshold =threshold)
        if not is_masked:
            self.masks = None
        
        self.model = ROBOD_LinearNet(input_dim_list = self.input_dim_list,
                                  num_models = num_model, 
                                  device = device,
                                  bias = True,
                                  dropout = dropout,
                                  is_masked = is_masked,
                                  masks = self.masks)
        
        self.optimizer = optimizer(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.num_model = num_model
        self.subsample_rate = subsample_rate
        
        
    def fit(self,train_data):
        losses = []
        self.model.train()
        self.model = self.model.to(self.device)
        total_time = 0.0
        self.train_subsampled_list, self.test_subsampled_list = \
                 subsampled_indices(self.subsample_rate, self.num_model, train_data.shape[0])
        loader = CustomizeDataLoader(data = train_data ,
                                             label = None,
                                             subsampled_indices = self.train_subsampled_list,
                                             num_models = self.num_model,
                                             batch_size= self.batch_size,
                                             device = self.device) 
        total_batches = loader.num_total_batches()
        for epoch in tqdm(range(self.epochs)):
            epoch_loss = 0.0
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                start_time = time.time()
                self.optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.model(data)
                #print(outputs)
                loss_lst = []
                for i in range(len(self.input_dim_list) -1):
                    scores = torch.sum((outputs[i] - data) ** 2, dim=tuple(range(1, outputs[i].dim())))
                    loss = torch.mean(scores)
                    loss_lst.append(loss)
                
                loss_list = torch.stack(loss_lst)
                loss = loss_list.sum(dim=0)
                epoch_loss += loss.item() / data.shape[0]
                loss.backward()
                self.optimizer.step()
                end_time  = time.time() - start_time
                total_time += end_time
            losses.append(epoch_loss / total_batches)
            
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return total_time, memory_allocated, memory_reserved
                
        
    def predict(self,test_data, test_label = None):
        """
        test_data should be the same as train_data.
        Subsampling doesnot support inference
        """
        self.model.eval()
        test_result = {}
        for i in range(test_data.shape[0]):
            test_result[i] = []
        
        with torch.no_grad():
            loader = CustomizeDataLoader(data = test_data ,
                                             label = None,
                                             subsampled_indices = self.test_subsampled_list,
                                             num_models = self.num_model,
                                             batch_size= self.batch_size,
                                             device = self.device) 
            total_batches = loader.num_total_batches()
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                data = data.to(self.device) 
                outputs = self.model(data)
                recons_list = []
                for i in range(len(self.input_dim_list) -1):
                    scores = torch.sum((outputs[i] - data) ** 2, dim=tuple(range(1, outputs[i].dim())))
                    reconstruction_loss = scores.detach().cpu().numpy()
                    recons_list.append(reconstruction_loss)
                if len(self.input_dim_list) > 2:
                    reconstruction_loss = np.sum(recons_list,axis=0) 
                for i, batch_i in enumerate(batch_index):
                    test_result[batch_i].append(reconstruction_loss[i])
        
        #concatenate the results
        #can be median? -> or median along axis
        mean_test_result = []
        count = 0
        for _,v in test_result.items():
            np.mean(v)
            mean_test_result.append(np.mean(v))
            
        outputs = np.array(mean_test_result)
        if test_label is not None:
            roc_score = roc_auc_score(test_label, outputs)
            print("ROCAUC score: ", roc_score )
        return outputs
    
    
    def create_mask(self,
                    num_layer,
                    num_model,
                    input_dim,
                    input_decay_list, 
                    threshold =1):
        """
        Parameters: num_layer: int, the number of layers in the ROBOD model
                num_models: int, the number of models in the ROBOD model
                input_dim: int, first input dim
                input_decay_list: how # of nodes in the next layer should change with 
                                  respect to the previous layer. For example, if the input
                                  has 784 dim and input_decay = 1.5, then the next layer
                                  should have int(784/1.5) = 522 nodes. 
                                  input_decay_list[0] determines the maximum # of nodes in the next
                                  input_decay_list[i] determines how many nodes the ith submodel in 
                                  the ROBOD should have. For example, input_decay_list[i] = 2.25,
                                  then ith submodel should have int(784/2.25) = 346 nodes. 
                                  Thus, 522 - 346 = 176 nodes hould be masked out,
                                  so we implicitly have a 346-node layer for ith submodel.
                threshold: the minimum # of nodes a layer can have. Default = 1.
        Return: 1. input_dim_list
                2. mask list
        """
        #initialize the largest input dim 
        largest_dim_list = []
        for i in range(num_layer):
            next_input = int(input_dim / (input_decay_list[0]**i))
            largest_dim_list.append(next_input) if next_input > threshold \
            else largest_dim_list.append(threshold)
    
        #find other sub-models' largest dims
        total_dims = []
        for model in range(num_model):
            dim_list = []
            for i in range(num_layer):
                next_input = int(input_dim/ (input_decay_list[model]**i))
                dim_list.append(next_input) if next_input > threshold \
                else dim_list.append(threshold)
            total_dims.append(dim_list)
    
        #initialize the mask
        masks = np.zeros((num_model,num_layer-1), dtype= int)
        for i,inputs in enumerate(total_dims):
            masks[i] = (np.array(largest_dim_list) - np.array(inputs))[1:]
        mask_list = []
        for i in range(masks.shape[1]):
            mask_list.append(masks[:,i].tolist())
    
        return largest_dim_list, mask_list