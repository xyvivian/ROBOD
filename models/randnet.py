import numpy as np
import sys
sys.path.append("..")
import os
import random
from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import torch.optim as optim
import torch.nn.utils.prune as prune
import scipy.io as sio
from torch.utils.data import TensorDataset
from scipy.io import arff
import pandas as pd
from networks.masked_AE import MaskLinearAE,MaskConvAE
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm



def turned_off_indices(layer,total_dim):
    """
    Turn off the indices from param_list, used for pre-training only
    """
    param_list = [False]*total_dim*2
    layer_list = list(set([layer*2, layer*2+1, total_dim*2 - layer*2 - 1, total_dim*2 - layer*2 - 2]))
    for idx in layer_list:
        param_list[idx] = True
    return param_list


    
    
class LinearRandNet():
    def __init__(self,
                 learning_rate = 0.001,
                 epochs = 250, 
                 num_models = 10, 
                 weight_decay = 0.0,
                 dropout = 0.2,
                 input_dim_list = [784,400,200],
                 device = "cuda",
                 pre_epochs = 100,
                 batch_size = 250):
        
        self.ensemble = []
        self.lr = learning_rate
        self.epochs = epochs
        self.num_models = num_models
        self.dropout = dropout
        self.input_dim_list = input_dim_list
        self.device = device
        self.pre_epochs = pre_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.input_dim_list = input_dim_list
        self.output_dim_list = input_dim_list[::-1][1:]
        self.total_dim = len(self.input_dim_list) + len(self.output_dim_list) -1
        self.ensemble = []
        
    def fit(self,train_data):

        start_time = time.time()
        loader = CustomizeDataLoader(data = train_data ,
                                     label = None,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        
        for model_num in range(self.num_models):
            AE_model = MaskLinearAE(self.input_dim_list,
                                    self.dropout,
                                    self.device,
                                    param_lst = [])
            criterion = nn.MSELoss()
            AE_model = AE_model.to(self.device)
            optimizer = optim.RMSprop(AE_model.parameters(), weight_decay = self.weight_decay, lr = self.lr)
            old_param_list = []

            n_layer = int(self.total_dim /2)
            for i in range(n_layer):
                print("training the %d outer layers" %i)
                param_list = turned_off_indices(i, self.total_dim) 
                for count,param in enumerate(AE_model.parameters()):
                    param.requires_grad = param_list[count]
                    if count in old_param_list:
                        param.grad = None
       
                for epoch in range(self.pre_epochs):
                    for idx in range(total_batches):                
                        batch_index, data = loader.get_next_batch(idx)
                        optimizer.zero_grad()
                        output = AE_model.pretrain(data,i)
                        loss = criterion(output, data)
                        loss.backward()
                        optimizer.step() 
          
                used_params = []
                for idx in range(len(param_list)):
                    if param_list[idx]: 
                        used_params.append(idx)  
                old_param_list = used_params
   
            print("Pre-training finished!!")
            print("Starting to train the model")
    
    
            #reset the AE model parameters to allow gradient updates
            AE_model = AE_model.to(self.device)
            for param in AE_model.parameters():
                param.requires_grad = True
        
            criterion = nn.MSELoss()
            optimizer = optim.RMSprop(AE_model.parameters(), weight_decay = self.weight_decay, lr = self.lr)
            
            for epoch in tqdm(range(self.epochs)):
                runningloss = 0.0
                data_size = 0
                for idx in range(total_batches):                
                    batch_index, data = loader.get_next_batch(idx)             
                    optimizer.zero_grad()
                    output = AE_model(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    runningloss += loss.item()
                    data_size += data.shape[0]
            self.ensemble.append(AE_model)
        
        total_time = time.time() - start_time
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return total_time, memory_allocated, memory_reserved     
        
        
    def predict(self,test_data, test_label):    
        """
        Evaluate the ensemble by finding the median of each feature dimension in each ensemble outputs
        Parameters: ensemble: a list of sub-models 
         """   
        test_result = np.array(np.ones((test_label.shape[0],)) * np.inf)
        loader = CustomizeDataLoader(data = test_data ,
                                     label = None,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        
        # make prediction by each ensemble component
        for idx in range(total_batches):                
            batch_index, data = loader.get_next_batch(idx)
            predictions = [model(data) for model in self.ensemble]
            # SSE based on reconstruction for each component
            reconstruction_loss = np.stack(
            [np.square((pred- data).detach().cpu().numpy()).sum(axis=1) for pred in predictions], axis=1 ) 
   
    
            # scale the std to account for different levels of overfitting
            scaler = StandardScaler(with_mean=False)
            reconstruction_loss = scaler.fit_transform(reconstruction_loss)
    
            # find the median loss for each sample
            median_loss = np.median(reconstruction_loss, axis=1)
            test_result[batch_index] = median_loss
       
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result

    
 
class ConvRandNet():
    def __init__(self,
                 learning_rate = 0.001,
                 epochs = 250, 
                 num_models = 10, 
                 weight_decay = 0.0,
                 dropout = 0.2,
                 device = "cuda",
                 pre_epochs = 100,
                 batch_size = 250,
                 input_dim_list = [3,4,8]):
        
        self.ensemble = []
        self.lr = learning_rate
        self.epochs = epochs
        self.num_models = num_models
        self.dropout = dropout
        self.input_dim_list = input_dim_list
        self.device = device
        self.pre_epochs = pre_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.input_dim_list = input_dim_list
        self.output_dim_list = input_dim_list[::-1][1:]
        self.total_dim = len(self.input_dim_list) + len(self.output_dim_list) -1
        self.ensemble = []
        
    def fit(self,train_data):
        start_time = time.time()
        loader = CustomizeDataLoader(data = train_data ,
                                     label = None,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        
        for model_num in range(self.num_models):
            AE_model = MaskConvAE(self.input_dim_list,
                                    self.dropout,
                                    self.device,
                                    param_lst = [])
            criterion = nn.MSELoss()
            AE_model = AE_model.to(self.device)
            optimizer = optim.RMSprop(AE_model.parameters(), weight_decay = self.weight_decay, lr = self.lr)
               
            for epoch in tqdm(range(self.epochs)):
                runningloss = 0.0
                data_size = 0
                for idx in range(total_batches):                
                    batch_index, data = loader.get_next_batch(idx)             
                    optimizer.zero_grad()
                    output = AE_model(data)
                    loss = criterion(output, data)
                    loss.backward()
                    optimizer.step()
                    runningloss += loss.item()
                    data_size += data.shape[0]
            self.ensemble.append(AE_model)
        
        total_time = time.time() - start_time
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return total_time, memory_allocated, memory_reserved     
        
        
    def predict(self,test_data, test_label):    
        """
        Evaluate the ensemble by finding the median of each feature dimension in each ensemble outputs
        Parameters: ensemble: a list of sub-models 
         """   
        test_result = np.array(np.ones((test_label.shape[0],)) * np.inf)
        loader = CustomizeDataLoader(data = test_data ,
                                     label = None,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        
        # make prediction by each ensemble component
        for idx in range(total_batches):                
            batch_index, data = loader.get_next_batch(idx)
            predictions = [model(data) for model in self.ensemble]
            # SSE based on reconstruction for each component
            reconstruction_loss = np.stack(
            [np.square((pred- data).detach().cpu().numpy()).sum(axis=tuple(range(1, test_data.ndim)))\
             for pred in predictions], axis=1 ) 
   
    
            # scale the std to account for different levels of overfitting
            scaler = StandardScaler(with_mean=False)
            reconstruction_loss = scaler.fit_transform(reconstruction_loss)
    
            # find the median loss for each sample
            median_loss = np.median(reconstruction_loss, axis=1)
            test_result[batch_index] = median_loss
       
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result
