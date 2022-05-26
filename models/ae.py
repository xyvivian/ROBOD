import numpy as np
import os
from tqdm import tqdm
import time
import math
import sys

sys.path.append("..")
from networks.ae import  LinearAE,ConvAE
from utils.shrink import l21shrink, l1shrink

from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import  roc_auc_score


class AEModel():
    def __init__(self, 
                 input_dim_list, 
                 learning_rate=1e-4, 
                 epochs = 250,
                 batch_size = 256, 
                 device = "cuda", 
                 dropout = 0.2,
                 weight_decay = 0.0):
        """
        Description: Autoencdoer model with LinearAE
        Parameters:
        input_dim_list: list: number of nodes at each level for the internal Autoencoder
        learning_rate: float, learning rate for the internal Autoencoder
        epochs: int, training iterations
        batch_size: int, batch size for Autoencoder
        device: str,default device = "cuda" (gpu) , can use "cpu"
        dropout: float, dropout for LinearAE
        weight_decay: float, regularization in optimizer
        """
        self.input_dim_list = input_dim_list
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device 
        self.dropout = dropout
        self.weight_decay = weight_decay

        #set up an internal auto encoder and the optimizer
        self.AE = LinearAE(input_dim_list = self.input_dim_list, 
                           symmetry = False, 
                           device = self.device,
                           dropout = self.dropout,
                           bias = True)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), \
                                          lr=self.learning_rate, weight_decay = self.weight_decay)
        
            
    def fit(self, train_X: np.ndarray):
        """
        X: all of the training data
        """
        total_time = 0.0
        loss_list = []
        loader = CustomizeDataLoader(data = train_X ,
                                             label = None,
                                             num_models = 1,
                                             batch_size= self.batch_size,
                                             device = self.device)
        total_batches = loader.num_total_batches()
        self.AE = self.AE.to(self.device)
        for it in tqdm(range(self.epochs)):
            final_loss = 0.0
            start_time = time.time()
            epoch_loss = 0.0
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                output = self.AE(data)
                loss = F.mse_loss(output, data)
                epoch_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss/ (total_batches+1)
            final_loss = epoch_loss
            loss_list.append(final_loss.item())            
            end_time = time.time()- start_time
            total_time += end_time
       
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_list, total_time, memory_allocated, memory_reserved     
            
    def predict(self, test_X:np.ndarray, test_label= None):
        test_result = np.array(np.ones((test_label.shape[0],)) * np.inf)
        loader = CustomizeDataLoader(data = test_X,
                                     label = test_label,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        for idx in range(total_batches):
            batch_index, data, labels = loader.get_next_batch(idx)
            prediction = self.AE(data)
            reconstruction_loss = np.mean(np.square((prediction- data).detach().cpu().numpy()), axis = 1)
            if reconstruction_loss.shape != batch_index.shape:
                reconstruction_loss = np.expand_dims(reconstruction_loss, axis=-1)
            #print(batch_index.shape, reconstruction_loss.shape,test_result.shape)
            test_result[batch_index] = reconstruction_loss
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result
    
    
    
class ConvAEModel():
    def __init__(self, 
                 input_dim_list, 
                 learning_rate=1e-4, 
                 epochs = 250,
                 batch_size = 256, 
                 device = "cuda", 
                 dropout = 0.2,
                 weight_decay = 0.0):
        """
        Description: Autoencdoer model with LinearAE
        Parameters:
        input_dim_list: list: number of nodes at each level for the internal Autoencoder
        learning_rate: float, learning rate for the internal Autoencoder
        epochs: int, training iterations
        batch_size: int, batch size for Autoencoder
        device: str,default device = "cuda" (gpu) , can use "cpu"
        dropout: float, dropout for LinearAE
        weight_decay: float, regularization in optimizer
        """
        self.input_dim_list = input_dim_list
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device 
        self.dropout = dropout
        self.weight_decay = weight_decay

        #set up an internal auto encoder and the optimizer
        self.AE = ConvAE(input_dim_list = self.input_dim_list, 
                         device = self.device)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), \
                                          lr=self.learning_rate, weight_decay = self.weight_decay)
        
        
    def fit(self, train_X: np.ndarray):
        """
        X: all of the training data
        """
        total_time = 0.0
        loss_list = []
        loader = CustomizeDataLoader(data = train_X ,
                                             label = None,
                                             num_models = 1,
                                             batch_size= self.batch_size,
                                             device = self.device)
        total_batches = loader.num_total_batches()
        self.AE = self.AE.to(self.device)
        for it in tqdm(range(self.epochs)):
            final_loss = 0.0
            epoch_loss = 0.0
            start_time = time.time()
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                output = self.AE(data)
                loss = F.mse_loss(output, data)
                epoch_loss += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            epoch_loss = epoch_loss/ (total_batches+1)
            final_loss = epoch_loss
            loss_list.append(final_loss.item())            
            end_time = time.time()- start_time
            total_time += end_time
       
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_list, total_time, memory_allocated, memory_reserved     
            
    def predict(self, test_X:np.ndarray, test_label= None):
        test_result = np.array(np.ones((test_label.shape[0],)) * np.inf)
        loader = CustomizeDataLoader(data = test_X,
                                     label = test_label,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        for idx in range(total_batches):
            batch_index, data, labels = loader.get_next_batch(idx)
            prediction = self.AE(data)
            reconstruction_loss = np.mean(np.square((prediction- data).detach().cpu().numpy()), axis=tuple(range(1, test_X.ndim)))
            if reconstruction_loss.shape != batch_index.shape:
                reconstruction_loss = np.expand_dims(reconstruction_loss, axis=-1)
            test_result[batch_index] = reconstruction_loss
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result