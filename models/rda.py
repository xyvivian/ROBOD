import numpy as np
import os
from tqdm import tqdm
import time
import math
import sys

sys.path.append("..")
from networks.ae import  LinearAE,ConvAE
from networks.lenet import MNIST_LeNet_AENoleaky, CIFAR10_LeNet_AENoleaky
from utils.shrink import l21shrink, l1shrink

from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import  roc_auc_score


class LinearRDA():
    def __init__(self, 
                 input_dim_list, 
                 lambda_=5e-6, 
                 symmetry= False,
                 learning_rate=1e-4, 
                 inner_iteration = 30, 
                 iteration = 20,
                 batch_size = 256, 
                 regularization = "l1",
                 device = "cuda", 
                 transductive = True,
                 dropout = 0.2,
                 weight_decay = 0.0):
        """
        Description: Robust Deep Autoencdoer model with LinearAE as the underlying model
        Parameters:
        lambda_: float, controls the level of L1/L21 regularization to separate S and L matrices
        input_dim_list: list: number of nodes at each level for the internal Autoencoder
        learning_rate: float, learning rate for the internal Autoencoder
        inner_iteration: int, autoencoder epochs after separation of L and S
        iteration: int, outer iterations
        batch_size: int, batch size for Autoencoder
        regularization: str, regularization type: default = "l1", optional: "l21"
        device: str,default device = "cuda" (gpu) , can use "cpu"
        transductive: boolean, if the transductive setting is turned on, the prediction becomes L1-norm of the S matrix
        symmetry: boolean, the AE has symmetry structure or not
        dropout: float, dropout for LinearAE
        """
        self.lambda_ = lambda_  
        self.input_dim_list = input_dim_list
        self.learning_rate = learning_rate
        self.inner_iteration = inner_iteration
        self.regularization = regularization
        self.iteration = iteration
        self.batch_size = batch_size
        self.device = device 
        self.transductive = transductive
        self.dropout = dropout
        self.weight_decay = weight_decay

        #set up an internal auto encoder and the optimizer
        self.AE = LinearAE(input_dim_list = self.input_dim_list, 
                           symmetry = symmetry, 
                           device = self.device,
                           dropout = self.dropout,
                           bias = True)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)
        
            
    def fit(self, train_X: np.ndarray):
        """
        X: all of the training data
        """
        total_time = 0.0
        self.L = np.zeros(train_X.shape, dtype = train_X.dtype)
        self.S = np.zeros(train_X.shape, dtype = train_X.dtype)
        
        self.AE = self.AE.to(self.device)
        assert train_X.shape[1] == self.input_dim_list[0]
        loss_list = []
        
        for it in tqdm(range(self.iteration)):
            self.L = train_X - self.S
            #  train for smaller iterations to update the AE
            final_loss = 0.0
            loader = CustomizeDataLoader(data = self.L ,
                                             label = None,
                                             num_models = 1,
                                             batch_size= self.batch_size,
                                             device = self.device)
            total_batches = loader.num_total_batches()
            start_time = time.time()
            for i in range(self.inner_iteration):               
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
            #print("Autoencoder loss: %.15f" % final_loss.item())
            loss_list.append(final_loss.item())

            # get optmized L
            new_L = np.zeros(self.L.shape, dtype = train_X.dtype)
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                output = self.AE(data).cpu().detach().numpy()
                new_L[batch_index] = output
            self.L = new_L

            # alternating projection, now project to S and shrink S
            if self.regularization== "l1":
                self.S = l1shrink(self.lambda_, train_X - self.L)
            elif self.regularization == "l21":
                self.S = l21shrink(self.lambda_, train_X - self.L)
            
            end_time = time.time()- start_time
            total_time += end_time
       
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_list, total_time, memory_allocated, memory_reserved
        
        
            
    def predict(self, test_X:np.ndarray, test_label= None):
        # in the transductive case, we need to acquire the L1 norm of the S matrix
        if self.transductive:
            prediction = np.linalg.norm(self.S,ord=1,axis=1)
            if test_label is not None:
                print("AUROC %.3f" % roc_auc_score(test_label, prediction))
            return prediction
        
        #otherwise, we use the internal AE to recover the testing data
        #and we identify the higher reconstruction loss -> more likely the data is from anomaly class
        test_result = np.array(np.ones(test_labels.shape) * np.inf)
        loader = CustomizeDataLoader(data = test_X,
                                     label = test_label,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        for idx in range(total_batches):
            batch_index, data, labels = loader.get_next_batch(idx)
            prediction = self.AE(data)
            reconstruction_loss = np.mean(np.square((output- data).detach().cpu().numpy()), axis = 1)
            test_result[batch_index] = reconstruction_loss
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result
    
    
    
class LeNetRDA():
    def __init__(self, 
                 conv_input_dim_list, 
                 fc_dim = 32,
                 lambda_=5e-6, 
                 symmetry= False,
                 learning_rate=1e-4, 
                 inner_iteration = 30, 
                 iteration = 20,
                 batch_size = 256, 
                 regularization = "l1",
                 device = "cuda", 
                 transductive = True,
                 dataset = "MNIST",
                 weight_decay = 0.0):
        """
        Description: Robust Deep Autoencdoer model with LinearAE as the underlying model
        Parameters:
        lambda_: float, controls the level of L1/L21 regularization to separate S and L matrices
        conv_input_dim_list: list: number of nodes at each level for the internal Autoencoder
        learning_rate: float, learning rate for the internal Autoencoder
        fc_dim: fully connected dimension in autoencoder
        inner_iteration: int, autoencoder epochs after separation of L and S
        iteration: int, outer iterations
        batch_size: int, batch size for Autoencoder
        regularization: str, regularization type: default = "l1", optional: "l21"
        device: str,default device = "cuda" (gpu) , can use "cpu"
        transductive: boolean, if the transductive setting is turned on, the prediction becomes L1-norm of the S matrix
        symmetry: boolean, the AE has symmetry structure or not
        dropout: float, dropout for LinearAE
        """
        self.lambda_ = lambda_  
        self.input_dim_list = conv_input_dim_list
        self.learning_rate = learning_rate
        self.inner_iteration = inner_iteration
        self.regularization = regularization
        self.iteration = iteration
        self.batch_size = batch_size
        self.device = device 
        self.transductive = transductive
        self.weight_decay = weight_decay

        #set up an internal auto encoder and the optimizer
        if dataset == "MNIST":
            self.AE = MNIST_LeNet_AENoleaky(conv_input_dim_list, fc_dim)
        elif dataset == "CIFAR10":
            self.AE = CIFAR10_LeNet_AENoleaky(conv_input_dim_list, fc_dim)
        self.optimizer = torch.optim.Adam(self.AE.parameters(), lr=self.learning_rate, weight_decay = self.weight_decay)   
        
        
    def fit(self, train_X: np.ndarray):
        """
        X: all of the training data
        """
        total_time = 0.0
        self.L = np.zeros((train_X.shape[0],train_X.shape[1]*train_X.shape[2]*train_X.shape[3]), dtype = train_X.dtype)
        self.S = np.zeros((train_X.shape[0],train_X.shape[1]*train_X.shape[2]*train_X.shape[3]), dtype = train_X.dtype)
        
        self.AE = self.AE.to(self.device)
        loss_list = []
        
        for it in tqdm(range(self.iteration)):
            self.L = train_X.reshape((train_X.shape[0],train_X.shape[1]*train_X.shape[2]*train_X.shape[3])) - self.S
            #  train for smaller iterations to update the AE
            final_loss = 0.0
            loader = CustomizeDataLoader(data = self.L.reshape(train_X.shape) ,
                                             label = None,
                                             num_models = 1,
                                             batch_size= self.batch_size,
                                             device = self.device)
            total_batches = loader.num_total_batches()
            start_time = time.time()
            for i in range(self.inner_iteration):               
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
            #print("Autoencoder loss: %.15f" % final_loss.item())
            loss_list.append(final_loss.item())

            # get optmized L
            new_L = np.zeros(self.L.shape, dtype = train_X.dtype)
            for idx in range(total_batches):
                batch_index, data = loader.get_next_batch(idx)
                output = self.AE(data).cpu().detach().numpy()
                new_L[batch_index] = output.reshape(output.shape[0],output.shape[1]*output.shape[2]*output.shape[3])
            self.L = new_L

            # alternating projection, now project to S and shrink S
            if self.regularization== "l1":
                self.S = l1shrink(self.lambda_, \
                                  train_X.reshape((train_X.shape[0], \
                                                   train_X.shape[1]*train_X.shape[2]*train_X.shape[3])) - self.L)
            elif self.regularization == "l21":
                self.S = l21shrink(self.lambda_,\
                                   train_X.reshape((train_X.shape[0], \
                                                    train_X.shape[1]*train_X.shape[2]*train_X.shape[3])) - self.L)
            
            end_time = time.time()- start_time
            total_time += end_time
       
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_list, total_time, memory_allocated, memory_reserved   
    
    def predict(self, test_X:np.ndarray, test_label= None):
        # in the transductive case, we need to acquire the L1 norm of the S matrix
        if self.transductive:
            prediction = np.linalg.norm(self.S,ord=1,axis=1)
            if test_label is not None:
                print("AUROC %.3f" % roc_auc_score(test_label, prediction))
            return prediction
        
        #otherwise, we use the internal AE to recover the testing data
        #and we identify the higher reconstruction loss -> more likely the data is from anomaly class
        test_result = np.array(np.ones(test_labels.shape) * np.inf)
        loader = CustomizeDataLoader(data = test_X,
                                     label = test_label,
                                     num_models = 1,
                                     batch_size= self.batch_size,
                                     device = self.device)
        total_batches = loader.num_total_batches()
        for idx in range(total_batches):
            batch_index, data, labels = loader.get_next_batch(idx)
            prediction = self.AE(data)
            reconstruction_loss = np.mean(np.square((output- data).detach().cpu().numpy()), axis = 1)
            test_result[batch_index] = reconstruction_loss
        if test_label is not None:
            print("AUROC %.3f" % roc_auc_score(test_label, test_result))
        return test_result
              