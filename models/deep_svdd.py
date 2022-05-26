import sys
import os
sys.path.append("..")
import torch
from networks.lenet import MNIST_LeNet, CIFAR10_LeNet_Autoencoder, MNIST_LeNet_Autoencoder, CIFAR10_LeNet
from networks.leaky_ae import LeakyLinearAE,LeakyLinearMLP
import time
import math
import numpy as np
from utils.data_loader import CustomizeDataLoader
from utils.dataset_generator import generate_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score


class ConvDeepSVDD():
    def __init__(self, 
                 conv_input_dim_list = [1,8,4], 
                 fc_dim = 32, 
                 relu_slope = 0.1,
                 pre_train = True, 
                 pre_train_weight_decay = 1e-6, 
                 train_weight_decay = 1e-6,
                 pre_train_epochs = 100, 
                 pre_train_lr = 1e-4, 
                 pre_train_milestones = [0], 
                 train_epochs = 250,
                 train_lr = 1e-4, 
                 train_milestones = [0],
                 batch_size = 250, 
                 device = "cuda", 
                 objective = 'one-class',
                 nu = 0.1,
                 warm_up_num_epochs = 10, 
                 dataset = "MNIST"):
        
        if pre_train:
            if dataset == "MNIST":
                self.ae_net = MNIST_LeNet_Autoencoder(conv_input_dim_list, fc_dim, relu_slope)
            elif dataset == "CIFAR10":
                self.ae_net = CIFAR10_LeNet_Autoencoder(conv_input_dim_list, fc_dim, relu_slope)
        if dataset == "MNIST":
            self.net  = MNIST_LeNet(conv_input_dim_list, fc_dim, relu_slope)
        elif dataset == "CIFAR10":
            self.net  = CIFAR10_LeNet(conv_input_dim_list, fc_dim, relu_slope)
        self.pre_train = pre_train
        self.device = device

        R = 0.0  # hypersphere radius R
        c = None  # hypersphere center c

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=device) if c is not None else None

        #Deep SVDD Hyperparameters
        self.nu = nu
        self.objective = objective
        self.batch_size = batch_size
        self.pre_train_weight_decay = pre_train_weight_decay
        self.train_weight_decay = train_weight_decay
        self.pre_train_epochs = pre_train_epochs
        self.pre_train_lr = pre_train_lr
        self.pre_train_milestones = pre_train_milestones
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.train_milestones = train_milestones
        self.warm_up_n_epochs = warm_up_num_epochs


    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        num_total_batches = train_loader.num_total_batches()
        net.eval()
        with torch.no_grad():
            for idx in range(num_total_batches):
                # get the inputs of the batch
                _,inputs  = train_loader.get_next_batch(idx)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    
    def get_radius(self ,dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)

    
    def fit(self, train_data):
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_data,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        total_time = 0.0
        #pretrain the autoencoder
        if self.pre_train:
            print("Training the autoencoders....")
            self.ae_net = self.ae_net.to(self.device)
            optimizer = optim.Adam(self.ae_net.parameters(), lr=self.pre_train_lr, weight_decay=
                                   self.pre_train_weight_decay,amsgrad=False)
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.pre_train_milestones, gamma=0.1)
            self.ae_net.train()
            
            num_total_batches = dataloader.num_total_batches()
            #training epochs
            
            for epoch in tqdm(range(self.pre_train_epochs)):
                for idx in range(num_total_batches):
                    _, inputs = dataloader.get_next_batch(idx)
                    start_time = time.time()
                    optimizer.zero_grad()
                    outputs = self.ae_net(inputs)
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    loss.backward()
                    optimizer.step()
                    finish_time = time.time()
                    total_time += finish_time - start_time

            #Update the parameters from the autoencoder
            ae_net_dict = self.ae_net.state_dict()
            net_dict = self.net.state_dict()
            # Filter out decoder network keys
            ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
            # Overwrite values in the existing state_dict
            net_dict.update(ae_net_dict)
            # Load the new state_dict
            self.net.load_state_dict(net_dict)

        # Initilalize the net
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay, 
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            self.c = self.init_center_c(dataloader, self.net)
            print('Center c initialized.')

        self.net.train()
        loss_lst = []
        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0
            for idx in range(num_total_batches):
                _, inputs  = dataloader.get_next_batch(idx)
                start_time = time.time()
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)
                loss_epoch += loss.item()
                end_time = time.time()
                total_time += end_time - start_time
            loss_lst.append(loss.detach().cpu().item())
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_lst, total_time, memory_allocated, memory_reserved


    def predict(self, test_data, test_labels):
        #load the overall data into a dataloader, so we can compute the score all together
        dataloader = CustomizeDataLoader(data = test_data,
                                         label = test_labels,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        num_total_batches = dataloader.num_total_batches()
        # Set device for network
        self.net = self.net.to(self.device)
        # Testing
        self.net.eval()
        test_result = np.array(np.ones(test_labels.shape) * np.inf)
        with torch.no_grad():
            for idx in range(num_total_batches):
                test_batch_indices,inputs,labels = dataloader.get_next_batch(idx)
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist 
                for i, batch_i in enumerate(test_batch_indices):
                    test_result[batch_i]= scores[i].detach().cpu().numpy()

            test_auc = roc_auc_score(test_labels, test_result)
            print('Test set AUC: {:.2f}%'.format(100. * test_auc))
        return test_result

class LinearDeepSVDD():
    def __init__(self, 
                 input_dim_list = [784,400], 
                 relu_slope = 0.1,
                 pre_train = True, 
                 pre_train_weight_decay = 1e-6, 
                 train_weight_decay = 1e-6,
                 pre_train_epochs = 100, 
                 pre_train_lr = 1e-4, 
                 pre_train_milestones = [0], 
                 train_epochs = 250,
                 train_lr = 1e-4, 
                 train_milestones = [0],
                 batch_size = 250, 
                 device = "cuda", 
                 objective = 'one-class',
                 nu = 0.1,
                 warm_up_num_epochs = 10,
                 symmetry = False,
                 dropout = 0.2):
        if pre_train:
            self.ae_net = LeakyLinearAE(input_dim_list = input_dim_list,
                                        symmetry = symmetry,
                                        device = device,
                                        dropout = dropout,
                                        negative_slope = relu_slope)
            
        self.net = LeakyLinearMLP(input_dim_list = input_dim_list,
                                  device = device,
                                  dropout = dropout,
                                  negative_slope = relu_slope)
        self.rep_dim = input_dim_list[-1]
        self.pre_train = pre_train
        self.device = device

        R = 0.0  # hypersphere radius R
        c = None  # hypersphere center c

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=device)  # radius R initialized with 0 by default.
        self.c = torch.tensor(c, device=device) if c is not None else None

        #Deep SVDD Hyperparameters
        self.nu = nu
        self.objective = objective
        self.batch_size = batch_size
        self.pre_train_weight_decay = pre_train_weight_decay
        self.train_weight_decay = train_weight_decay
        self.pre_train_epochs = pre_train_epochs
        self.pre_train_lr = pre_train_lr
        self.pre_train_milestones = pre_train_milestones
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.train_milestones = train_milestones
        self.warm_up_n_epochs = warm_up_num_epochs        
               
    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(self.rep_dim, device=self.device)
        num_total_batches = train_loader.num_total_batches()
        net.eval()
        with torch.no_grad():
            for idx in range(num_total_batches):
                # get the inputs of the batch
                _,inputs = train_loader.get_next_batch(idx)
                outputs = net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)
        c /= n_samples
        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c   
    
    def get_radius(self ,dist: torch.Tensor):
        """Optimally solve for radius R via the (1-nu)-quantile of distances."""
        return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu)   
       
    def fit(self, train_X):
        #load the datainto loader
        #train_loader = DataLoader(train_data, batch_size= self.batch_size, num_workers=self.n_jobs_dataloader)
        dataloader = CustomizeDataLoader(data = train_X,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        total_time = 0.0
        #pretrain the autoencoder
        if self.pre_train:
            print("Training the autoencoders....")
            self.ae_net = self.ae_net.to(self.device)
            optimizer = optim.Adam(self.ae_net.parameters(), lr=self.pre_train_lr, weight_decay=
                                   self.pre_train_weight_decay,amsgrad=False)
            #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.pre_train_milestones, gamma=0.1)
            self.ae_net.train()
            
            num_total_batches = dataloader.num_total_batches()
            #training epochs
            
            for epoch in tqdm(range(self.pre_train_epochs)):
                for idx in range(num_total_batches):
                    _, inputs = dataloader.get_next_batch(idx)
                    start_time = time.time()
                    optimizer.zero_grad()
                    outputs = self.ae_net(inputs)
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    loss.backward()
                    optimizer.step()
                    finish_time = time.time()
                    total_time += finish_time - start_time

            #Update the parameters from the autoencoder
            ae_net_dict = self.ae_net.state_dict()
            net_dict = self.net.state_dict()
            # Filter out decoder network keys
            ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
            # Overwrite values in the existing state_dict
            net_dict.update(ae_net_dict)
            # Load the new state_dict
            self.net.load_state_dict(net_dict)

        # Initilalize the net
        self.net = self.net.to(self.device)
        optimizer = optim.Adam(self.net.parameters(),
                               lr=self.train_lr,
                               weight_decay=self.train_weight_decay, 
                               amsgrad=False)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            print('Initializing center c...')
            #self.c = self.init_center_c(dataloader, self.net)
            self.c = torch.zeros(self.rep_dim, device="cuda")
            print('Center c initialized.')

        self.net.train()
        loss_lst = []
        for epoch in tqdm(range(self.train_epochs)):
            loss_epoch = 0.0
            for idx in range(num_total_batches):
                _, inputs = dataloader.get_next_batch(idx)
                start_time = time.time()
                optimizer.zero_grad()
                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                    loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
                else:
                    loss = torch.mean(dist)
                loss.backward()
                optimizer.step()
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs):
                    self.R.data = torch.tensor(self.get_radius(dist, self.nu), device= self.device)
                loss_epoch += loss.item()
                end_time = time.time()
                total_time += end_time - start_time
            loss_lst.append(loss.detach().cpu().item())
        memory_allocated = torch.cuda.max_memory_allocated(self.device) // (1024 ** 2)
        memory_reserved = torch.cuda.max_memory_reserved(self.device) // (1024 ** 2)
        print( f'Memory Peak: {memory_allocated} MB allocated, {memory_reserved} MB reserved.')
        return loss_lst, total_time, memory_allocated, memory_reserved


    def predict(self, test_data, test_labels):
        #load the overall data into a dataloader, so we can compute the score all together
        dataloader = CustomizeDataLoader(data = test_data,
                                         label = test_labels,
                                         num_models = 1,
                                         batch_size = self.batch_size,
                                         device = self.device)
        num_total_batches = dataloader.num_total_batches()
        # Set device for network
        self.net = self.net.to(self.device)
        # Testing
        self.net.eval()
        test_result = np.array(np.ones(test_labels.shape) * np.inf)
        with torch.no_grad():
            for idx in range(num_total_batches):
                test_batch_indices,inputs,labels = dataloader.get_next_batch(idx)
                outputs = self.net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                if self.objective == 'soft-boundary':
                    scores = dist - self.R ** 2
                else:
                    scores = dist 
                for i, batch_i in enumerate(test_batch_indices):
                    test_result[batch_i]= scores[i].detach().cpu().numpy()

            test_auc = roc_auc_score(test_labels, test_result)
            print('Test set AUC: {:.2f}%'.format(100. * test_auc))
        return test_result