
import math
import numpy as np
import torch

import sys
import os
import random

import click
import logging

from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score
from sklearn.model_selection import train_test_split

def subsampled_indices_random(subsample_rate, num_models, len_dataset):
    """
    Describe: Generate subsampling indices for each subnetwork
    Parameter: 
         subsample_rate: percentage of trainng data that each subnetwork sees
         num_networks: number of models in the ensemble model
         len_dataset: the length of the dataset
    Return:
         a list of numpy arrays, each array indices the training sample that a sub
         -network should see
    """
    train_subsampled_list = []
    test_subsampled_list = []
    #create a index array - subsampling from this index array
    #draw samples i.i.d for # of networks
    total_indices = np.arange(len_dataset)
    for i in range(num_models):
        train_indices, test_indices = train_test_split(total_indices, 
                                                       train_size = subsample_rate,
                                                       shuffle = True)
        train_subsampled_list.append(train_indices)
        test_subsampled_list.append(test_indices)
    return train_subsampled_list, test_subsampled_list


def subsampled_indices(subsample_rate, num_models, len_dataset):
    """
    Describe: Generate subsampling indices for each subnetwork
    Parameter: 
         subsample_rate: percentage of trainng data that each subnetwork sees
         num_networks: number of models in the ensemble model
         len_dataset: the length of the dataset
    Return:
         a list of numpy arrays, each array indices the training sample that a sub
         -network should see
    """
    total_indices = np.arange(len_dataset)
    np.random.shuffle(total_indices)
    indices_chunks = []
    num_test_workers = math.ceil(len_dataset/ int((1- subsample_rate) * len_dataset))
    num_testing_per_model = int((1-subsample_rate)*len_dataset) 
    assert num_models >= num_test_workers
    
    train_subsampled_list = []
    test_subsampled_list = []

    num_training_per_model = len_dataset - num_testing_per_model
    for i in range(num_test_workers):
        test_subsampled_list.append(total_indices[i*num_testing_per_model : (i+1) * num_testing_per_model])
        subsample_size = test_subsampled_list[-1].shape[0]
        if i == num_test_workers -1 and subsample_size < num_testing_per_model:
            test_subsampled_list[-1] = np.concatenate(( 
            test_subsampled_list[-1], total_indices[0: num_testing_per_model - subsample_size]))
            train_subsampled_list.append(total_indices[num_testing_per_model - subsample_size: i*num_testing_per_model])
        else:
            train_subsampled_list.append(
            np.concatenate((total_indices[0:i*num_testing_per_model], total_indices[(i+1) * num_testing_per_model :])))
    
    num_workers = num_test_workers             
    for i in range(num_models - num_workers):
        train_indices, test_indices = train_test_split(total_indices, 
                                                       train_size = num_training_per_model,
                                                       shuffle = True)
        train_subsampled_list.append(train_indices)
        test_subsampled_list.append(test_indices)
    return train_subsampled_list, test_subsampled_list  

    

