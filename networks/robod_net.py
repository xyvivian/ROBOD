import sys
import os
sys.path.append("..")
import random
from layers.batch_ensemble_layers import BatchEnsemble_Linear,BatchEnsemble_Conv,BatchEnsemble_BatchNorm2d

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
import time
from tqdm import tqdm


class ROBOD_LinearNet(nn.Module):
    """
    Describe: base architecture to concatenate the BatchEnsemble layers and create a network 
              parameters are similar to the BatchEnsemble layers
              Note: this is Linear layer ensembles
              
              Parameters: input_dim_list: the maximum number of nodes between each layers
                          num_models: number of submodels in ROBOD
                          device: the model is running on which GPUs or CPUs
                          dropout: dropout rate between layers
                          bias: bool, if we want to use bias, default = True
                          is_masked: bool, if we want to mask out some layers, default = False
                          masks: list, should be the number of nodes to mask out in each layer                   
    """
    def __init__(self, 
                 input_dim_list = [784, 400],
                 num_models = 2, 
                 device = "cuda",
                 dropout= 0.2,
                 bias = True,
                 is_masked = False,
                 masks = None):
        super(ROBOD_LinearNet, self).__init__()
        
        assert len(input_dim_list) >= 2
        
        #initialize the variables
        self.device = device
        self.input_dim_list = input_dim_list
        self.num_models = num_models  
        
        self.dropout = nn.Dropout(p=dropout)
        self.input_layer_list = nn.ModuleList()
        self.output_layer_list = nn.ModuleList()
        self.activation = torch.relu
        
        if masks != None:
            output_masks = masks[::-1][1:]   
        else:
            masks = [None for i in range(len(input_dim_list) -1)]
            output_masks = masks[::-1][1:]
            
        
        for i in range(len(input_dim_list) - 1):
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            self.input_layer_list.append(BatchEnsemble_Linear(in_channels= input_dim_list[i],
                                                             out_channels = input_dim_list[i+1], 
                                                             first_layer = first_layer, 
                                                             num_models =self.num_models, 
                                                             bias = True, 
                                                             constant_init = False,
                                                             device = "cuda",
                                                             is_masked = is_masked,
                                                             mask = masks[i]))      
        output_dim_list = input_dim_list[::-1]
        for i in range(len(output_dim_list) -2 ):
            self.output_layer_list.append(BatchEnsemble_Linear(in_channels = output_dim_list[i],
                                                               out_channels= output_dim_list[i+1],
                                                               first_layer = False, 
                                                               num_models =self.num_models, 
                                                               bias = True, 
                                                               constant_init = False,
                                                               device = device,
                                                               is_masked = is_masked,
                                                               mask = output_masks[i],))      
        self.output_layer_list.append(BatchEnsemble_Linear(in_channels = output_dim_list[-2],
                                                           out_channels= output_dim_list[-1],
                                                           first_layer = False, 
                                                           num_models =self.num_models, 
                                                           bias = True, 
                                                           constant_init = False,
                                                           device = device,
                                                           is_masked = False))
            
    def forward(self, x):
        output_list = []
        for i in range(len(self.input_dim_list) - 1):
            x = self.input_layer_list[i](x)
            x = self.dropout(self.activation(x))
            out = x
            for j in range(- (i+1), -1):
                out = self.output_layer_list[j](out)
                out = self.dropout(self.activation(out))
            out = torch.sigmoid(self.output_layer_list[-1](out))  
            output_list.append(out)
        return output_list
    
    
    
class ROBOD_ConvNet(nn.Module):
    def __init__(self,
                 input_dim_list = [1,8,4], 
                 device = "cuda",
                 masks = None,
                 is_masked = False,
                 num_models = 2):
        super(ROBOD_ConvNet, self).__init__()
        self.device = device
        self.masks = masks
        self.is_masked = is_masked
        self.num_models = num_models
        self.input_dim_list = input_dim_list
        self.input_layer_list = nn.ModuleList()
        self.output_layer_list = nn.ModuleList()
#         self.input_batch_list = nn.ModuleList()
#         self.output_batch_list = nn.ModuleList()
        if masks != None:
            output_masks = masks[::-1][1:]  
        else:
            masks = [None for i in range(len(input_dim_list) -1)]
            output_masks = masks[::-1][1:]
            
        for i in range(len(input_dim_list) - 1):
            first_layer = True if i ==0 else False
            self.input_layer_list.append(BatchEnsemble_Conv(in_channels = input_dim_list[i],
                                                     out_channels = input_dim_list[i+1], 
                                                     first_layer = first_layer, 
                                                     num_models = self.num_models, 
                                                     bias = True, 
                                                     constant_init = False,
                                                     device = self.device,
                                                     is_masked = self.is_masked,
                                                     mask = masks[i],
                                                     kernel_size =3,
                                                     stride=1, 
                                                     padding=1, 
                                                     conv_type = "Conv2d"))
#             self.input_batch_list.append(BatchEnsemble_BatchNorm2d(num_models= self.num_models,
#                                                                    num_features = input_dim_list[i+1],
#                                                                    eps=1e-04,
#                                                                    affine=False, 
#                                                                    device=self.device,                                             
#                                                                    masks = self.input_layer_list[-1].get_mask()))

        output_dim_list = input_dim_list[::-1]        
        
        for i in range(len(output_dim_list) -2 ):
            self.output_layer_list.append(BatchEnsemble_Conv(in_channels = output_dim_list[i],
                                                     out_channels = output_dim_list[i+1], 
                                                     first_layer = first_layer, 
                                                     num_models = self.num_models, 
                                                     bias = True, 
                                                     constant_init = False,
                                                     device = self.device,
                                                     is_masked = self.is_masked,
                                                     mask = output_masks[i],
                                                     kernel_size =2,
                                                     stride=2, 
                                                     conv_type = "ConvTranspose2d")) 
#             self.output_batch_list.append(BatchEnsemble_BatchNorm2d(num_models= self.num_models,
#                                                                    num_features = output_dim_list[i+1],
#                                                                    eps=1e-04,
#                                                                    affine=False, 
#                                                                    device=self.device,                                             
#                                                                    masks = self.output_layer_list[-1].get_mask()))                      
        self.output_layer_list.append(BatchEnsemble_Conv(in_channels = output_dim_list[-2],
                                                     out_channels = output_dim_list[-1], 
                                                     first_layer = False, 
                                                     num_models = self.num_models, 
                                                     bias = True, 
                                                     constant_init = False,
                                                     device = self.device,
                                                     is_masked = False,
                                                     mask = None,
                                                     kernel_size =2,
                                                     stride=2, 
                                                     conv_type = "ConvTranspose2d"))
        self.pool = nn.MaxPool2d(2, 2)
       

    def forward(self, x):
        output_list = []
        for i in range(len(self.input_dim_list) - 1):
            x = self.input_layer_list[i](x)
#             x = self.input_batch_list[i](x)
            x = self.pool(F.relu(x))
            out = x
            for j in range(- (i+1), -1):
                out = self.output_layer_list[j](out)
#                 out = self.output_batch_list[j+1](out)
                out = F.relu(out)
            out = torch.sigmoid(self.output_layer_list[-1](out))  
            output_list.append(out)
        return output_list