import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import math
import time



class ApplyMask:
    """Hook that applies a mask to a tensor.

    Parameters
    ----------
    mask: the mask on the certain layer
    """
    def __init__(self, mask, device = "cuda"):
        # Precompute the masked indices.
        self._zero_indices = torch.BoolTensor(mask  != 0.0).to(device)

    def __call__(self, x, device = "cuda"):
        # A simple element-wise multiplication doesn't work if there are NaNs.
        return torch.where(self._zero_indices, torch.tensor(0, dtype = x.dtype,device = x.get_device()),x)
    
    
    
class BatchEnsemble_Linear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 first_layer = False, 
                 num_models =100, 
                 bias = True, 
                 constant_init = False,
                 device = "cuda",
                 is_masked = True,
                 mask = None):
        """
        Parameters:
        in_channels: int, # of in channels
        out_channels: int, # of out channels
        first_layer: bool, default = False, whether the BatchEnsemble layer is the first
                layer or not. If first_layer =True and we are not subsampling, we will need to 
                expand the input variable dimension to let the model produces predictions for all
                data
        num_models: int, # of models
        bias: bool, default = True, whether to use the bias in the layer
        constant_init: bool, default = False, if set true, the hidden alphas and gammas are produced 
                   with constant initiation to 1.0
        device: str, default = "cuda"
        is_masked: bool, default = True, whether to use masks on the gammas to reduce the hidden neuron sizes
        """
        super(BatchEnsemble_Linear, self).__init__()
    
        self.in_features = in_channels
        self.out_features = out_channels
        self.first_layer = first_layer
        self.num_models = num_models
        self.device = device
        self.is_masked = is_masked
        self.first_layer = first_layer
        self.constant_init = constant_init
        self.mask_layer = None
        
        #initialize linear layer
        self.layer = nn.Linear(in_channels, out_channels, bias = False)
    
        #initialize alphas, gammas and bias
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
                                 
        if self.is_masked:
            #masked weights initialization
            self.mask_layer = self.init_mask(self.num_models,
                                         out_channels, 
                                         size = mask)
            #mask the gammas -> initialized to zero and 
            #register the backward prop hook that stops update the gradients
            self.gamma.data = self.gamma.data * (self.mask_layer == 0.0)
            self.gamma.register_hook(ApplyMask(self.mask_layer))
            if bias:
                self.bias.data = self.bias.data * (self.mask_layer == 0.0)
                self.bias.register_hook(ApplyMask(self.mask_layer))
  
    def reset_parameters(self):
        """
        Reset the alphas, gammas
        And the weights, bias in the neural network
        """
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            nn.init.constant_(self.gamma, 1.)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            nn.init.normal_(self.gamma, mean=1., std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
                                 
    def init_mask(self, 
                  num_models, 
                  out_features,
                  size = [4,2]) -> np.ndarray:
        """
        Build mask for gamma, if bias exists, also build a mask for the bias
        Return: np.ndarray, which is the mask utilized in this layer
        """
        mask = np.zeros(shape=(num_models, out_features))
        for i in range(num_models):
            zero_idx = np.random.choice(out_features, size=size[i], replace=False)
            mask[i][zero_idx] = 1.0
        return mask        
    
    def get_mask(self):
        """
        Get the mask created in this BatchEnsemble layer
        Return: np.ndarray (mask)
        """
        return self.mask_layer
                                 
                                 
    def forward(self, x):
        """
        Forward propagation, contains two phases
        First, change the alpha, gamma and bias with broadcasting, to fit the data dimension
        Second, apply one forward propagation descired in the paper(fast version)
        """

        num_examples_per_model = int(x.size(0) / self.num_models)
        alpha = self.alpha.repeat(num_examples_per_model,1)
        gamma = self.gamma.repeat(num_examples_per_model,1)
        bias = self.bias.repeat(num_examples_per_model,1)
        #forward propagation
        result = self.layer(x*alpha)*gamma
        return result + bias if self.bias is not None else result
    
    
    
class BatchEnsemble_Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels, 
                 first_layer = False, 
                 num_models =100, 
                 bias = True, 
                 constant_init = False,
                 device = "cuda",
                 is_masked = True,
                 mask = None,
                 kernel_size =2,
                 stride=1, 
                 padding=0, 
                 dilation=1,
                 conv_type = "Conv2d"):
        """
        Parameters:
        in_channels: int, # of in channels
        out_channels: int, # of out channels
        first_layer: bool, default = False, whether the BatchEnsemble layer is the first
                layer or not. If first_layer =True and we are not subsampling, we will need to 
                expand the input variable dimension to let the model produces predictions for all
                data
        num_models: int, # of models
        bias: bool, default = True, whether to use the bias in the layer
        constant_init: bool, default = False, if set true, the hidden alphas and gammas are produced 
                   with constant initiation to 1.0
        device: str, default = "cuda"
        is_masked: bool, default = True, whether to use masks on the gammas to reduce the hidden neuron sizes
        conv_type: str, "Conv2d" or "ConvTranspose2d"
        layer parameters:
        kernel_size, stride, padding, dilation
        """
        super(BatchEnsemble_Conv, self).__init__()
    
        self.in_features = in_channels
        self.out_features = out_channels
        self.first_layer = first_layer
        self.num_models = num_models
        self.device = device
        self.is_masked = is_masked
        self.first_layer = first_layer
        self.constant_init = constant_init
        self.mask_layer = None
        
        #initialize the Conv layer(can be both Conv2d or Transposed Conv2d)
        if conv_type == "Conv2d":
            self.layer = nn.Conv2d(in_channels= self.in_features, 
                                out_channels = self.out_features, 
                                kernel_size = kernel_size, 
                                stride= stride, 
                                padding= padding, 
                                dilation= dilation, 
                                bias= False)
        elif conv_type == "ConvTranspose2d":
            self.layer = nn.ConvTranspose2d(in_channels = self.in_features,
                                         out_channels = self.out_features, 
                                         kernel_size = kernel_size, 
                                         stride= stride, 
                                         padding= padding, 
                                         bias=False, 
                                         dilation=dilation)
            
        #initialize alphas, gammas and bias
        self.alpha = nn.Parameter(torch.Tensor(num_models, in_channels))
        self.gamma = nn.Parameter(torch.Tensor(num_models, out_channels))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.num_models, out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
                                 
        if self.is_masked:
            #masked weights initialization
            self.mask_layer = self.init_mask(self.num_models,
                                         out_channels, 
                                         size = mask)
            #mask the gammas -> initialized to zero and 
            #register the backward prop hook that stops update the gradients
            self.gamma.data = self.gamma.data * (self.mask_layer == 0.0)
            self.gamma.register_hook(ApplyMask(self.mask_layer))
            if bias:
                self.bias.data = self.bias.data * (self.mask_layer == 0.0)
                self.bias.register_hook(ApplyMask(self.mask_layer))
  
    def reset_parameters(self):
        """
        Reset the alphas, gammas
        And the weights, bias in the neural network
        """
        if self.constant_init:
            nn.init.constant_(self.alpha, 1.)
            nn.init.constant_(self.gamma, 1.)
        else:
            nn.init.normal_(self.alpha, mean=1., std=0.5)
            nn.init.normal_(self.gamma, mean=1., std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)          
                                 
    def init_mask(self, 
                  num_models, 
                  out_features,
                  size = [4,2]) -> np.ndarray:
        """
        Build mask for gamma, if bias exists, also build a mask for the bias
        Return: np.ndarray, which is the mask utilized in this layer
        """
        mask = np.zeros(shape=(num_models, out_features))
        #print(num_models, out_features, size)
        for i in range(num_models):
            zero_idx = np.random.choice(out_features, size=size[i], replace=False)
            mask[i][zero_idx] = 1.0
        return mask        
    
    def get_mask(self):
        """
        Get the mask created in this BatchEnsemble layer
        Return: np.ndarray (mask)
        """
        return self.mask_layer
                                                                  
    def forward(self, x):
        """
        Forward propagation, contains two phases
        First, change the alpha, gamma and bias with broadcasting, to fit the data dimension
        Second, apply one forward propagation descired in the paper(fast version)
        """
        num_examples_per_model = int(x.size(0) / self.num_models)
        alpha = self.alpha.repeat(num_examples_per_model,1)
        gamma = self.gamma.repeat(num_examples_per_model,1)
        bias = self.bias.repeat(num_examples_per_model,1)
        
        alpha.unsqueeze_(-1).unsqueeze_(-1)
        gamma.unsqueeze_(-1).unsqueeze_(-1) 
        if self.bias is not None:
            bias.unsqueeze_(-1).unsqueeze_(-1)
        #forward propagation
        result = self.layer(x*alpha)*gamma
        return result + bias if self.bias is not None else result
    
   


class BatchEnsemble_BatchNorm1d(nn.Module):
    def __init__(self,
                num_models, 
                features,
                eps=1e-05,
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                device="cuda", 
                constant_init = False,
                masks = None):
        """
        Parameters:
        affine: bool, whether the batch_norm layer is affine or not
        constant_init: bool, whether to use constant init for batch_norm weights
                         and biases
        num_features: int, # of input features(without masks)
        masks: np.ndarray: a mask put onto the input       
        """
        super(BatchEnsemble_BatchNorm1d, self).__init__()
        self.num_models = num_models
        self.features
        self.constant_init = constant_init
        self.mask_layer = masks
        self.num_features_list = self.masked_num_features()
                     
        #init the batch norm from nn.BatchNorm
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(num_feature, eps = eps, momentum = momentum, affine=affine,
                           track_running_stats = track_running_stats, device = device)
            for num_feature in self.num_features_list])
        
        self.reset_parameters()
        
    def masked_num_features(self):
        """
        Retrieve the num of features that are not masked out by the mask_layer
        Return: list: each element in the list is # of non-zero features in the subnetwork
        """
        num_features_list = []
        for i in range(self.num_models):
            num_features_list.append(self.features -np.count_nonzero( self.mask_layer[i]))
        return num_features_list
                                    
    def reset_parameters(self):
        if self.affine:
            for l in self.batch_norms:
                nn.init.constant_(l.bias, 0.)
                if self.constant_init:
                    nn.init.constant_(l.weight, 1.)
                else:
                    nn.init.normal_(l.weight, mean=1., std=0.5)
                                  
    def forward(self,x):
        inputs = torch.chunk(x, self.num_models, dim=0)
        for i in inputs:
            if self.mask_layer is not None: 
                output_list =[]
                for i in range(len(inputs)):
                    output = inputs[i].clone()
                    output[:,(self.mask_layer[i] == 0.0)] = \
                        self.batch_norms[i](inputs[i][:,(self.mask_layer[i] == 0.0)])
                    output_list.append(output)
                return torch.cat(output_list,dim =0)  
            else:
                return torch.cat([self.batch_norms[0](inpt) 
                              for i,inpt in enumerate(inputs)],dim=0)
        
        

class BatchEnsemble_BatchNorm2d(nn.Module):
    def __init__(self,
                num_models,
                num_features,
                eps=1e-05,
                momentum=0.1, 
                affine=True, 
                track_running_stats=True,
                device="cuda", 
                constant_init = False,
                masks = None):
        """
        Parameters:
        affine: bool, whether the batch_norm layer is affine or not
        constant_init: bool, whether to use constant init for batch_norm weights
                         and biases
        num_features: int, # of input features(without masks)
        masks: np.ndarray: a mask put onto the input       
        """
        super(BatchEnsemble_BatchNorm2d, self).__init__()
        self.num_models = num_models
        self.constant_init = constant_init
        self.affine = affine
        self.mask_layer = masks
        self.features = num_features
        self.num_features_list = self.masked_num_features()
        self.device = device
        #init the batch norm from nn.BatchNorm
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm2d(num_feature, eps = eps, momentum = momentum, affine=affine,
                           track_running_stats = track_running_stats, device = device)
            for num_feature in self.num_features_list])
        self.reset_parameters()   
        
    def masked_num_features(self):
        """
        Retrieve the num of features that are not masked out by the mask_layer
        Return: list: each element in the list is # of non-zero features in the subnetwork
        """
        num_features_list = []
        for i in range(self.num_models):
            num_features_list.append(self.features -np.count_nonzero( self.mask_layer[i]))
        return num_features_list
                                    
    def reset_parameters(self):
        if self.affine:
            for l in self.batch_norms:
                nn.init.constant_(l.bias, 0.)
                if self.constant_init:
                    nn.init.constant_(l.weight, 1.)
                else:
                    nn.init.normal_(l.weight, mean=1., std=0.5)
                                  
    def forward(self,x):
        inputs = torch.chunk(x, self.num_models, dim=0)
        for i in inputs:
            if self.mask_layer is not None: 
                output_list =[]
                for i in range(len(inputs)):
                    output = inputs[i].clone()
                    output[:,(self.mask_layer[i] == 0.0),:,:] = \
                        self.batch_norms[i](inputs[i][:,(self.mask_layer[i] == 0.0),:,:])
                    output_list.append(output)
                return torch.cat(output_list,dim =0)  
            else:
                return torch.cat([self.batch_norms[0](inpt) 
                              for i,inpt in enumerate(inputs)],dim=0)
    

    
class BatchEnsemble_Activation(nn.Module):
    def __init__(self,
                activation,
                num_models):
        super(BatchEnsemble_Activation, self).__init__()
        self.activation = activation
        self.num_models = num_models
        
    def forward(self,x, mask_layer):
        if self.activation == torch.relu or mask_layer is None:
            return self.activation(x) 
        else:
            num_examples_per_model = int(x.size(0) / self.num_models)
            if num_examples_per_model > 0:
                mask = np.repeat(mask_layer, repeats = num_examples_per_model, axis = 0 )      
            inputs = x.clone()            
            if len(inputs.shape) == 2:
                nonzero_entries = (mask != 0.0)
                inputs[nonzero_entries,:]  = self.activation(x[nonzero_entries,:])
                return inputs        
            elif len(inputs.shape) == 4:
                nonzero_entries = (mask != 0.0)
                inputs[:,nonzero_entries,:,:] = self.activation(x[:,nonzero_entries,:,:])
                return inputs