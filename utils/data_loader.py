import numpy as np
import math
import torch

class CustomizeDataLoader():
    def __init__(self, 
                 data,
                 label = None,
                 num_models = 1,
                 batch_size = 300,
                 subsampled_indices = [],
                 device = "cuda"):
        super(CustomizeDataLoader, self).__init__()
        self.data = data
        self.label = label
        self.num_models = num_models
        self.total_batch_size = batch_size * num_models
        self.original_batch_size = batch_size
        self.device = device
            
        #we donot use subsampling here
        if subsampled_indices == []:
            self.subsampled_indices = [np.random.permutation(np.arange(self.data.shape[0]))] * num_models
        else:
            self.subsampled_indices = subsampled_indices
    
    def num_total_batches(self):
        return math.ceil(self.subsampled_indices[0].shape[0] /(self.original_batch_size) )
    
    def get_next_batch(self, idx):
        """
        Describe: Generate batch X and batch y according to the subsampled indices
        Parameter: 
             idx: the index of iteration in the current batch
        Return:
             batch_index: the indices of subsampling
             batch_X: numpy array with subsampled indices
        """    
        num_per_network = self.original_batch_size
        batch_index = [self.subsampled_indices[i][idx *num_per_network : (idx + 1)*num_per_network] 
                                                                           for i in range(self.num_models)]
        batch_index = np.concatenate(batch_index, axis=0)
        if self.label is not None:
            return batch_index, \
                   torch.tensor(self.data[batch_index]).to(self.device), \
                   self.label[batch_index]
        else:  
           # print(torch.from_numpy(self.data[batch_index]).shape)
            return batch_index, torch.tensor(self.data[batch_index]).to(self.device)