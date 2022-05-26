import sys
sys.path.append("..")

import time
from models.isoforest import IsoForest

import numpy as np
import os
from utils.dataset_generator import generate_data,generate_numpy_data
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import torch 

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# In[12]:


#general settings
parser = HyperOptArgumentParser(strategy='grid_search')
parser.add_argument('--data', default='MNIST', help='currently support MNIST only')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--normal_class', type = int, default = 4)
parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--model', type = str, default= "IsolationForest")
parser.add_argument('--transductive', type = str2bool, default= True)

args=parser.parse_args("")
torch.set_num_threads(12)

train_X, train_y =  generate_data(args.normal_class, dataset= args.data, transductive = True, flatten = True, GCN = False)
input_dim = train_X.shape[1]


# In[16]:


if not args.transductive:
    save_dir = "../results/%s/%s/disjoint/" % (args.model, args.data) + str(args.normal_class)
else:
    save_dir = "../results/%s/%s/transductive/" % (args.model, args.data) + str(args.normal_class)  


# In[17]:


parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--num_trees', default = 100, type = int, tunable = True, options = [100,50,200,500])
parser.opt_list('--subsample', default = 256, type = int, tunable = True, options = [512,256,128,64])
model_hparams = parser.parse_args("")


# In[18]:


for hparam in model_hparams.trials(48):
    print(hparam)
    for exp_num in range(3):
        start_time = time.time()
        hp_name = "num_trees-%d subsample-%d" % (hparam.num_trees, hparam.subsample)
        model = IsoForest( n_estimators=hparam.num_trees, max_samples = hparam.subsample)
        total_time = model.fit(train_X)
        scores = model.predict(train_X,train_y)
        auroc = roc_auc_score(train_y, scores)
        print("total_time: %.2f, auroc: %.2f" % (total_time, auroc))
        with open(os.path.join(save_dir,"%s%s" %(args.model,".txt")), "a+") as result_file:
            result_file.write("hpname: %s\n" % hp_name)
            result_file.write("exp_num %d\n" % exp_num)
            result_file.write("training time: %.5f\n" % total_time)
            result_file.write("aucroc score: %.5f\n" % auroc)
            result_file.write("\n")


# In[ ]:




