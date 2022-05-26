import numpy as np
import sys
sys.path.append("..")
import os
import random
from utils.dataset_generator import generate_data,generate_numpy_data
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser
import torch
from models.robod_sub import LinearROBODSub
from models.robod import LinearROBOD

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
 
#general settings
parser = HyperOptArgumentParser(strategy='grid_search')
parser.add_argument('--data', default='MNIST', help='currently support MNIST only')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--normal_class', type = int, default = 4)
parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--cuda_device', type = int, default = 5)
parser.add_argument('--model', type = str, default= "LinearROBODSub")
parser.add_argument('--transductive', type = str2bool, default= True)
parser.add_argument('--gpu_num', type = int, default= 4)
parser.add_argument('--subsample', type = float, default = 0.1)

args=parser.parse_args()


if args.gpu_num != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    print("GPU num: %d, GPU device: %d" %( torch.cuda.device_count(), args.gpu_num)) # print 1
    torch.set_num_threads(4)
else:
    print("Default GPU:0 used")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train_X, train_y =  generate_data(args.normal_class, dataset= args.data, transductive = True, flatten = True, GCN = True)
input_dim = train_X.shape[1] 

parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--num_layer', default = 2, type = int, tunable = True, options = [6])
parser.opt_list('--weight_decay', default = 0, type = float, tunable = True, options = [0,1e-5])
parser.opt_list('--num_models', default = 7, type = int, tunable= True, options = [8])
parser.opt_list('--lr', default =0.0001, type = float, tunable = True, options = [1e-3, 1e-4])
parser.opt_list('--epochs', default = 250, type = int, tunable = True, options = [500])#250,500
parser.opt_list('--threshold', default = 3, type = int, tunable = True, options = [1])
parser.opt_list('--dropout', default = 0, type = float, tunable = True, options = [0,0.2])
model_hparams = parser.parse_args("")


input_decay_list = [1.5,1.75,2.0,2.25,2.5,2.75,3,3.25]

if not args.transductive:
    save_dir = "../results/%s/%s/disjoint/" % (args.model + str(args.subsample), args.data) + str(args.normal_class)
else:
    save_dir = "../results/%s/%s/transductive/" % (args.model+ str(args.subsample), args.data) + str(args.normal_class)  

if not os.path.exists(save_dir):
     os.makedirs(save_dir)


for hparam in model_hparams.trials(100):
    print(hparam)
    for exp_num in range(3):
        torch.cuda.empty_cache()     
        hp_name = str('num_layer-%d epochs-%d lr-%.5f weight_decay-%.5f dropout-%.2f num_models-%d'
                           % (hparam.num_layer, 
                              hparam.epochs,
                              hparam.lr,
                              hparam.weight_decay,
                              hparam.dropout,
                              hparam.num_models))
             
        model = LinearROBODSub(lr = hparam.lr,
                 epochs = hparam.epochs,
                 num_layer = hparam.num_layer,
                 weight_decay = hparam.weight_decay,
                 dropout = hparam.dropout,
                 input_dim = input_dim,
                 input_decay_list = input_decay_list,
                 optimizer = torch.optim.Adam,
                 batch_size = args.batch_size,
                 num_model = hparam.num_models,
                 device = "cuda",
                 is_masked = True,
                 threshold = hparam.threshold,
                 subsample_rate = 0.5)
        
        total_time, memory_allocated, memory_reserved = model.fit(train_X)
        result = model.predict(train_X,train_y)
        
        roc_score = roc_auc_score(train_y, result)
        print("roc_score: ", roc_score)
        with open(os.path.join(save_dir,"%s%s" %(args.model,".txt")), "a+") as result_file:
            result_file.write("hpname: %s\n" % hp_name)
            result_file.write("exp_num %d\n" % exp_num)
            result_file.write("training time: %.5f\n" % total_time)
            result_file.write("aucroc score: %.5f\n" % roc_score)
            result_file.write("memory allocated: %.2f\n" % memory_allocated)
            result_file.write("memory reserved: %.2f\n" % memory_reserved)
            result_file.write("\n")
        if not os.path.exists(os.path.join(save_dir, hp_name)):
             os.makedirs(os.path.join(save_dir, hp_name))
        np.save(save_dir + "/" + hp_name + "/" + "%d%s" %(exp_num, "_prediction.npy"), result)
        torch.cuda.empty_cache()

