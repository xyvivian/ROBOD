import numpy as np
import sys
sys.path.append("..")
import os
import random
from utils.dataset_generator import generate_data,generate_numpy_data
from models.rda import LinearRDA
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
 

parser = HyperOptArgumentParser(strategy='grid_search')
parser.add_argument('--data', default='lympho', help='currently support MNIST only')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--normal_class', type = int, default = 4)
parser.add_argument('--device', type = str, default = 'cuda')
parser.add_argument('--cuda_device', type = int, default = 5)
parser.add_argument('--exp_num', type = int, default = 0)
parser.add_argument('--model', type = str, default= "LinearRDA")
parser.add_argument('--transductive', type = str2bool, default= True)
parser.add_argument('--gpu_num', type = int, default= -1)

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

def generate_input_dim_lst(num_layer, input_decay, input_dim, threshold = 3):
    input_dim_list = []
    for i in range(num_layer):
        if int(input_dim / (input_decay**i)) >= threshold:
            input_dim_list.append(int(input_dim / (input_decay**i)))
        else:
            input_dim_list.append(threshold)
    print(input_dim_list)
    return input_dim_list





parser = HyperOptArgumentParser(strategy='grid_search')
parser.opt_list('--num_layer', default = 2, type = int, tunable = True, options = [3])  #,5,6
parser.opt_list('--weight_decay', default = 0, type = float, tunable = True, options = [0,1e-5])
parser.opt_list('--inner_iteration', default= 10, type = int, tunable = True, options = [20,30]) #100
parser.opt_list('--lr', default =0.0001, type = float, tunable = True, options = [ 1e-3,1e-4] ) 
parser.opt_list('--iteration', default= 20, type = int, tunable = True, options = [20,30]) #350
parser.opt_list('--threshold', default = 3, type = int, tunable = True, options = [1])
parser.opt_list('--input_decay', default = 2, type = float, tunable = True, options = [4]) #3,3.25 
parser.opt_list('--dropout', default = 0, type = float, tunable = True, options = [0,0.2])
parser.opt_list('--lambda_', default = 1e-3, type = float, tunable = True, options = [1e-4, 1e-5])
model_hparams = parser.parse_args("")




if not args.transductive:
    save_dir = "../results/%s/%s/disjoint/" % (args.model, args.data) + str(args.normal_class)
else:
    save_dir = "../results/%s/%s/transductive/" % (args.model, args.data) + str(args.normal_class)  

if not os.path.exists(save_dir):
     os.makedirs(save_dir)





for hparam in model_hparams.trials(4000):
    print(hparam)
    for exp_num in range(3):
        torch.cuda.empty_cache()
        
        #memory_a = torch.cuda.max_memory_allocated(args.device) // (1024 ** 2)
        #memory_r = torch.cuda.max_memory_reserved(args.device) // (1024 ** 2)
        #print("Memory allocated during testing:%d, reserved: %d" %(memory_a, memory_r))
        
        hp_name = str('num_layer-%d input_decay-%.2f lambda_-%.6f inner_iteration-%d lr-%.5f iteration-%d weight_decay-%.5f dropout-%.2f'
                           % (hparam.num_layer, 
                              hparam.input_decay, 
                              hparam.lambda_, 
                              hparam.inner_iteration,
                              hparam.lr,
                              hparam.iteration,
                              hparam.weight_decay,
                              hparam.dropout))
            
        input_dim_list =  generate_input_dim_lst(hparam.num_layer, 
                                                 hparam.input_decay,
                                                 input_dim, 
                                                 threshold = hparam.threshold)
        
        
        model = LinearRDA(input_dim_list, 
                 lambda_= hparam.lambda_, 
                 symmetry= False,
                 learning_rate=hparam.lr, 
                 inner_iteration = hparam.inner_iteration, 
                 iteration = hparam.iteration,
                 batch_size = args.batch_size, 
                 regularization = "l1",
                 device = "cuda", 
                 transductive = args.transductive,
                 dropout = hparam.dropout)
        
        loss_lst, total_time, memory_allocated, memory_reserved = model.fit(train_X)
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
#         if not os.path.exists(os.path.join(save_dir, hp_name)):
#              os.makedirs(os.path.join(save_dir, hp_name))
#         np.save(save_dir + "/" + hp_name + "/" + "%d%s" %(exp_num, "_prediction.npy"), result)
#         np.save(save_dir + "/" + hp_name + "/" + "%d%s" %(exp_num, "_loss.npy"), loss_lst)
        torch.cuda.empty_cache()



