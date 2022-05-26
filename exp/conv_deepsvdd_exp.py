import numpy as np
import sys
sys.path.append("..")
import os
import random
from utils.dataset_generator import generate_data,generate_numpy_data
from models.deep_svdd import LinearDeepSVDD, ConvDeepSVDD
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
parser.add_argument('--model', type = str, default= "ConvDeepSVDD")
parser.add_argument('--transductive', type = str2bool, default= True)
parser.add_argument('--gpu_num', type = int, default= -1)

args=parser.parse_args()


if args.gpu_num != -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    print("GPU num: %d, GPU device: %d" %( torch.cuda.device_count(), args.gpu_num)) # print 1
    torch.set_num_threads(12)
else:
    print("Default GPU:0 used")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train_X, train_y =  generate_data(args.normal_class, dataset= args.data, transductive = True, flatten = False, GCN = True)
print(train_X.shape)
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
parser.opt_list('--conv_dim', default=8, type=int, tunable=True, options=[32]) #8,16,32
parser.opt_list('--fc_dim', default = 32, type = int, tunable = True, options = [128]) #16,32
parser.opt_list('--weight_decay', default = 0, type = float, tunable = True, options = [1e-6]) #1e-5 #1e-6, 0
parser.opt_list('--relu_slope', default = 1e-1, type = float, tunable = True, options = [1e-1]) # 1e-1,1e-3
parser.opt_list('--pre_train_epochs', default= 350, type = int, tunable = True, options = [350])   #,100
parser.opt_list('--pre_train_lr', default =0.0001, type = float, tunable = True, options = [1e-4] ) 
parser.opt_list('--train_epochs', default= 250, type = int, tunable = True, options = [250]) #500
parser.opt_list('--train_lr', default =0.0001, type = float, tunable = True, options = [1e-5, 1e-4])
parser.opt_list('--threshold', default = 3, type = int, tunable = True, options = [1])
#parser.opt_list('--input_decay', default = 2, type = float, tunable = True, options = [1.5,1.75,2,2.25,2.5,2.75,3,3.25])
parser.opt_list('--dropout', default = 0, type = float, tunable = True, options = [0]) #0.3
model_hparams = parser.parse_args("")


if not args.transductive:
    save_dir = "../results/%s/%s/disjoint/" % (args.model, args.data) + str(args.normal_class)
else:
    save_dir = "../results/%s/%s/transductive/" % (args.model, args.data) + str(args.normal_class)  

if not os.path.exists(save_dir):
     os.makedirs(save_dir)


for hparam in model_hparams.trials(1000):
    print(hparam)
    for exp_num in range(5):
        torch.cuda.empty_cache()
        
        hp_name = str('conv_dim-%d fc_dim-%d relu_slope-%0.3f pre_train_epochs-%d pre_train_lr-%.5f train_epochs-%d train_lr-%.5f weight_decay-%.6f'
                           % (hparam.conv_dim, 
                              hparam.fc_dim, 
                              hparam.relu_slope, 
                              hparam.pre_train_epochs,
                              hparam.pre_train_lr,
                              hparam.train_epochs,
                              hparam.train_lr,
                              hparam.weight_decay))
            
#         input_dim_list =  generate_input_dim_lst(hparam.num_layer, 
#                                                  hparam.input_decay,
#                                                  input_dim, 
#                                                  threshold = hparam.threshold)
        #set the convolutional layers
        if args.data == "MNIST":
            conv_dim_list = [input_dim, hparam.conv_dim, int(hparam.conv_dim / 2)]
        elif args.data == "CIFAR10":
            conv_dim_list = [input_dim, hparam.conv_dim, int(hparam.conv_dim * 2), int(hparam.conv_dim * 4)]        
        
        model = ConvDeepSVDD(
                 conv_input_dim_list = conv_dim_list, 
                 fc_dim = hparam.fc_dim, 
                 relu_slope = hparam.relu_slope,
                 pre_train = True, 
                 pre_train_weight_decay = hparam.weight_decay, 
                 train_weight_decay = hparam.weight_decay,
                 pre_train_epochs = hparam.pre_train_epochs, 
                 pre_train_lr = hparam.pre_train_lr, 
                 pre_train_milestones = [0], 
                 train_epochs = hparam.train_epochs,
                 train_lr = hparam.train_lr, 
                 train_milestones = [0],
                 batch_size = args.batch_size, 
                 device = args.device, 
                 objective = 'one-class',
                 nu = 0.1,
                 warm_up_num_epochs = 10,
                 dataset = args.data)
        
        loss_lst, total_time, memory_allocated, memory_reserved = model.fit(train_X)
        result = model.predict(train_X,train_y)
        
        roc_score = roc_auc_score(train_y, result)
        print("roc_score: ", roc_score)
        with open(os.path.join(save_dir,"%s%s"%(args.model, ".txt")), "a+") as result_file:
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

