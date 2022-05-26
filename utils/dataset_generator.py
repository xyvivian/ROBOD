import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torchvision
from torchvision import transforms
import numpy as np
import random
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler


#MNIST precomputed min_max score
# Pre-computed min and max values (after applying GCN) from train data per class
mnist_min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

#CIFAR precomputed in min_max sore
cifar_min_max =    [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]


def generate_data(normal_class = 4,
                  transductive= True, 
                  flatten = True, 
                  GCN = False, 
                  resize = None,
                  dataset = "MNIST"):
    if dataset in ["cardio","thyroid","lympho"]:
        return generate_tabular_dataset(dataset = dataset)
    if transductive:
        dataset = generate_transductive_dataset(normal_class = normal_class,
                                                data_dir= '../dataset', 
                                                flatten = flatten, 
                                                GCN = GCN ,
                                                resize = resize, 
                                                dataset= dataset)
        return generate_numpy_data(dataset)
    else:
        trainset, testset = generate_disjoint_dataset(normal_class = normal_class,
                                                      data_dir= '../dataset', 
                                                      flatten = flatten, 
                                                      GCN = GCN, 
                                                      resize= resize, 
                                                      dataset =dataset)
        return generate_numpy_data(trainset), generate_numpy_data(testset)


def global_contrast_normalization(x: torch.tensor, scale='l2'):
    """
    Apply global contrast normalization to tensor, i.e. subtract
    mean across features (pixels) and normalize by scale,
    which is either the standard deviation, L1- or L2-norm across 
    features (pixels).
    Note this is a *per sample* normalization globally across 
    features (and not across the dataset).
    """

    assert scale in ('l1', 'l2')
    n_features = int(np.prod(x.shape))
    mean = torch.mean(x)  # mean over all features (pixels) per sample
    x -= mean
    if scale == 'l1':
        x_scale = torch.mean(torch.abs(x))
    if scale == 'l2':
        x_scale = torch.sqrt(torch.sum(x ** 2)) / n_features
    x /= x_scale
    return x

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()


def generate_downsampled_indices(dataset, 
                                 normal_class, 
                                 down_sample_rate = 0.1):
    targets = torch.tensor(dataset.targets)
    idx = np.arange(len(targets))
    # Get indices to keep
    idx_to_keep = targets[idx]== normal_class
    down_sampled_idx = targets[idx] != normal_class
    # Nomial idex consists only with 1 label
    # Abnormal idx consists of all other labels
    nomial_idx = idx[idx_to_keep]
    abnormal_idx = idx[down_sampled_idx]
    m = nomial_idx.shape[0]
    np.random.seed(4321)
    abnormal_idx = np.random.choice(abnormal_idx, 
                                    size= int(down_sample_rate * m),
                                    replace = False)
    overall_idx  = np.append(nomial_idx, abnormal_idx, axis = 0)
    random.seed(4321)
    random.shuffle(overall_idx)
    return overall_idx

def generate_disjoint_dataset(normal_class,
                              data_dir= 'dataset',
                              flatten = True, 
                              GCN = False,
                              resize = None, 
                              dataset = "MNIST"):  
    if resize != None:
        transform_lst = [transforms.Resize(resize), transforms.ToTensor()]
    else:
        transform_lst = [transforms.ToTensor()]
    if GCN:
        if dataset == "MNIST":
            transform_lst.extend([transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([mnist_min_max[normal_class][0]],
                                                            [mnist_min_max[normal_class][1] - mnist_min_max[normal_class][0]])])
        elif dataset == "CIFAR10":
            transform_lst.extend([transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([cifar_min_max[normal_class][0]] * 3,
                                                            [cifar_min_max[normal_class][1] - cifar_min_max[normal_class][0]]*3)]) 
    if flatten:
        transform_lst.extend([lambda x: x.numpy().flatten()])
    transform = transforms.Compose(transform_lst)
    if dataset == "MNIST":
        train_set = torchvision.datasets.MNIST(data_dir,
                                               train=True, 
                                               download=True,
                                               transform = transform)
        test_set = torchvision.datasets.MNIST(data_dir,
                                              train=False,
                                              download=True,
                                              transform = transform)
    elif dataset == "CIFAR10":
        train_set = torchvision.datasets.CIFAR10(data_dir, 
                                                 train=True,
                                                 download=True, 
                                                 transform = transform)
        test_set = torchvision.datasets.CIFAR10(data_dir, 
                                                train=False, 
                                                download=True, 
                                                transform = transform)
    #Subset the training data
    targets = torch.tensor(train_set.targets)
    idx = get_target_label_idx(targets.clone().data.cpu().numpy(), normal_class)
    #Subset of the sample
    train_set.data = train_set.data[idx]
    train_set.targets = torch.tensor(train_set.targets)[idx]  
    test_set.targets = torch.tensor(test_set.targets)
    train_set = relabel_dataset(normal_class, train_set)
    test_set = relabel_dataset(normal_class, test_set)
    
    return train_set, test_set
    
    

def generate_transductive_dataset(normal_class, 
                                  data_dir= 'dataset', 
                                  flatten = True, 
                                  GCN = False , 
                                  resize= None, 
                                  dataset = "MNIST"):  
    if resize != None:
        transform_lst = [transforms.Resize(resize), transforms.ToTensor()]
    else:
        transform_lst = [transforms.ToTensor()]
    if GCN:
        if dataset == "MNIST":
            transform_lst.extend([transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([mnist_min_max[normal_class][0]],
                                                            [mnist_min_max[normal_class][1] - mnist_min_max[normal_class][0]])])
        elif dataset == "CIFAR10":
            transform_lst.extend([transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([cifar_min_max[normal_class][0]] * 3,
                                                            [cifar_min_max[normal_class][1] - cifar_min_max[normal_class][0]]*3)]) 
    if flatten:
        transform_lst.extend([lambda x: x.numpy().flatten()])
    transform = transforms.Compose(transform_lst)
    if dataset == "MNIST":
        dataset = torchvision.datasets.MNIST(data_dir, 
                                             train=True, 
                                             download=True, 
                                             transform = transform)
    elif dataset == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(data_dir, 
                                               train=True, 
                                               download=True, 
                                               transform = transform)        
    #Downsample the dataset with the normal class
    idx = generate_downsampled_indices(dataset, normal_class)
    #Subset of the sample
    dataset.data = dataset.data[idx]
    dataset.targets = torch.tensor(dataset.targets)[idx]
    #Replacing the label, normal class = 0, abnormal class = 1
    return relabel_dataset(normal_class, dataset)


def relabel_dataset(normal_class, dataset):
    for i in range(len(dataset)):
        if dataset.targets[i] == normal_class:
            dataset.targets[i] = 0
        else:
            dataset.targets[i] = 1
    return dataset


def generate_numpy_data(dataset):    
    X = []
    y = []
    for i in range(len(dataset)):
        if type(dataset[i][0]) == np.ndarray:
            X.append(dataset[i][0])
        else:
            X.append(dataset[i][0].detach().cpu().numpy())
        y.append(dataset[i][1])
    return np.array(X), np.array(y)



def generate_tabular_dataset(dataset = "cardio"):
    try:
        data = sio.loadmat('../dataset/tabular/' + dataset + '.mat')
    except:
        print("dataset not found, need to have data in the correct directory")
        return None
    scaler = MinMaxScaler()
    features = data['X']
    scaler.fit(features)
    features = scaler.transform(features)
    labels = data['y']
    count = 0
    for i in labels:
        if i == 1:
            count += 1        
    print("percentage of anomaly: %.3f" % (count / labels.shape[0]))
    return features.astype('float32'), labels
    