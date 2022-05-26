import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10_LeNet(nn.Module):

    def __init__(self,conv_input_dim_list = [3,32, 64, 128],fc_dim = 128,relu_slope = 0.1):
        super().__init__()

        self.rep_dim = fc_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.slope = relu_slope

        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)
        self.bn2d1 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(conv_input_dim_list[1], conv_input_dim_list[2], 5, bias=False, padding=2)
        self.bn2d2 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(conv_input_dim_list[2], conv_input_dim_list[3], 5, bias=False, padding=2)
        self.bn2d3 = nn.BatchNorm2d(conv_input_dim_list[3], eps=1e-04, affine=False)
        self.fc1 = nn.Linear(conv_input_dim_list[3] * 4 * 4, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x), negative_slope= self.slope))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x), negative_slope= self.slope))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x), negative_slope= self.slope))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
    
class CIFAR10_LeNet_Autoencoder(nn.Module):

    def __init__(self,conv_input_dim_list = [3,32, 64, 128],fc_dim = 128,relu_slope = 0.1):
        super().__init__()

        self.slope = relu_slope
        self.rep_dim = fc_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(conv_input_dim_list[1], conv_input_dim_list[2], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(conv_input_dim_list[2], conv_input_dim_list[3], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(conv_input_dim_list[3], eps=1e-04, affine=False)
        self.fc1 = nn.Linear(conv_input_dim_list[3] * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), conv_input_dim_list[3], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(conv_input_dim_list[3], eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(conv_input_dim_list[3],conv_input_dim_list[2], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(conv_input_dim_list[2], conv_input_dim_list[1], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(conv_input_dim_list[1], conv_input_dim_list[0], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x),negative_slope=self.slope))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x),negative_slope=self.slope))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x),negative_slope=self.slope))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn2d4(x),negative_slope=self.slope), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn2d5(x),negative_slope=self.slope), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn2d6(x),negative_slope=self.slope), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class MNIST_LeNet(nn.Module):

    def __init__(self,conv_input_dim_list = [1,8,4],fc_dim = 32,relu_slope = 0.1):
        super().__init__()

        self.rep_dim = fc_dim  # final dense layer of 32 units
        self.pool = nn.MaxPool2d(2, 2)  # use 2x2 max-pooling
        self.slope = relu_slope

        # torch.nn.Conv2d(in_channels, out_channels, kernel_size,
        # stride=1, padding=2, dilation=1, groups=1, bias=False)

        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)  # conv layer 8*5*5*1
        self.bn1 = nn.BatchNorm2d( conv_input_dim_list[1], eps=1e-04, affine=False)

        self.conv2 = nn.Conv2d( conv_input_dim_list[1],  conv_input_dim_list[2], 5,
                               bias=False, padding=2)  # conv layer 4*5*5*1
        self.bn2 = nn.BatchNorm2d( conv_input_dim_list[2], eps=1e-04, affine=False)

        self.fc1 = nn.Linear( conv_input_dim_list[2] * 7 * 7, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x), negative_slope= self.slope))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x), negative_slope= self.slope))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MNIST_LeNet_Autoencoder(nn.Module):

    def __init__(self,conv_input_dim_list = [1,8,4],fc_dim= 32, relu_slope = 0.1):
        super().__init__()

        self.rep_dim = fc_dim
        self.pool = nn.MaxPool2d(2, 2)
        self.slope = relu_slope

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(conv_input_dim_list[1], conv_input_dim_list[2], 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.fc1 = nn.Linear(conv_input_dim_list[2] * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / 16), conv_input_dim_list[2] , 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(conv_input_dim_list[2] , eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(conv_input_dim_list[2] , conv_input_dim_list[1] , 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(conv_input_dim_list[1] , eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(conv_input_dim_list[1] , conv_input_dim_list[0] , 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x),negative_slope=self.slope))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x),negative_slope=self.slope))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x, negative_slope=self.slope), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x), negative_slope=self.slope), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x), negative_slope=self.slope), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x

    
    
    
    
class CIFAR10_LeNet_AENoleaky(nn.Module):

    def __init__(self,conv_input_dim_list = [3,32, 64, 128],fc_dim = 128):
        super().__init__()

        self.rep_dim = fc_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d1 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(conv_input_dim_list[1], conv_input_dim_list[2], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d2 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(conv_input_dim_list[2], conv_input_dim_list[3], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d3 = nn.BatchNorm2d(conv_input_dim_list[3], eps=1e-04, affine=False)
        self.fc1 = nn.Linear(conv_input_dim_list[3] * 4 * 4, self.rep_dim, bias=False)
        self.bn1d = nn.BatchNorm1d(self.rep_dim, eps=1e-04, affine=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / (4 * 4)), conv_input_dim_list[3], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d4 = nn.BatchNorm2d(conv_input_dim_list[3], eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(conv_input_dim_list[3],conv_input_dim_list[2], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d5 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(conv_input_dim_list[2], conv_input_dim_list[1], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.bn2d6 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(conv_input_dim_list[1], conv_input_dim_list[0], 5, bias=False, padding=2)
        nn.init.xavier_uniform_(self.deconv4.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(torch.relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(torch.relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = self.bn1d(self.fc1(x))
        x = x.view(x.size(0), int(self.rep_dim / (4 * 4)), 4, 4)
        x = F.leaky_relu(x)
        x = self.deconv1(x)
        x = F.interpolate(torch.relu(self.bn2d4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(torch.relu(self.bn2d5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(torch.relu(self.bn2d6(x)), scale_factor=2)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x


class MNIST_LeNet_AENoleaky(nn.Module):

    def __init__(self,conv_input_dim_list = [1,8,4],fc_dim= 32):
        super().__init__()

        self.rep_dim = fc_dim
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(conv_input_dim_list[0], conv_input_dim_list[1], 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(conv_input_dim_list[1], eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(conv_input_dim_list[1], conv_input_dim_list[2], 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(conv_input_dim_list[2], eps=1e-04, affine=False)
        self.fc1 = nn.Linear(conv_input_dim_list[2] * 7 * 7, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(int(self.rep_dim / 16), conv_input_dim_list[2] , 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(conv_input_dim_list[2] , eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(conv_input_dim_list[2] , conv_input_dim_list[1] , 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(conv_input_dim_list[1] , eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(conv_input_dim_list[1] , conv_input_dim_list[0] , 5, bias=False, padding=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(torch.relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(torch.relu(self.bn2(x)))
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        x = x.view(x.size(0), int(self.rep_dim / 16), 4, 4)
        x = F.interpolate(torch.relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(torch.relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(torch.relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        x = torch.sigmoid(x)
        return x
