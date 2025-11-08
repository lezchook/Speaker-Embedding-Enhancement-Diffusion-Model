# Portions of this file are adapted and modified from:
# VoxCeleb Trainer (https://github.com/clovaai/voxceleb_trainer)
# Copyright (c) 2020-present NAVER Corp.
# Licensed under the MIT License.
#
# Modifications © 2025 Sergei Leshchenko
#
# This script implements the ResNet architecture and the corresponding training and evaluation procedures

import os

import torch
import torch.nn as nn
import torchaudio

from common import PreEmphasis


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, norm_layer=None, activation=nn.ReLU):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, groups=1, bias=False, dilation=dilation)
        self.bn1        = norm_layer(planes)
        self.relu       = activation(inplace=True)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, groups=1, bias=False, dilation=dilation)
        self.bn2        = norm_layer(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class MaxoutLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super(MaxoutLinear, self).__init__()

        self.linear1 = nn.Linear(*args, **kwargs)
        self.linear2 = nn.Linear(*args, **kwargs)

    def forward(self, x):
        return torch.max(self.linear1(x), self.linear2(x))
    
class ResNet(nn.Module):
    def __init__(self, block, layers, activation, num_filters, nOut, n_mels=64, log_input=True, **kwargs):
        super(ResNet, self).__init__()

        self.inplanes     = num_filters[0]
        self.n_mels       = n_mels
        self.log_input    = log_input

        self.torchfb        = torch.nn.Sequential(PreEmphasis(), 
                                                  torchaudio.transforms.MelSpectrogram(sample_rate=16000, 
                                                                                       n_fft=512, 
                                                                                       win_length=400, 
                                                                                       hop_length=160, 
                                                                                       window_fn=torch.hamming_window, 
                                                                                       n_mels=n_mels))
        self.instancenorm   = nn.InstanceNorm1d(n_mels)

        self.conv1  = nn.Conv2d(1, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(num_filters[0])
        self.relu   = activation(inplace=True)
        
        self.layer1 = self._make_layer(block, num_filters[0], layers[0], stride=1, activation=activation)
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=2, activation=activation)
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=2, activation=activation)
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=2, activation=activation)

        outmap_size = int(self.n_mels/8)

        self.attention = nn.Sequential(nn.Conv1d(num_filters[3]*outmap_size, 128, kernel_size=1), 
                                       nn.ReLU(), 
                                       nn.BatchNorm1d(128), 
                                       nn.Conv1d(128, num_filters[3]*outmap_size, kernel_size=1), 
                                       nn.Softmax(dim=2))
 
        out_dim = num_filters[3]*outmap_size*2

        self.fc = nn.Sequential(MaxoutLinear(out_dim, nOut), nn.BatchNorm1d(nOut, affine=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, activation=nn.ReLU):
        downsample = None

        if stride != 1 or self.inplanes != planes*block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, activation=activation))
        self.inplanes = planes*block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, activation=activation))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                x = self.torchfb(x) + 1e-6
                if self.log_input: x = x.log()
                x = self.instancenorm(x).unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        mean = x.mean(dim=3)  # (B, C, F)
        std = x.std(dim=3)    # (B, C, F)
        x = torch.cat((mean, std), dim=2)  # (B, C, 2*F)
        x = torch.flatten(x, start_dim=1)  # (B, 2*C*F)

        x = self.fc(x)

        return x

class MainModel(nn.Module):
    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__();

        self.__S__ = model
        self.__L__ = trainfunc

    def forward(self, data, label=None):
        data = data.reshape(-1, data.size()[-1]).cuda() 
        outp = self.__S__.forward(data)

        if label == None:
            return outp

        else:
            outp = outp.reshape(1, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)
            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1
        
def train_network(train_loader, main_model, optimizer, scheduler, num_epoch, verbose=False):
    main_model.train()

    loss    = 0
    top1    = 0
    counter = 0

    for data, data_label in train_loader:
        data = data.transpose(1, 0)
        data = data.cuda()
        data_label = data_label.cuda()

        optimizer.zero_grad()
        nloss, prec1 = main_model(data, data_label)
        nloss.backward()
        optimizer.step()

        loss += nloss.item()
        top1 += prec1.item()
        counter += 1
        
        if verbose:
            print("Epoch {:1.0f}, Batch {:1.0f}, LR {:f} Loss {:f}, Accuracy {:2.3f}%".format(num_epoch, counter, optimizer.param_groups[0]['lr'], loss/counter, top1/counter))

        scheduler.step()

    return (loss/counter, top1/counter)

def test_network(test_loader, main_model):
    main_model.eval()

    loss    = 0
    top1    = 0
    counter = 0

    for data, data_label in test_loader:
        data = data.transpose(1, 0)
        data = data.cuda()
        data_label = data_label.cuda()

        with torch.no_grad():
            nloss, prec1 = main_model(data, data_label)

        loss += nloss.item()
        top1 += prec1
        counter += 1

    return (loss/counter, top1/counter)

def saveParameters(model, optimizer, scheduler, num_epoch, path):
    checkpoint = {}
    checkpoint['model']     = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduler'] = scheduler.state_dict()
    checkpoint['num_epoch'] = num_epoch
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    torch.save(checkpoint, os.path.join(path, ''.join(['lab3_model_', str(num_epoch).zfill(4), '.pth'])))
    
def loadParameters(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['num_epoch']