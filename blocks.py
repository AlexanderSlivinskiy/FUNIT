"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
from globalConstants import GlobalConstants
import skimage.io as sk

inplace_bool = False

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type, inception=False):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type,
                                    inception=inception)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', inception=False):
        super(ResBlock, self).__init__()
        model = []        

        #REMOVE:
        inception = False

        if (inception):
            model += [InceptionBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
            model += [InceptionBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        else:
            model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
            model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=inplace_bool)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=inplace_bool)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=inplace_bool)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=inplace_bool)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.pad(x)
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.pad(x)
            x = self.conv(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        isNotFloat = (x_reshaped.dtype != torch.float32)
        isNotFloatWeight = (self.weight.dtype != torch.float32)
        if (isNotFloat):
            x_reshaped = x_reshaped.float()
            if (isNotFloatWeight): #Should never be called since initialized as float()
                self.weight = self.weight.float()
                self.bias = self.bias.float()
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        out = out.view(b, c, *x.size()[2:])
        if (isNotFloat):
            out = GlobalConstants.setTensorToPrecision(out)
        return out

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class InceptionBlock(Conv2dBlock):
    """
    Is a copy of a copy of Conv2dBlock, but with Inception layers
    """
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding=0,
        norm='none', activation='relu', pad_type='zero',
        use_bias=True, activation_first=False, kernels = [
            1, 3, 5, "max_pooling"
        ]):

        super(InceptionBlock, self).__init__(in_dim, out_dim, kernel_size, stride,
        padding, norm, activation, pad_type, use_bias, activation_first)

        """
        Really not sure what sized to take for the Threads themselves
        We probably want to keep the 3x3 Convolutions most relevant
        so they should have half the output size
        1/2 3x3
        1/4 5x5
        1/8 1x1
        1/8 maxpool
        """
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.inceptionThreads = nn.ModuleList()
        layers_3Conv=int(out_dim*0.5)
        remaining_out_dim = out_dim - layers_3Conv
        layers_5Conv=int(out_dim*0.25)
        remaining_out_dim = remaining_out_dim - layers_5Conv
        layers_1Conv=int(out_dim*0.125)
        remaining_out_dim = remaining_out_dim - layers_1Conv
        layers_MaxPool=remaining_out_dim
    
        for size in kernels:
            if (size == 1):
                self.inceptionThreads.append(nn.Conv2d(in_dim, layers_1Conv, 1, 1, bias=self.use_bias))
            elif (size == 3):
                intermediate_dim = int(out_dim*0.75)
                conv = [
                    nn.Conv2d(in_dim, intermediate_dim, 1, 1, bias=self.use_bias),
                    ParallelConv2dBlock(intermediate_dim, layers_3Conv, 3, 1, bias=self.use_bias, padding_mode=self.pad_type),
                    #nn.Conv2d(intermediate_dim, layers_3Conv, 3, 1, bias=self.use_bias, padding=1)
                ]
                self.inceptionThreads.append(nn.Sequential(*conv))
            elif (size == 5):
                intermediate_dim = int(out_dim*0.5)
                conv = [
                    nn.Conv2d(in_dim, intermediate_dim, 1, 1, bias=self.use_bias),
                    nn.Conv2d(intermediate_dim, intermediate_dim, 3, 1, padding=1, padding_mode=self.pad_type, bias=self.use_bias),
                    ParallelConv2dBlock(intermediate_dim, layers_5Conv, 3, 1, bias=self.use_bias, padding_mode=self.pad_type),
                    #nn.Conv2d(intermediate_dim, layers_5Conv, 5, 1, bias=self.use_bias, padding=2)
                ]
                self.inceptionThreads.append(nn.Sequential(*conv))
            elif (size == "max_pooling"):
                conv = [
                    nn.MaxPool2d(3, 1, padding=1),
                    nn.Conv2d(in_dim, remaining_out_dim, 1, 1, bias=self.use_bias)
                ]
                self.inceptionThreads.append(nn.Sequential(*conv))

    def forward(self, x, log=False):
        #print("BLOCKS, INCEPTIONBLOCK: BE AWARE THAT THE CONV2DBLOCK DIDN'T PAD AS MUCH AS THIS ONE")
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.inceptionForward(x)
            if self.norm:
                x = self.norm(x)
        else:
            x = self.inceptionForward(x)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

    def inceptionForward(self, x):
        results = []
        res = None
        for thread in self.inceptionThreads:
            y = thread(x)
            if res is None:
                res = y
            else:
                res = torch.cat((res, y), dim=1)
            results += y
        return res

class ParallelConv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, bias=True, padding_mode='zeros'):
        super(ParallelConv2dBlock, self).__init__()

        left_dim = int(0.5*out_dim)
        right_dim = out_dim - left_dim
        
        self.left = nn.Conv2d(in_dim, left_dim, (1,3), stride=1, padding=(0,1), padding_mode=padding_mode, bias=bias)
        self.right = nn.Conv2d(in_dim, right_dim, (3,1), stride=1, padding=(1,0), padding_mode=padding_mode, bias=bias)

    def forward(self, x):
        l = self.left(x)
        r = self.right(x)
        return torch.cat((l,r), dim=1)

class Printer(nn.Module):
    def __init__(self, perma_save_counter = 10000, running_save_counter = 100):
        super(Printer, self).__init__()

        """
        1. 
        Think about how you to structure naming.
        Best practice would be probably to hand to the inception layer the caller-class.
            But what about ResBlock? They also would need to pass on this parameter
            I don't really want to introduce new parameters that are not related to the Layer
            Or you have some intermediate static class like globalsConstants that you mistreat for this
        You also need some index if the same class creates various InceptionBlocks

        2. 
        Each x is a batch -> more than one pic. That should probably be different sequences
        Or the same one in a grid way (which sounds like a lot of work)

        3.
        How to save something as a sequence lol

        4.
        What to do about others parallel Layers in the same Inception?
        Do you want to depict them in the same sequence File or another one?
        Probably we could also have one folder per InceptionBlock

        """
        self.name = "?"
        
        self.counter = 0
        try:
            self.outputPath = GlobalConstants.getOutputPath()
        except:
            print("produced an error")
            self.outputPath = None
        
        self.perma_save_counter = perma_save_counter
        self.running_save_counter = running_save_counter

    def forward(self, x):
        if counter%running_save_counter==0:
            print("Leave Britney alone! This is not implemented!")


        self.counter += 1