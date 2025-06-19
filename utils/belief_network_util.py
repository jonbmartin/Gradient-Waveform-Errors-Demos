import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import sys
import shutil
import copy
import re
from collections import OrderedDict

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn.utils import weight_norm

# distributed computation utilities
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

def MSEloss_with_integral(output, target):
    loss = nn.MSELoss(output,target) + nn.MSELoss(torch.cumsum(output,0),torch.cumsum(target,0))
    return loss


class BeliefModelBidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers=2, lstm_dropout=0.35):
        super().__init__()
        # IMPORTANT NOTE. If num_layers = 1, there will be no dropout applied within
        # the LSTM layers. Must be >= 2. 
        # self.linear1 = nn.Linear(input_size, hidden_size)
        # self.act1 = nn.SiLU()
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size, num_layers=num_lstm_layers, 
                             batch_first=True, dropout=lstm_dropout, bidirectional=True)
        # formerly got good results with dropout=0.3
        self.linear4 = nn.Linear(hidden_size*2, 1)

    def forward(self, x):
        # x = self.act1(self.linear1(x))
        # x = self.linear2(x)
        x, _ = self.lstm(x)
        x = self.linear4(x)
        return x

class BeliefModelBidirectionalSimple(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers=2, lstm_dropout=0.35):
        super().__init__()
        # IMPORTANT NOTE. If num_layers = 1, there will be no dropout applied within
        # the LSTM layers. Must be >= 2. 
        self.lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size, num_layers=num_lstm_layers, 
                             batch_first=True, dropout=lstm_dropout, bidirectional=False)
        # formerly got good results with dropout=0.3
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x = self.act1(self.linear1(x))
        # x = self.linear2(x)
        x, _ = self.lstm(x)
        x = self.linear4(x)
        return x


class BeliefModelUnidirectional(nn.Module):
    def __init__(self, input_size, hidden_size, num_lstm_layers=2, lstm_dropout=0.35):
        super().__init__()
        # IMPORTANT NOTE. If num_layers = 1, there will be no dropout applied within
        # the LSTM layers. Must be >= 2. 
        self.lstm = nn.LSTM(input_size=input_size,
                             hidden_size=hidden_size, num_layers=num_lstm_layers, 
                             batch_first=True, dropout=lstm_dropout, bidirectional=False)
        # formerly got good results with dropout=0.3
        self.linear4 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear4(x)
        return x

class LSTM(nn.Module):
# very simple LSTM from Gridin 2022. 

    def __init__(self,
                 hidden_size,
                 num_layers,
                 dropout,
                 in_size = 1,
                 out_size = 1):
        super(LSTM, self).__init__()
        self.model_name = "LSTM"
        self.lstm = nn.LSTM(
            input_size = in_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            batch_first = True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h = None):
        out, h = self.lstm(x, h)
        last_hidden_states = out[:, -1]
        out = self.fc(last_hidden_states)
        return out, h

class Crop(nn.Module):
# crop layer is responsible for trimming the tensor from the right when creating
# a casual convolution operation. Source: Gridin 2022

    def __init__(self, crop_size):
        super(Crop, self).__init__()
        self.crop_size = crop_size

    def forward(self, x):
        return x[:, :, :-self.crop_size].contiguous()

class TemporalCasualLayer(nn.Module):
# Source: Gridin 2022

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout = 0.2):
        super(TemporalCasualLayer, self).__init__()
        padding = (kernel_size - 1) * dilation
        conv_params = {
            'kernel_size': kernel_size,
            'stride':      stride,
            'padding':     padding,
            'dilation':    dilation
        }

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, **conv_params))
        self.crop1 = Crop(padding)
        self.relu1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, **conv_params))
        self.crop2 = Crop(padding)
        self.relu2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.crop1, self.relu1, self.dropout1,
                                 self.conv2, self.crop2, self.relu2, self.dropout2)

        self.bias = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.GELU()

    def forward(self, x):
        y = self.net(x)
        b = x if self.bias is None else self.bias(x)
        return self.relu(y + b)

class TemporalConvolutionNetwork(nn.Module):
# Source: Gridin 2022
    def __init__(self, num_inputs, num_channels, kernel_size = 2, dropout = 0.2):
        super(TemporalConvolutionNetwork, self).__init__()
        layers = []
        num_levels = len(num_channels)
        tcl_param = {
            'kernel_size': kernel_size,
            'stride':      1,
            'dropout':     dropout
        }
        for i in range(num_levels):
            dilation = 2**i
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            tcl_param['dilation'] = dilation
            tcl = TemporalCasualLayer(in_ch, out_ch, **tcl_param)
            layers.append(tcl)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
# Same as Temporal ConvolutionNetwork class, but with a linear output layer to collect for prediction
# Source: Gridin 2022

    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.model_name = "TCN"
        self.tcn = TemporalConvolutionNetwork(input_size, num_channels, kernel_size = kernel_size, dropout = dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        y = self.tcn(x)
        return self.linear(y[:, :, -1])


class RNN(nn.Module):

    def __init__(self, hidden_size, num_layers, nonlinearity, dropout, in_size = 1, out_size = 1):
        super(RNN, self).__init__()
        self.model_name = "RNN"
        self.rnn = nn.RNN(
            input_size = in_size,
            hidden_size = hidden_size, 
            num_layers = num_layers,
            nonlinearity = nonlinearity,
            dropout = dropout,
            batch_first = True)
        self.fc = nn.Linear(hidden_size, out_size)

    def forward(self, x, h = None):
        out, _ = self.rnn(x, h)
        last_hidden_states = out[:, -1]
        out = self.fc(last_hidden_states)
        return out, last_hidden_states


############################################
# TRAINING UTILITIES 
############################################

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# master function to control training
def train_and_test_belief_network(n_epochs, model, loss_fn, optimizer, dataloader, X_train, X_test, y_train, y_test, device, verbose=True, plotting=True, eval_interval=10):
    if verbose:
    	print("Training and testing belief network")

    scaler = GradScaler()

    training_loss = []
    test_loss = []
    min_test_loss = sys.maxsize

    # send everything to the appropriate device
    model.to(device)
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)
    print(X_train.size())


    for epoch in range(n_epochs):
    	# perform network training
        model.train()
        train_cost = 0
        n_batches = 0
        for X_batch, y_batch in dataloader:
            with autocast():
                optimizer.zero_grad()
                if model.model_name == "TCN":
                    y_pred = model(X_batch)
                else:
                    # Recurrent nets return hidden as well
                    y_pred, _ = model(X_batch)

                loss = loss_fn(y_pred, y_batch)


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_cost += loss
            n_batches +=1
        train_cost /= n_batches

	    # Validation
        # if epoch % eval_interval != 0:
        #     continueput p
        model.eval()
        with torch.no_grad():
            if model.model_name == "TCN":
                y_pred = model(X_test)
            else:
                # Recurrent nets return hidden as well
                y_pred, _ = model(X_test)
            test_cost = loss_fn(y_pred.squeeze(), y_test.squeeze())

            training_loss.append(train_cost.item())
            test_loss.append(test_cost.item())
            if test_cost.item() < min_test_loss:
                best_params = copy.deepcopy(model.state_dict())
                min_test_loss = test_cost.item()
                if verbose:
                    print('New best test cost: ')


        if verbose:
        	print("Epoch %d: train cost %.2E, test cost %.2E" % (epoch, train_cost.item(), test_cost.item()))

    if plotting:
        plt.title('Training Progress')
        plt.yscale("log")
        plt.plot(training_loss, label = 'train')
        plt.plot(test_loss, label = 'test')
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    return model, min_test_loss, best_params


def create_dataset(dataset_x, dataset_y, window_size, predict_point, pct_data_to_keep=1, subtract_baseline=False, 
                   predict_single_timepoint=False, return_traj_y=True):
    """Transform a time series into a prediction dataset. Output type float.
    Note that this is predicting the CUMULATIVE SUM OF THE INPUT GRADIENT

    Args:
        dataset_x: A numpy array of time series, first dimension is the time steps
        window_size: Size of window for prediction
        predict_point: point in window to predict. 0 first point in window
        window_spacing: Don't necessarily need continuous windows. Separate start point by n samples
        pct_data_to_keep: Don't necessarily need continuous windows. But also don't want
            regular window spacing, as this can sync up with dynamics. Randomly disperse
            samples instead
        subtract_baseline: if true, start timeseries target y at 0 for each window
        return_traj_y: if true, y is the INTEGRAL of the observed gradient up to the midpoint of the window
    """

    # check to make sure that predict_point is valid
    if predict_point > window_size:
        raise ValueError('predict_point must be less than window_size')
    if predict_point < 0:
        raise ValueError('predict_point must be greater than 0')
    
    X, y = [], []
    # What to do if window size is greater than the length of the dataset?? pad with 0's in that dim

    if window_size > len(dataset_x)-1:
        while window_size > len(dataset_x)-1:
            dataset_x = np.concatenate((dataset_x, np.zeros((1,3))),axis=0)
            dataset_y = np.concatenate((dataset_y, np.zeros((1,1))),axis=0)

    target_cumsum = np.cumsum(dataset_y)
    
    # get the indices that will be kept
    data_length = len(dataset_x)-window_size
    if pct_data_to_keep == 1:
        # use all data in dataset
        samples_to_keep = range(0, data_length, 1)
    else:
        # use a randomly selected subset
        samples_to_keep = random.sample(range(data_length),int(pct_data_to_keep*data_length))
        samples_to_keep.sort()
    print(samples_to_keep)
    for i in samples_to_keep:
        feature = dataset_x[i:i+window_size,:]

        # predict a specified point in the window
        if predict_single_timepoint:
            if return_traj_y:
                # NOTE: predicting the GRADIENT INTEGRAL AT WINDOW MIDPOINT
                # looking at the FIRST DIFFERENCE of the cumsum
                target = target_cumsum[i+predict_point+1]-target_cumsum[i+predict_point]
            else:
                target = dataset_y[i+predict_point]
        else:
            print('error: predict multiple timepoints not implemented')

        X.append(feature)
        y.append(target)

    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y))
    #X = X.type(torch.FloatTensor)
    #y = y.type(torch.FloatTensor)


    return X, y

def get_windowed_integral(input_timeseries, window_size, window_spacing):
    output_timeseries = []
    # need to pad input time series so that output is the same size


    # returns the integral over a specified window at a specified timepoint
    for i in range(0, len(input_timeseries)-window_size, window_spacing):

        # NOTE: predicting the GRADIENT INTEGRAL AT WINDOW MIDPOINT
        target = np.sum(input_timeseries[i:i+window_size//2])/(window_size//2)
        output_timeseries.append(target)
        #target = dataset_y[i+lookback//2]
    return np.array(output_timeseries)



def save_checkpoint(state, checkpoint_dir):
    f_path = checkpoint_dir
    torch.save(state, f_path)
        

def load_checkpoint(checkpoint_fpath, model, optimizer, distributed=False):
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cpu'))

    # if want to load DDP saved model to non-DDP model, must strip "module" prefix
    if distributed:
        model_dict = OrderedDict()
        state_dict = checkpoint['state_dict']
        pattern = re.compile('module.')

        for k, v in state_dict.items():
            if re.search("module",k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(checkpoint['state_dict'])

    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']
