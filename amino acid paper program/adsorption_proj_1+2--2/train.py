from model.simpleNN import FCNet
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from random import shuffle
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import Dataset,DataLoader,TensorDataset
import warnings
warnings.filterwarnings('ignore')
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(10)

def softmax(vec):
    """Compute the softmax in a numerically stable way."""
    vec = vec - np.max(vec)  # softmax(x) = softmax(x+c)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

# training function at each epoch
def train_FCNet(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL):
    print('Training on {} samples...'.format(len(train_loader.dataset)))

    for batch_idx, data in enumerate(train_loader):

        feature, label = data[0], data[1]
        feature = feature.to(torch.float32)
        label = label.to(torch.float32)
        feature, label = feature.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(feature)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [({:.0f}%)]\tLoss: {:.6f}\t'.format(epoch,
                                                                        100. * batch_idx / len(train_loader),
                                                                        loss.item(),
                                                                        ))

def predicting(model, device, loader):
    model.eval()
    total_labels = torch.Tensor()
    total_raw_preds = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            feature, label = data[0], data[1]
            feature = feature.to(torch.float32)
            label = label.to(torch.float32)
            feature, label = feature.to(device), label.to(device) 

            output = model(feature)
            total_labels = torch.cat((total_labels, label.cpu()), 0)
            total_raw_preds = torch.cat((total_raw_preds, output.cpu()), 0)

    return total_labels.flatten().numpy(),total_raw_preds.flatten()

def train():

    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 32
    TEST_BATCH_SIZE = 32
    LR = 0.0005
    LOG_INTERVAL = 5
    NUM_EPOCHS = 200

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    x_file = 'dataset/middlefile/x_data.npy'
    y_file = 'dataset/middlefile/y_data.npy'

    feature_dim1, feature_dim2 = 2, 21

    if ((not os.path.isfile(x_file)) or (not os.path.isfile(y_file))):
        print('please run 01_preprocessing.py to prepare dataset.')
    else:

        x_train = np.load(x_file)
        y_train = np.load(y_file)

        # convert to tensor
        x_train, y_train = np.asarray(x_train), np.asarray(y_train)
        x_train, y_train = torch.from_numpy(x_train.astype('float')), torch.from_numpy(y_train.astype('float'))
        
        train_data = TensorDataset(x_train, y_train)
        
        # auto split
        train_size = int(0.9 * len(train_data))
        valid_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

        # data loader
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=VALID_BATCH_SIZE, shuffle=False)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = FCNet(feature_dim1, feature_dim2).to(device)
        print(model)
        loss_fn = nn.MSELoss() # Mean Squared Error
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 200000
        best_epoch = -1
        model_file_name = 'model/weights/weights.model'

        for epoch in range(NUM_EPOCHS):
            train_FCNet(model, device, train_loader, optimizer, epoch+1, loss_fn, LOG_INTERVAL)
            print('predicting for valid data')
            G,T = predicting(model, device, valid_loader)
            #print(G) # int labels
            #print(T) # predictions           
            val = mean_squared_error(G,T)
            if val < best_mse:
                ret = [epoch, r2_score(G,T), mean_squared_error(G,T)]
                best_mse = val
                best_epoch = epoch+1
                torch.save(model.state_dict(), model_file_name)
                print('predicting for test data')
                G,T = predicting(model, device, valid_loader)
                print('mse decreased at epoch ', best_epoch, '; best_mse, best_r2:', ret[2], ret[1])
            else:
                print('No improvement since epoch ', best_epoch, '; best_mse, best_r2:', ret[2], ret[1])         

if __name__=='__main__':
    
    # lr=0.0005
    train()