from model.simpleNN import FCNet
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.getcwd())
from random import shuffle
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
from torch.utils.data import Dataset,DataLoader,TensorDataset
import warnings
warnings.filterwarnings('ignore')
import random
import json

eyes = np.eye(20) # onehot coding
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}

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

def AA_mapping(test_file):
    single = pd.read_csv('dataset/rawdata/single.csv')
    double = pd.read_csv('dataset/rawdata/double_need_pred.csv', keep_default_na=False)
    sample = double.shape[0]
    x_list = []
    x_sample = np.zeros((2,21),dtype=float)
    # generate datasets
    for i in range(sample):
        AA = double['AA'][i]
        x_sample = np.zeros((2,21),dtype=float)
        x_sample[0][-1] = single[single['AA'] == AA[0]]['value1']
        x_sample[0][0:20] = protein_dict[AA[0]]
        x_sample[1][-1] = single[single['AA'] == AA[1]]['value1']
        x_sample[1][0:20] = protein_dict[AA[1]]
        x_list.append(x_sample)
        
    x_data = np.array(x_list) # (sample,2,21)
    np.save('dataset/middlefile/test.npy', x_data)


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

    return total_raw_preds

def generate_pred(feature_dim1,feature_dim2,save_path):

    # generate pseudo labels

    cuda_name = "cuda:0"
    if len(sys.argv)>3:
        cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
    print('cuda_name:', cuda_name)
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    
    x_test = np.load('dataset/middlefile/test.npy')
    y_test = np.zeros((x_test.shape[0]),dtype=float) # Simulated value

    # convert to tensor
    x_test, y_test = np.asarray(x_test), np.asarray(y_test)
    x_test, y_test = torch.from_numpy(x_test.astype('float')), torch.from_numpy(y_test.astype('float'))

    test_data = TensorDataset(x_test,y_test)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

    model_file_name = 'model/weights/weights.model'
    model = FCNet(feature_dim1,feature_dim2).to(device)
    model.load_state_dict(torch.load(model_file_name))
    print('predicting for test data')
    P = predicting(model, device, test_loader)
    np.save('dataset/middlefile/prediction.npy',P)
    print(P)
    print(P.shape)

    pred = np.load('dataset/middlefile/prediction.npy')
    double = pd.read_csv('dataset/rawdata/double_need_pred.csv', keep_default_na=False)
    for i in range(double.shape[0]):
        double['value1'][i] = pred[i][0]
        double['value2'][i] = pred[i][1]
    double.to_csv('dataset/rawdata/double_pred_result.csv',index=False)

if __name__=='__main__':
    
    feature_dim1, feature_dim2 = 2, 21
    AA_mapping('dataset/rawdata/double_need_pred.csv')
    generate_pred(feature_dim1,feature_dim2,'dataset/rawdata/double_pred_result.csv')