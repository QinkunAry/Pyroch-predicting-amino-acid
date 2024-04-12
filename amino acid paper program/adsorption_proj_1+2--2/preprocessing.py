import numpy as np
import pandas as pd

eyes = np.eye(20) # onehot coding
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}

def split():

    # list fasta ids
    fasta_list = []

    # load sequence and label
    with open('dataset/rawdata/sequence_label.txt') as r:
        # split sequence and label(for mapping other features like onehot, hhblits, pssm...)
        line = r.readline()
        while line:
            # fasta id
            fasta_id = line.strip()[1:]
            # sequence
            line = r.readline()
            sequence = line.strip()
            # label
            line = r.readline()
            label = line.strip()
            # write files
            with open('dataset/middlefile/fasta/' + fasta_id + '.fasta','w+') as w:
                w.write(sequence)
            with open('dataset/middlefile/label/' + fasta_id + '.label','w+') as w:
                w.write(label)
            fasta_list.append(fasta_id)
            line = r.readline()

    # write fasta ids
    with open('dataset/middlefile/dataset_list.txt','w+') as w:
        for i in fasta_list:
            w.write(i + '\n')

def feature_coding():

    # load single and double file
    single = pd.read_csv('dataset/rawdata/single.csv')
    double = pd.read_csv('dataset/rawdata/double.csv', keep_default_na=False)

    sample = double.shape[0]
    print('The number of training and validation samples: ',sample)

    x_list = []
    y_list = []
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
        y_list.append([float(double['value1'][i]),float(double['value2'][i])])
    
    x_data = np.array(x_list) # (sample,2,21)
    print(x_data)
    y_data = np.array(y_list) # (sample,2)

    np.save('dataset/middlefile/x_data.npy', x_data)
    np.save('dataset/middlefile/y_data.npy', y_data)

if __name__=='__main__':

    feature_coding()
