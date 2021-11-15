import numpy as np
import random

def preprocess(filename, label_dict):
    dataset = []
    with open(filename, 'r') as datafile:
        while True:
            line = datafile.readline()
            tgt = line.split(',')
            if len(tgt) != 5:
                break
            feature = np.array(tgt[:-1], dtype='float')
            label = label_dict[tgt[-1].strip('\n')]
            dataset.append(np.hstack((feature, label)))
    data = np.array(dataset)
    sampler = random.sample(range(data.shape[0]), int(data.shape[0] * 2/3))
    rest = []
    for i in range(data.shape[0]):
        if i not in sampler:
            rest.append(i)
    # rest = np.random.shuffle(rest)
    # print(data.shape)
    np.save('data/train_data.npy', data[sampler][:])
    np.save('data/val_data.npy', data[rest][:])

if __name__ == '__main__':
    file = 'data/iris.txt'
    label_diction = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}

    preprocess(file, label_diction)