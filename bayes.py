import numpy as np
import math

def sig_normal(u, sig, x):
    return np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)

train_data = np.load("data/train_data.npy")
val_data = np.load("data/val_data.npy")

data_list = [[], [], []]
for item in train_data:
    data_list[int(item[-1])].append(item)

class_list = []
for i in range(len(data_list)):
    class_list.append(np.array(data_list[i]))

info_list = []
for item in class_list:
    info_list.append([np.mean(item, axis=0), np.std(item, axis=0), item.shape[0] / train_data.shape[0]])
# info_list [[m,s,p], [], []]

# val process
# y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
total = 0
for item in val_data:
    idx = -1
    probability = -math.inf
    for i in range(len(info_list)):
        prob = info_list[i][-1]
        for j in range(4):
            prob = prob * sig_normal(info_list[i][0][j], info_list[i][1][j], item[j])
        if prob > probability:
            probability = prob
            idx = i
    if idx == int(item[-1]):
        total += 1

print("Val result for naive bayes: {}".format(total / val_data.shape[0]))
