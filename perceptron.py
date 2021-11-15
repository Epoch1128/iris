import numpy as np


class perceptron(object):
    def __init__(self, in_dim, out_dim, lr, bias=True) -> None:
        super(perceptron, self).__init__()
        self.dim = in_dim + 1 if bias is True else in_dim
        self.lr = lr
        self.bias = bias
        self.out_dim = out_dim
        self.weight = [np.random.random(size=(self.dim, 1)) for _ in range(out_dim - 1)]

    def train(self, data):
        # first classify between 0 and 1 & 2
        for i in range(self.out_dim - 1):
            # initialization: every sample is not legal
            status = [False for _ in range(data.shape[0])]
            while False in status:
                for idx, item in enumerate(data):
                    if item[-1] != i:
                        y = -1
                    else:
                        y = 1
                    item[-1] = 1
                    # print(item.reshape(-1, 1).shape)
                    if y * np.matmul(np.transpose(self.weight[i]), item) < 0:
                        status[idx] = False
                        self.weight[i] = self.weight[i] - self.lr * item.reshape(-1, 1)
                    else:
                        status[idx] = True

        return self.weight

    def validation(self, data):
        pos = 0
        for item in data:
            x = item
            x[-1] = 1
            for i in range(self.out_dim - 1):
                result = np.matmul(np.transpose(self.weight[i]), x)
                if result > 0:
                    if i == item[-1]:
                        pos += 1
                        break
                if result < 0 and i == self.out_dim - 2 and i + 1 == item[-1]:
                    pos += 1
        return pos / data.shape[0]


if __name__ == '__main__':
    train_data = np.load('data/train_data.npy')
    val_data = np.load('data/val_data.npy')
    # print(train_data.shape)
    # print(val_data.shape)
    model = perceptron(4, 3, 0.1, bias=True)
    train_result = model.train(train_data)
    print("w1:{} \nw2:{}".format(train_result[0].transpose(), train_result[1].transpose()))
    val_result = model.validation(val_data)
    print("val result for perceptron: {}".format(val_result))