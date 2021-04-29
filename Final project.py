import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class NeuralNetwork():
    def __init__(self):
        self.weights = np.random.random((5, 1))
        self.bias = random.randrange(0, 1)
        self.lr = 0.001

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_der(self, x):
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))

    def train(self, inputs, outputs, iterations):
        for _ in range(iterations):
            preOutput = self.learn(inputs)

            error = preOutput - outputs

            deriv = error * self.__sigmoid_der(preOutput)
            final_deriv = np.dot(inputs.T, deriv)
            self.weights = self.weights - self.lr * final_deriv

            for i in deriv:
                self.bias = self.bias - self.lr * i

    def learn(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.weights) + self.bias)


if __name__ == "__main__":
    model = NeuralNetwork()
    f = open('Data2.csv', 'r')
    data = f.read().split('\n')
    data.pop()

    inputs = []
    outputs = []

    for i in range(len(data)):
        d = list(map(int, data[i].split(',')))
        inputs.append(d[:5])
        outputs.append([d[5]])

    inputs = np.array(inputs, dtype=np.float64)
    outputs = np.array(outputs, dtype=np.float64)

    train_i, test_i, train_o, test_o = train_test_split(inputs, outputs, test_size=0.1)

    model.train(train_i, train_o, 10)

    train_pre = model.learn(train_i)
    accu_train = accuracy_score(train_o, train_pre.round())
    print("Accuracy of train data : ", accu_train)
    print(train_o.shape, train_pre.shape)
    test_pre = model.learn(test_i)
    accu_test = accuracy_score(test_o, test_pre.round())
    print("Accuracy of test data : ", accu_test)

    ### for one by one prediction ###

    print('Enter data to predict')
    pre = np.array(list(map(int, input().split())))

    while (True):
        result = model.learn(pre)[-1]
        print('Predicted result\n', result)
        if result == 1:
            print('Chance of cold or cough')
        elif result == 0:
            print('Chance of feaver')

        print('\nEnter data to predict or Press E to Exit')
        pre = input()
        if pre == 'e' or pre == 'E':
            break
        else:
            pre = np.array(list(map(int, pre.split())))