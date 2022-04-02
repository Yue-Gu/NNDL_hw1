import numpy as np
from numpy.linalg import eig
import pandas as pd
import matplotlib.pyplot as plt
import math
# from tqdm.notebook import tqdm
import struct

class NeuralNetwork:
    def __init__(self, input_num, hidden_num, output_num, lr, epoch, l2_lambda):
        """
        :param input_num: 输入层节点个数
        :param hidden_num: 输出层节点个数
        :param output_num: 输出层节点个数
        :param lr: 学习率
        :param epoch: 迭代次数
        """
        self.input_num = input_num
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.learning_rate = lr
        self.epoch = epoch
        self.l2_lambda = l2_lambda

        # 参数初始化
        self.W1 = np.random.randn(self.hidden_num, self.input_num) * 0.01
        self.W2 = np.random.randn(self.output_num, self.hidden_num)*0.01
        self.b1 = np.zeros(self.hidden_num)
        self.b2 = np.zeros(self.output_num)

    def lossFunction(self, label, pred, W1, W2, l2_lambda):
        """
        :param label: 标签列, np.array
        :param pred: 预测值, np.array
        :param W1: 参数权重矩阵
        :param W2:
        :param l2_lambda: 正则项惩罚力度
        :return: 损失函数返回值，考虑正则项
        """

        # l2对W求导后为W
        l2 = np.sum(W1 ** 2)/2 + np.sum(W2 ** 2)/2
        origin_loss = np.sum((label - pred)**2)/(2)
        total_loss = origin_loss + l2_lambda * l2

        return total_loss

    def softmax(self, x):
        """
        :param naive_output: 要输入到函数中的np.array
        :return: softmax结果
        """

        # s = 1.0/(1.0 + np.exp(-naive_output))
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)
        # return s

    def ReLU(self, x):
        """
        :param X:
        :return:
        """
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(x_ravel[index])
            else:
                y.append(0)
        return np.array(y).reshape(x.shape)

    def fowardPropagation(self, W, b, x):
        """
        :param W: 权重矩阵
        :param b: 常数项
        :param x: 输入向量
        :return: 返回两个向量，一个是Wx+b，另一个是softmax之后的结果
        """
        z = np.add(np.dot(W, x), b)
        a = self.softmax(z)
        return z, a

    def BP_softmax(self, x):
        """
        :param x: softmax输出值向量
        :return: BP结果
        """
        bp_softmax = x*(1-x)
        # x_ravel = x.ravel()  # 将numpy数组展平
        # length = len(x_ravel)
        # y = []
        # for index in range(length):
        #     if x_ravel[index] >= 0:
        #         y.append(1)
        #     else:
        #         y.append(0)
        # return np.array(y).reshape(x.shape)
        return bp_softmax


    def BP_z2(self, z2, a2, label0):
        """
        :param z2: output_num维向量，z2 = W2*a1+b2
        :param a2: softmax(z2)
        :param label0: label array, 维数为output_num
        :return: L对z2求导结果
        """
        # 链式法则，a2-a0是损失函数对a2求导的结果，bp_softmax(z2)是a2对z2求导的结果
        return (a2-label0)*self.BP_softmax(a2)
        # return (a2-label0)

    def BP_z1(self, z1, W2, delta_z2):
        """
        :param z1: z1 = W1*a0+b1
        :param W2:
        :param delta_z2: L对z2求导的结果
        :return: L对z1求导的结果
        """
        # partial_L_z2 = delta_z2
        # # z2 = W2*a1+b2
        # partial_z2_a1 = W2
        # a1 = softmax(z1)
        # partial_a1_z1 = self.BP_softmax(z1)
        a1 = self.softmax(z1)
        return self.BP_softmax(a1)*(np.dot(W2.T, delta_z2))

    def BP_W2(self, delta_z2, a1, l2_lambda, W2):
        """
        :param delta_z2: L对z2求导的结果
        :param a1: a1=softmax(z1)
        :param l2_lambda: L2正则项系数
        :param W2: 当前第二层权重矩阵参数估计
        :return: L对W2求导的结果
        """
        return np.outer(delta_z2, a1) + l2_lambda * W2

    def BP_W1(self, delta_z1, a0, l2_lambda, W1):
        """
        :param delta_z1:
        :param a0:
        :param l2_lambda:
        :param W1:
        :return:
        """
        return np.outer(delta_z1, a0) + l2_lambda * W1

    def trainNN(self, inputdata, labeldata, testdata, testlabel):
        train_loss = []
        test_loss = []
        test_accuracy = []
        for epo in range(self.epoch):
            print('Epoch%d' % epo)
            naive_loss_train = 0
            for i in range(inputdata.shape[0]):
                a0 = inputdata[i, :]
                label0 = labeldata[i, :]

                # 向前传播：
                z1, a1 = self.fowardPropagation(self.W1, self.b1, a0)
                z2, a2 = self.fowardPropagation(self.W2, self.b2, a1)
                # 向后传播梯度
                dz2 = self.BP_z2(z2, a2, label0)
                dz2 = a2-label0
                dz1 = self.BP_z1(z1, self.W2, dz2)
                dW2 = self.BP_W2(dz2, a1, self.l2_lambda, self.W2)
                dW1 = self.BP_W1(dz1, a0, self.l2_lambda, self.W1)

                self.W2 -= self.learning_rate * dW2
                self.W1 -= self.learning_rate * dW1
                self.b2 -= self.learning_rate * dz2
                self.b1 -= self.learning_rate * dz1

                # naive_loss_train += np.sum((label0 - a2)**2)/2
                naive_loss_train += self.lossFunction(label0, a2, self.W1, self.W2, self.l2_lambda)
            testacc, testlos = self.testNN(testdata, testlabel)
            test_accuracy.append(testacc)
            test_loss.append(testlos)
            train_loss.append(naive_loss_train/inputdata.shape[0])
            print('After Epoch%d, the train loss is %f, the test loss is %f and the test accuracy is %f' % (epo, naive_loss_train/inputdata.shape[0], testlos, testacc))
            print('The learning rate in Epoch%d is %f' % (epo, self.learning_rate))
            self.learning_rate = self.learning_rate * (0.9)**(epo)
        return self.W1, self.W2, self.b1, self.b2, train_loss, test_loss, test_accuracy

    def testNN(self, testdata, testlabel):
        accuracy = 0
        loss = 0
        for i in range(testdata.shape[0]):
            test_a0 = testdata[i, :]
            test_a1 = self.softmax(np.add(np.dot(self.W1, test_a0), self.b1))
            test_a2 = self.softmax(np.add(np.dot(self.W2, test_a1), self.b2))
            test_pred0 = np.argmax(test_a2)
            test_label0 = np.argmax(testlabel[i, :])
            # loss += np.sum((test_label0 - test_a2)**2)/2
            loss += self.lossFunction(testlabel[i, :], test_a2, self.W1, self.W2, self.l2_lambda)
            if test_pred0 == test_label0:
                accuracy += 1
        return accuracy/testdata.shape[0], loss/testdata.shape[0]

def modelSelection(lr_list, hiddennum_list, l2lambda_list, x_train, y_train, x_test, y_test):
    """
    :param lr_list:
    :param hiddennum_list:
    :param l2lambda_list:
    :return:
    """
    best_params = [0, 0, 0]
    best_acc = 0
    for lr0 in lr_list:
        for hiddennum0 in hiddennum_list:
            for l2lambda0 in l2lambda_list:
                dl = NeuralNetwork(784, hiddennum0, 10, lr0, 5, l2lambda0)
                output = dl.trainNN(x_train.T, y_train.T, x_test.T, y_test.T)
                acc0 = output[6][-1]
                if acc0 > best_acc:
                    best_acc = acc0
                    best_params = [lr0, hiddennum0, l2lambda0]
    return best_params

def softmax(x):
    """
    :param naive_output: 要输入到函数中的np.array
    :return: softmax结果
    """

    # s = 1.0/(1.0 + np.exp(-naive_output))
    x_ravel = x.ravel()  # 将numpy数组展平
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)

def testModel(W1, W2, b1, b2, testdata, testlabel):
    """
    :param W1:
    :param W2:
    :param b1:
    :param b2:
    :param testdata:
    :param testlabel:
    :return: accuracy
    """
    accuracy = 0
    loss = 0
    for i in range(testdata.shape[0]):
        test_a0 = testdata[i, :]
        test_a1 = softmax(np.add(np.dot(W1, test_a0), b1))
        test_a2 = softmax(np.add(np.dot(W2, test_a1), b2))
        test_pred0 = np.argmax(test_a2)
        test_label0 = np.argmax(testlabel[i, :])
        if test_pred0 == test_label0:
            accuracy += 1
    return accuracy / testdata.shape[0]

# 读取原始数据并进行预处理
def data_fetch_preprocessing():
    train_image = open('train-images.idx3-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',
                             train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,
                                         dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    # y_test = np.fromfile(test_label,
    #                      dtype=np.uint8).reshape(10000, 1)
    y_test_label = np.array(np.fromfile(test_label,
                                         dtype=np.uint8), ndmin=1)
    y_test = np.ones((10, 10000)) * 0.01
    for i in range(10000):
        y_test[y_test_label[i]][i] = 0.99

    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    # x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784).T
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test_label), 784).T
    # print(x_train.shape)
    # 可以通过这个函数观察图像
    data=x_train[:,1].reshape(28,28)
    plt.imshow(data,cmap='Greys',interpolation=None)
    plt.savefig("example.png")
    plt.show()

    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test

def lossPlot(train_loss, test_loss, epoch):
    """
    :param train_loss:
    :param test_loss:
    :return:
    """
    plt.xlabel('Epoch')  # x轴标题
    plt.ylabel('Loss')  # y轴标题

    plt.plot(epoch, train_loss, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小
    plt.plot(epoch, test_loss, marker='o', markersize=3)

    plt.legend(['train loss', 'test loss'])  # 设置折线名称

    plt.savefig("lossplot.png")
    plt.show()  # 显示折线图

def accPlot(acc, epoch):
    plt.xlabel('Epoch')  # x轴标题
    plt.ylabel('Accuracy')  # y轴标题

    plt.plot(epoch, acc, marker='o', markersize=3)  # 绘制折线图，添加数据点，设置点的大小

    plt.legend(['test accuracy'])  # 设置折线名称

    plt.savefig("accplot.png")
    plt.show()  # 显示折线图

def pcaPrincipleComponent(X, k):
    """
    :param X: 待进行PCA的numpy矩阵
    :param k: 提取主成分个数
    :return: k个主成分向量
    """
    X = X - X.mean(axis=0)  # 向量X去中心化
    X_cov = np.cov(X.T, ddof=0)
    eigenvalues, eigenvectors = eig(X_cov)
    klarge_index = eigenvalues.argsort()[-k:][::-1]  # 选取最大的K个特征值及其特征向量
    k_eigenvectors = eigenvectors[klarge_index]  # 用X与特征向量相乘
    return np.dot(X, k_eigenvectors.T)

def greyScale(vec, width, length, name0):
    """
    :param vec:
    :param width:
    :param length:
    :return:
    """
    a, b = 0, 255
    Ymax = max(vec)
    Ymin = min(vec)
    k = (b - a) / (Ymax - Ymin)
    norY = a + k * (vec - Ymin)
    data = norY.reshape(width, length)
    plt.imshow(data, cmap='Greys', interpolation=None)
    plt.savefig(name0+".png")
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()

    # PART 1
    # 参数选择
    # bestparams = modelSelection(lr_list=[0.05, 0.1, 0.2], hiddennum_list=[100,200,300], l2lambda_list=[0,0.005,0.01], x_train=x_train1, y_train=y_train1, x_test=x_test1, y_test=y_test1)

    # PART 2
    # 最优参数组合训练
    dl = NeuralNetwork(784, 200, 10, 0.05, 12, 0)
    output = dl.trainNN(x_train.T, y_train.T, x_test.T, y_test.T)

    # PART 3
    # 结果储存
    np.savetxt('W1.txt', output[0])
    np.savetxt('W2.txt', output[1])
    np.savetxt('b1.txt', output[2])
    np.savetxt('b2.txt', output[3])
    np.savetxt('train_loss.txt', output[4])
    np.savetxt('test_loss.txt', output[5])
    np.savetxt('test_acc.txt', output[6])

    # PART 4
    # 导入模型并测试
    W1 = np.loadtxt('W1.txt')
    W2 = np.loadtxt('W2.txt')
    b1 = np.loadtxt('b1.txt')
    b2 = np.loadtxt('b2.txt')
    testacc0 = testModel(W1, W2, b1, b2, x_test.T, y_test.T)
    print('Test Accuracy%f' % (testacc0))

    # PART 5
    # 可视化训练和测试的loss曲线，测试的accuracy曲线
    trainloss0 = np.loadtxt('train_loss.txt', dtype=np.float)
    testloss0 = np.loadtxt('test_loss.txt', dtype=np.float)
    testacc0 = np.loadtxt('test_acc.txt', dtype=np.float)
    lossPlot(trainloss0, testloss0, range(12))
    accPlot(testacc0, range(12))

    # PART 6
    # 可视化网络参数
    w1 = np.loadtxt('w1.txt')
    w2 = np.loadtxt('w2.txt')
    w1_pca = pcaPrincipleComponent(w1.T, 10)
    w = np.dot(w1.T, w2.T)
    greyScale(w1_pca[:, 0], 28, 28, "w1")
    for i in range(10):
        name0 = "w2_"+str(i)
        greyScale(w[:, i], 28, 28, name0)


