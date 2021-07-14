"""
# neuralnetwork.py
# modified by Robin 2015/03/03
https://www.cnblogs.com/hhh5460/p/4310083.html
"""

import sys
import copy
import random
from math import exp  # , pow
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.linalg import norm, pinv

random.seed(0)


class Layer:
    """
    Description
    -------
    定义各层的结构，包括层号、激活函数、神经元数量、权重和偏置矩阵
    """
    def __init__(self, w, b, neural_number, transfer_function, layer_index):
        self.layer_index = layer_index
        self.transfer_function = transfer_function
        self.neural_number = neural_number
        self.w = w
        self.b = b


class NetStruct:
    """
    Description
    -------
    定义神经网络结构
    """
    def __init__(self, ni, nh, no, active_fun_list):
        """
        Description
        -------
        构造神经网络
        
        Parameters
        -------
        number of input, hidden, and output nodes
        ni : int
            输入层节点.
        nh : int or list
            隐藏层节点.
        no : int
            输出层节点.
        active_fun_list : list
            隐藏层激活函数类型.

        Returns
        -------
        """
        # ==> 1
        self.neurals = []  # 各层的神经元数目
        self.neurals.append(ni)
        if isinstance(nh, list):
            self.neurals.extend(nh)
        else:
            self.neurals.append(nh)
        self.neurals.append(no)

        # ==> 2
        if len(self.neurals) - 2 == len(active_fun_list):
            active_fun_list.append('line')
        self.active_fun_list = active_fun_list

        # ==> 3
        self.layers = []  # 所有的层
        layer_struct = []
        for i in range(0, len(self.neurals)):
            if i == 0:
                self.layers.append(Layer([], [], self.neurals[i], 'line', i))
                continue
            f = self.neurals[i - 1]
            s = self.neurals[i]
            self.layers.append(
                Layer(np.random.randn(s, f), np.random.randn(s, 1),
                      self.neurals[i], self.active_fun_list[i - 1], i))
            layer_struct.append([f, s])
        print(f"Network struct : {layer_struct}")
        print(f"Layer active funciation : {active_fun_list}")


class NeuralNetwork:
    """
    Description
    -------
    多层反向神经网络
    """
    def __init__(self, net_struct, mu=1e-3, beta=10, iteration=100, tol=0.1):
        self.net_struct = net_struct
        self.layer_num = len(net_struct.layers)
        self.mu = mu
        self.beta = beta
        self.iteration = iteration
        self.tol = tol

    def train(self, x, y, method='lm'):
        """
        训练
        """
        self.net_struct.x = x.T
        self.net_struct.y = y.reshape(1, -1)
        if method == 'lm':
            self.lm()

    def predict(self, x):
        """
        预测
        """
        self.net_struct.x = x.T
        self.forward()
        layer_num = len(self.net_struct.layers)
        predict = self.net_struct.layers[layer_num - 1].output_val
        return predict[0, :]

    def actFun(self, z, active_type='sigm'):
        """
        激活函数
        """
        # activ_type: 激活函数类型有 sigm、tanh、radb、line
        if active_type == 'sigm':
            f = 1.0 / (1.0 + np.exp(-z))
        elif active_type == 'tanh':
            f = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        elif active_type == 'radb':
            f = np.exp(-z * z)
        elif active_type == 'line':
            f = z
        return f

    def actFunGrad(self, z, active_type='sigm'):
        """
        激活函数的变化（派生）率
        """
        # active_type: 激活函数类型有 sigm、tanh、radb、line
        y = self.actFun(z, active_type)
        if active_type == 'sigm':
            grad = y * (1.0 - y)
        elif active_type == 'tanh':
            grad = 1.0 - y * y
        elif active_type == 'radb':
            grad = -2.0 * y * y
        elif active_type == 'line':
            m = y.shape[0]
            n = y.shape[1]
            grad = np.ones((m, n))
        return grad

    def forward(self):
        """
        前向
        """
        # layer_num = len(self.net_struct.layers)
        for i in range(0, self.layer_num):
            if i == 0:
                curr_layer = self.net_struct.layers[i]
                curr_layer.input_val = self.net_struct.x
                curr_layer.output_val = self.net_struct.x
                continue
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            curr_layer.input_val = np.dot(
                curr_layer.w, before_layer.output_val) + curr_layer.b
            curr_layer.output_val = self.actFun(
                curr_layer.input_val, self.net_struct.active_fun_list[i - 1])

    def backward(self):
        """
        反向
        """
        layer_num = len(self.net_struct.layers)
        last_layer = self.net_struct.layers[layer_num - 1]
        last_layer.error = -self.actFunGrad(
            last_layer.input_val,
            self.net_struct.active_fun_list[layer_num - 2])
        layer_index = list(range(1, layer_num - 1))
        layer_index.reverse()
        for i in layer_index:
            curr_layer = self.net_struct.layers[i]
            curr_layer.error = np.dot(
                last_layer.w.transpose(), last_layer.error) * self.actFunGrad(
                    curr_layer.input_val,
                    self.net_struct.active_fun_list[i - 1])
            last_layer = curr_layer

    def parDeriv(self):
        """
        标准梯度（求导）
        """
        layer_num = len(self.net_struct.layers)
        for i in range(1, layer_num):
            befor_layer = self.net_struct.layers[i - 1]
            befor_input_val = befor_layer.output_val.transpose()
            curr_layer = self.net_struct.layers[i]
            curr_error = curr_layer.error
            curr_error = curr_error.reshape(-1, 1, order='F')
            # curr_error = curr_error.reshape(curr_error.shape[0] *
            #                                 curr_error.shape[1],
            #                                 1,
            #                                 order='F')
            row = curr_error.shape[0]
            col = befor_input_val.shape[1]
            a = np.zeros((row, col))
            num = befor_input_val.shape[0]
            neural_number = curr_layer.neural_number
            for i in range(0, num):
                a[neural_number * i:neural_number * i +
                  neural_number, :] = np.repeat([befor_input_val[i, :]],
                                                neural_number,
                                                axis=0)
            tmp_w_par_deriv = curr_error * a
            curr_layer.w_par_deriv = np.zeros(
                (num, befor_layer.neural_number * curr_layer.neural_number))
            for i in range(0, num):
                tmp = tmp_w_par_deriv[neural_number * i:neural_number * i +
                                      neural_number, :]
                tmp = tmp.reshape(tmp.shape[0] * tmp.shape[1], order='C')
                curr_layer.w_par_deriv[i, :] = tmp
                curr_layer.b_par_deriv = curr_layer.error.transpose()

    def jacobian(self):
        """
        雅可比行列式
        """
        layers = self.net_struct.neurals
        row = self.net_struct.x.shape[1]
        col = 0
        for i in range(0, len(layers) - 1):
            col = col + layers[i] * layers[i + 1] + layers[i + 1]
        j = np.zeros((row, col))
        layer_num = len(self.net_struct.layers)
        index = 0
        for i in range(1, layer_num):
            curr_layer = self.net_struct.layers[i]
            w_col = curr_layer.w_par_deriv.shape[1]
            b_col = curr_layer.b_par_deriv.shape[1]
            j[:, index:index + w_col] = curr_layer.w_par_deriv
            index = index + w_col
            j[:, index:index + b_col] = curr_layer.b_par_deriv
            index = index + b_col
        return j

    def gradCheck(self):
        """
        梯度检查
        """
        W1 = self.net_struct.layers[1].w
        b1 = self.net_struct.layers[1].b
        n = self.net_struct.layers[1].neural_number
        W2 = self.net_struct.layers[2].w
        b2 = self.net_struct.layers[2].b
        x = self.net_struct.x
        p = []
        p.extend(W1.reshape(1, W1.shape[0] * W1.shape[1], order='C')[0])
        p.extend(b1.reshape(1, b1.shape[0] * b1.shape[1], order='C')[0])
        p.extend(W2.reshape(1, W2.shape[0] * W2.shape[1], order='C')[0])
        p.extend(b2.reshape(1, b2.shape[0] * b2.shape[1], order='C')[0])
        old_p = p
        jac = []
        for i in range(0, x.shape[1]):
            xi = np.array([x[:, i]])
            xi = xi.transpose()
            ji = []
            for j in range(0, len(p)):
                W1 = np.array(p[0:2 * n]).reshape(n, 2, order='C')
                b1 = np.array(p[2 * n:2 * n + n]).reshape(n, 1, order='C')
                W2 = np.array(p[3 * n:4 * n]).reshape(1, n, order='C')
                b2 = np.array(p[4 * n:4 * n + 1]).reshape(1, 1, order='C')

                z2 = np.dot(W1, xi) + b1  # W1.dot(xi) + b1
                a2 = self.actFun(z2)
                z3 = np.dot(W2, a2) + b2  # W2.dot(a2) + b2
                h1 = self.actFun(z3)
                p[j] = p[j] + 0.00001
                W1 = np.array(p[0:2 * n]).reshape(n, 2, order='C')
                b1 = np.array(p[2 * n:2 * n + n]).reshape(n, 1, order='C')
                W2 = np.array(p[3 * n:4 * n]).reshape(1, n, order='C')
                b2 = np.array(p[4 * n:4 * n + 1]).reshape(1, 1, order='C')

                z2 = np.dot(W1, xi) + b1  # W1.dot(xi) + b1
                a2 = self.actFun(z2)
                z3 = np.dot(W2, a2) + b2  # W2.dot(a2) + b2
                h = self.actFun(z3)
                g = (h[0][0] - h1[0][0]) / 0.00001
                ji.append(g)
            jac.append(ji)
            p = old_p
        return jac

    def jjje(self):
        """
        计算jj与je
        """
        layer_num = len(self.net_struct.layers)
        e = self.net_struct.y - self.net_struct.layers[layer_num -
                                                       1].output_val
        e = e.transpose()
        j = self.jacobian()
        # check gradient
        # j1 = -np.array(self.gradCheck())
        # jk = j.reshape(1,j.shape[0]*j.shape[1])
        # jk1 = j1.reshape(1,j1.shape[0]*j1.shape[1])
        # plt.plot(jk[0])
        # plt.plot(jk1[0],'.')
        # plt.show()
        jj = np.dot(j.transpose(), j)  # j.transpose().dot(j)
        je = np.dot(-j.transpose(), e)  # -j.transpose().dot(e)
        return [jj, je]

    def lm(self):
        """
        Levenberg-Marquardt训练算法
        """
        mu = self.mu
        beta = self.beta
        iteration = self.iteration
        tol = self.tol
        y = self.net_struct.y
        layer_num = len(self.net_struct.layers)
        self.forward()
        pred = self.net_struct.layers[layer_num - 1].output_val
        pref = self.perfermance(y, pred)
        for i in range(0, iteration):
            if i % 100 == 0:
                print(f'iter : {i} error {pref}')
            # 1) 第一步:
            if pref < tol:
                break
            # 2) 第二步:
            self.backward()
            self.parDeriv()
            jj, je = self.jjje()
            while 1:
                # 3) 第三步:
                A = jj + mu * np.diag(np.ones(jj.shape[0]))
                delta_w_b = np.dot(pinv(A), je)  # pinv(A).dot(je)
                # 4) 第四步:
                old_net_struct = copy.deepcopy(self.net_struct)
                self.updataNetStruct(delta_w_b)
                self.forward()
                pred1 = self.net_struct.layers[layer_num - 1].output_val
                pref1 = self.perfermance(y, pred1)
                if pref1 < pref:
                    mu = mu / beta
                    pref = pref1
                    break
                mu = mu * beta
                self.net_struct = copy.deepcopy(old_net_struct)
        # pred_1 = self.net_struct.layers[layer_num - 1].output_val

    def updataNetStruct(self, delta_w_b):
        """
        更新网络权重及阈值
        """
        layer_num = len(self.net_struct.layers)
        index = 0
        for i in range(1, layer_num):
            before_layer = self.net_struct.layers[i - 1]
            curr_layer = self.net_struct.layers[i]
            w_num = before_layer.neural_number * curr_layer.neural_number
            b_num = curr_layer.neural_number
            w = delta_w_b[index:index + w_num]
            w = w.reshape(curr_layer.neural_number,
                          before_layer.neural_number,
                          order='C')
            index = index + w_num
            b = delta_w_b[index:index + b_num]
            index = index + b_num
            curr_layer.w += w
            curr_layer.b += b

    def perfermance(self, y, pred):
        """
        性能函数
        """
        error = y - pred
        return norm(error) / len(y)


# 以下函数为测试样例
def plotSamples(n=40):
    x = np.array([np.linspace(0, 3, n)])
    x = x.repeat(n, axis=0)
    y = x.transpose()
    z = np.zeros((n, n))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i, j] = sampleFun(x[i, j], y[i, j])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='autumn', cstride=2, rstride=2)
    ax.set_xlabel("X-Label")
    ax.set_ylabel("Y-Label")
    ax.set_zlabel("Z-Label")
    plt.show()
    plt.close()
    return None


def sinSamples(n):
    x = np.array([np.linspace(-0.5, 0.5, n)])
    y = x + 0.2
    z = np.sin(x * y)
    X = np.vstack((x, y))
    return X.transpose(), z[0, :]


def peaksSamples(n):
    x = np.array([np.linspace(-3, 3, n)])
    x = x.repeat(n, axis=0)
    y = x.transpose()
    z = np.zeros((n, n))
    for i in range(0, x.shape[0]):
        for j in range(0, x.shape[1]):
            z[i, j] = sampleFun(x[i, j], y[i, j])
    X = np.zeros((n * n, 2))
    X[:, 0] = x.flatten()
    X[:, 1] = y.flatten()
    return X, z.flatten()


def sampleFun(x, y):
    z = 3 * pow((1 - x), 2) * exp(-(pow(x, 2)) - pow(
        (y + 1), 2)) - 10 * (x / 5 - pow(x, 3) - pow(y, 5)) * exp(
            -pow(x, 2) - pow(y, 2)) - 1 / 3 * exp(-pow((x + 1), 2) - pow(y, 2))
    return z


def load_iris(show=False):
    iris = pd.read_csv('data/iris.csv').sample(frac=1).reset_index(drop=True)
    # Create numeric classes for species (0,1,2)
    iris.loc[iris['species'] == 'virginica', 'species_id'] = 0
    iris.loc[iris['species'] == 'versicolor', 'species_id'] = 1
    iris.loc[iris['species'] == 'setosa', 'species_id'] = 2
    # iris = iris[iris['species_id']!=2]
    raw_feature = iris.iloc[0:, 0:4].values.astype(float)
    ele = iris['species_id'].values
    x_0, x_1 = raw_feature[:100, :], raw_feature[101:, :]
    y_0, y_1 = ele[:100], ele[101:]
    # Make a scatter plot
    if show:
        # Create Input and Output columns
        x = iris[['petal_length', 'petal_width']].values.T
        y = iris[['species_id']].values.astype('uint8').T
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x[0, :], x[1, :], c=y[0, :], s=40, cmap=plt.cm.Spectral)
        plt.title(
            "IRIS DATA | Blue - Setosa, Yellow - Versicolor, Red - Virginica ")
        plt.xlabel('Petal Length')
        plt.ylabel('Petal Width')
        plt.tight_layout()
        plt.show()
        plt.close()
    return x_0, y_0, x_1, y_1


if __name__ == '__main__':
    train_x, train_y = peaksSamples(20)  # 产生训练数据
    test_x, test_y = peaksSamples(40)  # 产生测试数据

    # 第二个测试数据
    # train_x, train_y = sinSamples(20)
    # test_x, test_y = sinSamples(40)

    # Iris
    # train_x, train_y, test_x, test_y = load_iris()
    print(train_x.shape)
    print(train_y.shape)

    # 设置各隐层的激活函数类型，可以设置为sigm, radb, tanh, line类型，如果不显式的设置最后一层为line
    active_fun_list = ['sigm', 'sigm', 'sigm']
    ns = NetStruct(train_x.shape[1], [10, 10, 10], 1, active_fun_list)
    nn = NeuralNetwork(ns)

    nn.train(train_x, train_y)

    pred_y = nn.predict(test_x)
    # class_y = np.ones_like(test_y)
    # class_y[pred_y<0.5] = 0
    # class_y[pred_y>1.5] = 2

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(test_y)  # 画出真实值 real data
    ax.plot(pred_y, 'r.')  # 画出预测值 predict data
    plt.legend(('real data', 'predict data'))
    plt.tight_layout()
    plt.show()
    plt.close()
