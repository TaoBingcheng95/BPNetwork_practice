"""
https://www.cnblogs.com/hhh5460/p/4304628.html
http://www.hankcs.com/ml/back-propagation-neural-network.html
http://iamtrask.github.io/2015/07/12/basic-python-network/
http://neuralnetworksanddeeplearning.com/chap1.html
https://wiki.jikexueyuan.com/project/neural-networks-and-deep-learning-zh-cn/chapter2.html
http://www.hankcs.com/wp-content/uploads/2015/11/The%20back-propagation%20algorithm.pdf
"""

import math
import random
import numpy as np

random.seed(0)


def sigmoid(z):
    """
    Description
    -------
    sigmoid 函数
    1/(1+e^-x)
    """
    return 1.0 / (1.0 + math.exp(-z))


def dsigmoid(z):
    """
    Description
    -------
    sigmoid 函数的导数
    z * (1 - z)
    """
    return z * (1 - z)


def tanh(z):
    """
    Description
    -------
    tanh 函数
    tanh(z)
    """
    return np.tanh(z)  # math.tanh(z)


def dtanh(z):
    """
    Description
    -------
    tanh 函数的导数
    1.0 - z**2
    """
    return 1.0 - z**2


class NN:
    """
    Description
    -------
    三层反向传播神经网络
    """
    def __init__(self, ni, nh, no):
        """
        Description
        -------
        构造神经网络
        
        Parameters
        -------
        number of input, hidden, and output nodes
        ni : int
            输入单元数量.
        nh : int
            隐藏单元数量.
        no : int
            输出单元数量.

        Returns
        -------

        """
        self.ni = ni + 1  # 增加一个偏差节点
        self.nh = nh
        self.no = no

        # 激活神经网络的所有节点（向量）
        self.ai = np.ones((1, self.ni))
        self.ah = np.ones((1, self.nh))
        self.ao = np.ones(self.no)
        # 建立权重（随机矩阵）
        self.wi = 2 * np.random.random((self.ni, self.nh)) - 1
        self.wo = 2 * np.random.random((self.nh, self.no)) - 1
        # 建立动量因子（矩阵）
        self.ci = np.full((self.ni, self.nh), 0.0)
        self.co = np.full((self.nh, self.no), 0.0)
        ls = [[1, self.ni], self.wi.shape, self.wo.shape]
        print(f"Net layer strcut : {ls}")

    def train(self, xx, yy, iterations=1000, N=0.5, M=0.1):
        """
        Description
        -------
        训练过程
        
        Parameters
        -------
        xx : numpy.array
            输入数据
        yy : numpy.array
            输出数据
        iterations ：int, optional. 
            迭代次数, The default is 1000.
        N : float, optional.
            学习速率(learning rate), The default is 0.5.
        M : float, optional.
            动量因子(momentum factor), The default is 0.1.

        Returns
        -------
        
        """
        input_num = yy.shape[0]

        for i in range(iterations):
            error = 0.0
            for ii in range(input_num):
                inputs = xx[ii, :]
                targets = yy[ii, :]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)

            if i % 100 == 0:
                print(f'Epoch {i} Combined error : {error:.5f}')

    def update(self, inputs):
        """
        Description
        -------
        前向传播更新
        
        Parameters
        -------
        inputs : numpy.array
            输入数据

        Returns
        -------
        ao : numpy.array
            输出节点

        """
        if len(inputs) != self.ni - 1:
            raise ValueError('与输入层节点数不符！')  # incorrect number of inputs

        # 激活输入层
        # for i in range(self.ni - 1):
        #     self.ai[0, i] = inputs[i]  # sigmoid(inputs[i])
        self.ai[0, :-1] = inputs

        # 激活隐藏层
        # for j in range(self.nh):
        #     sum_h = np.dot(self.ai, self.wi[:, j])
        #     self.ah[0, j] = tanh(sum_h)
        self.ah = tanh(np.dot(self.ai, self.wi))

        # 激活输出层
        # for k in range(self.no):
        #     sum_o = np.dot(self.ah, self.wo[:, k])
        #     self.ao[k] = tanh(sum_o)
        self.ao = tanh(np.dot(self.ah, self.wo))

        return self.ao

    def backPropagate(self, targets, N, M):
        """
        Description
        -------
        后向传播算法
        http://www.youtube.com/watch?v=aVId8KMsdUU&feature=BFa&list=LLldMCkmXl4j9_v0HeKdNcRA

        Parameters
        -------
        targets : numpy.array or list
            输入的实例
        N : float
            本次学习率
        M : float
            上次学习率

        Returns
        -------
        error: float
            最终的误差平方和的一半
        
        """

        if len(targets) != self.no:
            raise ValueError('与输出层节点数不符！')  # incorrect number of outputs

        # 计算输出层误差 output_deltas
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        # output_deltas = np.zeros(self.no)
        # for k in range(self.no):
        #     error = targets[k] - self.ao[k]
        #     output_deltas[k] = error * dtanh(self.ao[k])
        output_deltas = (targets - self.ao) * dtanh(self.ao)

        # 计算隐藏层误差 hidden_deltas
        # hidden_deltas = np.zeros(self.nh)
        # for j in range(self.nh):
        #     error = np.dot(output_deltas, self.wo[j, :])
        #     hidden_deltas[j] = error * dtanh(self.ah[0, j])
        hidden_deltas = np.dot(output_deltas, self.wo.T) * dtanh(self.ah)

        # 更新输出层权重
        # for j in range(self.nh):
        #     change = np.dot(output_deltas, self.ah[:, j])
        #     self.wo[j, :] += N * change + M * self.co[j, :]
        #     self.co[j, :] = change
        #     # print(N*change, M*self.co[j, k])
        change = np.dot(output_deltas, self.ah)
        self.wo += N * change.T + M * self.co
        self.co = change.T

        # 更新输入层权重
        # for i in range(self.ni):
        #     change = hidden_deltas[0, :] * self.ai[0, i]
        #     self.wi[i, :] += N * change + M * self.ci[i, :]
        #     self.ci[i, :] = change
        #     # for j in range(self.nh):
        #     #     print('activation', self.ai[0, i], 'synapse', i, j, 'change', change)
        change = np.dot(self.ai.T, hidden_deltas)
        self.wi += N * change + M * self.ci
        self.ci = change

        # 计算误差和
        error = 0.5 * np.sum(np.square(targets - self.ao))
        # error = 0.0
        # for k in range(len(targets)):
        #     error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def test(self, xx, yy):
        """
        Description
        -------
        测试模型效果

        Parameters
        -------
        xx : numpy.array or list
            测试数据中的输入值
        yy : numpy.array or list
            测试数据中的输出值

        Returns
        -------

        """
        for inputs, targets in zip(xx, yy):
            predict = self.update(inputs)
            print(f"Inputs: {inputs} --> {predict} --> Target: {targets}")

    def weights(self):
        """
        打印权值矩阵
        """
        print('Input weights : \n{}\nInput bias : \n{}'.format(
            self.wi[:-1, :], self.wi[-1, :]))
        print('Output weights : \n{}\nOutput bias : \n{}'.format(
            self.wo[:-1, :], self.wo[-1, :]))
        return None


if __name__ == '__main__':
    # 逻辑异或（XOR）
    x = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([[0], [1], [1], [0]])
    print(x.shape, y.shape)

    # 创建一个神经网络：输入层有两个节点、隐藏层有两个节点、输出层有一个节点
    n = NN(2, 2, 1)
    n.train(x, y)
    # 训练好的权重
    # n.weights()
    n.test(x, y)
