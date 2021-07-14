# https://www.cnblogs.com/hhh5460/p/5324748.html

import math
import numpy as np


def sigmoid(z):
    """
    Description
    -------
    sigmoid 函数
    1/(1+e^-x)
    """
    return 1.0 / (1.0 + np.exp(-z))


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
    return np.tanh(z)


def dtanh(z):
    """
    Description
    -------
    tanh 函数的导数
    1.0 - z**2
    """
    return 1.0 - z**2


def demo_1():
    ''' 
    神经网络
    固定三层，两类
    # 只适合 0, 1 两类。若不是，要先转化
    '''
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([0, 1, 1, 0])
    # print(X.shape, y.shape)
    y = y.reshape(-1, 1)  # 此处reshape是为了便于算法简洁实现

    wi = 2 * np.random.random((3, 5)) - 1  # wi = 2 * np.random.randn(3, 5) - 1
    wh = 2 * np.random.random((5, 1)) - 1

    print("Layrt Struct : [[1, 3], [3,5], [5, 1]]")

    li = X
    for j in range(10000):
        # lh = 1 / (1 + np.exp(-(np.dot(li, wi))))  # 4x3 3x5 -> 4x5
        # lo = 1 / (1 + np.exp(-(np.dot(lh, wh))))  # 4x5 5x1 -> 4x1
        lh = sigmoid(np.dot(li, wi))
        lo = sigmoid(np.dot(lh, wh))

        # lo_delta = (y - lo) * (lo * (1 - lo))  # 4x1
        # lh_delta = np.dot(lo_delta, wh.T) * (lh * (1 - lh))  # 4x5
        lo_delta = (y - lo) * dsigmoid(lo)  # 4x1
        lh_delta = np.dot(lo_delta, wh.T) * dsigmoid(lh)  # 4x5

        wh += np.dot(lh.T, lo_delta)  # 5x1
        wi += np.dot(li.T, lh_delta)  # 3x5

        if j % 500 == 0:
            error = np.sum(lo_delta)
            print(f'Epoch {j} Combined error : {error:.5f}')
            if abs(error) < 1e-4:
                break

    print('真实数据：', y)
    print('训练结果：', lo)
    return None


def demo_2():
    '''
    神经网络
    层数可变，两类
    # 只适合 0, 1 两类。若不是，要先转化
    '''
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    y = np.array([0, 1, 1, 0])
    # print(X.shape, y.shape)
    y = y.reshape(-1, 1)  # 此处reshape是为了便于算法简洁实现

    neurals = [3, 15, 1]
    # w = [np.random.randn(i, j)
    #      for i, j in zip(neurals[:-1], neurals[1:])] + [None]
    w = [np.random.random((i, j))
         for i, j in zip(neurals[:-1], neurals[1:])] + [None]
    l = [None] * len(neurals)
    l_delta = [None] * len(neurals)

    ls = [item.shape for item in w[:-1]]
    print(f"Layrt Struct : {ls}")

    l[0] = X
    for j in range(1000):

        for i in range(1, len(neurals)):
            # l[i] = 1 / (1 + np.exp(-(np.dot(l[i - 1], w[i - 1]))))
            l[i] = sigmoid(np.dot(l[i - 1], w[i - 1]))

        # l_delta[-1] = (y - l[-1]) * (l[-1] * (1 - l[-1]))
        l_delta[-1] = (y - l[-1]) * dsigmoid(l[-1])
        for i in range(len(neurals) - 2, 0, -1):
            # l_delta[i] = np.dot(l_delta[i + 1], w[i].T) * (l[i] * (1 - l[i]))
            l_delta[i] = np.dot(l_delta[i + 1], w[i].T) * dsigmoid(l[i])

        for i in range(len(neurals) - 2, -1, -1):
            w[i] += np.dot(l[i].T, l_delta[i + 1])

        if j % 100 == 0:
            error = np.sum(l_delta[-1])
            print(f'Epoch {j} Combined error : {error:.5f}')
            if abs(error) < 1e-4:
                break

    print('真实数据：', y)
    print('训练结果：', l[-1])
    return None


def demo_3():
    '''
    神经网络
    固定三层，多类
    '''
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    #y = np.array([0,1,1,0]) # 可以两类
    y = np.array([0, 1, 2, 3])  # 可以多类

    wi = 2 * np.random.random((3, 5)) - 1  # wi = np.random.randn(3, 5)
    wh = 2 * np.random.random((5, 4)) - 1
    bh = 2 * np.random.random((1, 5)) - 1
    bo = 2 * np.random.random((1, 4)) - 1

    ls = [(1, 3)] + [item.shape for item in [wi, wh]]
    print(f"Layrt Struct : {ls}")

    epsilon = 0.01  # 学习速率
    lamda = 0.01  # 正则化强度

    li = X
    for j in range(1000):
        lh = np.tanh(np.dot(li, wi) + bh)  # tanh 函数  4*5
        lo = np.exp(np.dot(lh, wh) + bo)  # 4*4
        probs = lo / np.sum(lo, axis=1, keepdims=True)

        # 后向传播
        lo_delta = np.copy(probs)
        lo_delta[range(X.shape[0]), y] -= 1
        lh_delta = np.dot(lo_delta, wh.T) * (1 - np.power(lh, 2))  # dtanh

        # 更新权值、偏置
        wh -= epsilon * (np.dot(lh.T, lo_delta) + lamda * wh)
        wi -= epsilon * (np.dot(li.T, lh_delta) + lamda * wi)

        bo -= epsilon * np.sum(lo_delta, axis=0, keepdims=True)
        bh -= epsilon * np.sum(lh_delta, axis=0)

        if j % 100 == 0:
            error = np.sum(lo_delta)
            print(f'Epoch {j} Combined error : {error}')
            # if abs(error) < 1e-4:
            #     break
    print(probs.shape)
    print('训练结果：', np.argmax(probs, axis=1))
    return None


def demo_4():
    '''
    神经网络
    层数可变，多类
    '''
    X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    #y = np.array([0,1,1,0]) # 可以两类
    y = np.array([0, 1, 2, 3])  # 可以多类

    neurals = [3, 10, 8, 4]
    # w = [np.random.randn(i, j)
    #      for i, j in zip(neurals[:-1], neurals[1:])] + [None]
    # b = [None] + [np.random.randn(1, j) for j in neurals[1:]]
    w = [
        2 * np.random.random((i, j)) - 1
        for i, j in zip(neurals[:-1], neurals[1:])
    ] + [None]
    b = [None] + [2 * np.random.random((1, j)) - 1 for j in neurals[1:]]
    l = [None] * len(neurals)
    l_delta = [None] * len(neurals)

    ls = [item.shape for item in w[:-1]]
    print(f"Layrt Struct : {ls}")

    epsilon = 0.01  # 学习速率
    lamda = 0.01  # 正则化强度

    l[0] = X
    for j in range(1000):
        # 前向传播
        for i in range(1, len(neurals) - 1):
            l[i] = np.tanh(np.dot(l[i - 1], w[i - 1]) + b[i])  # tanh 函数
        l[-1] = np.exp(np.dot(l[-2], w[-2]) + b[-1])
        probs = l[-1] / np.sum(l[-1], axis=1, keepdims=True)

        # 后向传播
        l_delta[-1] = np.copy(probs)
        l_delta[-1][range(X.shape[0]), y] -= 1
        for i in range(len(neurals) - 2, 0, -1):
            l_delta[i] = np.dot(l_delta[i + 1], w[i].T) * (
                1 - np.power(l[i], 2))  # tanh 函数的导数

        # 更新权值、偏置
        b[-1] -= epsilon * np.sum(l_delta[-1], axis=0, keepdims=True)
        for i in range(len(neurals) - 2, -1, -1):
            w[i] -= epsilon * (np.dot(l[i].T, l_delta[i + 1]) + lamda * w[i])
            if i == 0: break
            b[i] -= epsilon * np.sum(l_delta[i], axis=0)

        # 打印损失
        if j % 100 == 0:
            loss = np.sum(-np.log(probs[range(X.shape[0]), y]))
            loss += lamda / 2 * np.sum(
                [np.sum(np.square(wi)) for wi in w[:-1]])  # 可选
            loss *= 1 / X.shape[0]  # 可选
            print(f'Epoch {j} loss : {loss}')

    print('训练结果：', np.argmax(probs, axis=1))
    return None


if __name__ == '__main__':

    demo_4()
