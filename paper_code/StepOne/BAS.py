import numpy as np
from math import sqrt
import ReadData


def normalize(x):  # 单位化向量
    return x / sqrt(sum(e ** 2 for e in x))


def sign(a):  # 符号函数
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0


def loss(input):  # 计算损失函数，越小越准确
    goal = [8.63, 5.11]
    return sqrt(pow(input[0] - goal[0], 2) + pow(input[1] - goal[1], 2))


def run(dataSet,iterNum):
    # 应用不同问题时主要改这里的参数和函数
    eta = 0.8  # 步长调整比例
    step = 5  # 初始搜索步长
    d0 = 6  # 触须间距
    k = 1  # 变量维数
    # x = np.random.rand(k)  # 随机生成天牛质心坐标
    x = 20
    xl = x  # 左触须坐标
    xr = x  # 右触须坐标
    ex = 1e-15  # 精度
    res=[]
    ###
    iter = 0
    while iter < iterNum:  # 开始迭代
    # while loss(x) > ex:  # 开始迭代
    #     dir = normalize(np.random.rand(k))
    #     xl = x + d0 * dir / 2
    #     xr = x - d0 * dir / 2
        xl = int(x - d0 / 2)
        xr = int(x + d0 / 2)
        # x = x - step * dir * sign(loss(xl) - loss(xr))
        if xl<3:
            x = x + step
            res.append(x)
            continue
        x = int(x - step * sign(ReadData.run(dataSet,xl) - ReadData.run(dataSet,xr)))
        res.append(x)
        step *= eta
        print("========结果========")
        print(res)
        print("========迭代次数========")
        print(iter)
        iter+=1