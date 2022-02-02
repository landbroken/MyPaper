# -*- coding: UTF-8 -*-

import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl  # 显示中文
from sklearn.model_selection import train_test_split  # 这里是引用了交叉验证
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn import metrics
import numpy as np

from sklearn import preprocessing

pd_data = pd.read_csv('eth2.csv')  # 归一化数表


# pd_data=pd.read_csv('eth.csv')#原始数表


def display_all():
    # 画出所有因素图
    pd_data.plot()
    plt.show()


# def display_lr():
#     # 画出单因素拟合情况
#     print('pd_data.head(10)=\n{}'.format(pd_data.head(10)))
#     # mpl.rcParams['font.sans-serif'] = ['SimHei']  #配置显示中文，否则乱码
#     mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号，如果是plt画图，则将mlp换成plt
#     sns.pairplot(pd_data, x_vars=['activeAddresses', 'adjustedVolume', 'paymentCount', 'exchangeVolume', 'priceBTC'],
#                  y_vars='priceUSD', kind="reg", size=5, aspect=0.7)
#     plt.show()


def build_lr():
    X = pd_data.loc[:, ('activeAddresses', 'adjustedVolume', 'paymentCount', 'exchangeVolume', 'priceBTC')]
    y = pd_data.loc[:, 'priceUSD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=532)  # 选择20%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n,  y_test.shape={}'.format(X_train.shape,
                                                                                              y_train.shape,
                                                                                              X_test.shape,
                                                                                              y_test.shape))
    linreg = LinearRegression()
    # 训练
    model = linreg.fit(X_train, y_train)
    print('模型参数:')
    print(model)
    # 训练后模型截距
    print('模型截距:')
    print(linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    print('参数权重:')
    print(linreg.coef_)

    y_pred = linreg.predict(X_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))  # 测试级的数量
    # calculate RMSE
    print("RMSE by hand:", sum_erro)
    # 做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()


if __name__ == '__main__':
    build_lr()
