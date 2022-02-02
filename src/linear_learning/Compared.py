# -*- coding: UTF-8 -*-

import pandas as pd
import csv
import matplotlib.pyplot as plt


def Compared():
    # 利用方程进行拟合 对比 并存储数据到eth3.csv
    pd_data = pd.read_csv('eth2.csv')
    sam = []
    a = ['priceUSD', 'activeAddresses', 'adjustedVolume', 'paymentCount', 'exchangeVolume', 'priceBTC']
    dic = {}
    for i in a:
        y = pd_data.loc[:, i]
        dic[i] = list(y)  # 归一化
    print(dic)
    for i in range(len(dic['priceUSD'])):
        x = 0.00406340113944 + float(dic['activeAddresses'][i]) * 0.49474868663194016 + float(
            dic['adjustedVolume'][i]) * 0.42552157541384 + float(dic['paymentCount'][i]) * 0.12214416604623446 + float(
            dic['exchangeVolume'][i]) * (-0.23814049518276936) + float(dic['priceBTC'][i]) * 0.21567132432245326
        sam.append(x)

    with open('eth3.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['priceUSD', 'Predictive value'])
        for i in range(len(sam)):
            writer.writerow([dic['priceUSD'][i], sam[i]])
    print('完毕')
    pd_data = pd.read_csv('eth3.csv')
    pd_data.plot()
    plt.show()


if __name__ == '__main__':
    Compared()
