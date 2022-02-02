# -*- coding: UTF-8 -*-

import pandas as pd
import csv
from sklearn import preprocessing


def Normalization():
    # 对数据进行归一化处理 并存储到eth2.csv
    pd_data = pd.read_csv('eth.csv')
    sam = []
    a = ['priceUSD', 'activeAddresses', 'adjustedVolume', 'paymentCount', 'exchangeVolume', 'priceBTC']
    for i in a:
        y = pd_data.loc[:, i]
        ys = list(preprocessing.scale(y))  # 归一化
        sam.append(ys)

    print(len(sam))
    with open('eth2.csv', 'w') as file:
        writer = csv.writer(file)
        for i in range(len(sam[0])):
            writer.writerow([sam[0][i], sam[1][i], sam[2][i], sam[3][i], sam[4][i], sam[5][i]])
    print('完毕')


if __name__ == '__main__':
    Normalization()
