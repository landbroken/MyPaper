#!/usr/bin/python3.9
# -*- coding: utf-8 -*-
#
# Copyright (C) 2021 LinYulong. All Rights Reserved 
#
# @Time    : 2021/10/27
# @Author  : LinYulong
# @Descriptor:
# 参考
# https://www.cnblogs.com/listenfwind/p/11310924.html
# https://zhuanlan.zhihu.com/p/266880465?ivk_sa=1024320u

import pandas
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn import tree


def get_watermelon_data() -> pandas.DataFrame:
    titles_columns = \
        ['color', 'root', 'sound', 'picture', 'umbilicus', 'touch', 'good']
    #   ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']
    df = pd.DataFrame([
        ['green', '蜷缩', '浊响', 'clear', '凹陷', 'hard', 'yes'],  # 1
        ['black', '蜷缩', '沉闷', 'clear', '凹陷', 'hard', 'yes'],
        ['black', '蜷缩', '浊响', 'clear', '凹陷', 'hard', 'yes'],
        ['green', '蜷缩', '沉闷', 'clear', '凹陷', 'hard', 'yes'],
        ['white', '蜷缩', '浊响', 'clear', '凹陷', 'hard', 'yes'],  # 5

        ['green', '稍蜷', '浊响', 'clear', '稍凹', 'soft', 'yes'],  # 6
        ['black', '稍蜷', '浊响', 'little_fuzz', '稍凹', 'soft', 'yes'],
        ['black', '稍蜷', '浊响', 'clear', '稍凹', 'hard', 'yes'],
        ['black', '稍蜷', '沉闷', 'little_fuzz', '稍凹', 'hard', 'no'],
        ['green', '硬挺', '清脆', 'clear', '平坦', 'soft', 'no'],  # 10

        ['white', '硬挺', '清脆', 'fuzz', '平坦', 'hard', 'no'],
        ['white', '蜷缩', '浊响', 'fuzz', '平坦', 'soft', 'no'],
        ['green', '稍蜷', '浊响', 'little_fuzz', '凹陷', 'hard', 'no'],
        ['white', '稍蜷', '沉闷', 'little_fuzz', '凹陷', 'hard', 'no'],
        ['black', '稍蜷', '浊响', 'clear', '稍凹', 'soft', 'no'],  # 15

        ['white', '蜷缩', '浊响', 'fuzz', '平坦', 'hard', 'no'],
        ['green', '蜷缩', '沉闷', 'little_fuzz', '稍凹', 'hard', 'no'],  # 17
    ],
        columns=titles_columns)
    return df


def test_watermelon_decision_tree():
    data = get_watermelon_data()
    vec = DictVectorizer(sparse=False)
    # sparse=False意思yes不产生稀疏矩阵
    vec = DictVectorizer(sparse=False)
    # 先用 pandas 对每行生成字典，然后进行向量化
    feature = data[['color', 'root', 'sound', 'picture', 'umbilicus', 'touch']]
    dict_feature = feature.to_dict(orient='records')
    x_train = vec.fit_transform(dict_feature)
    # 打印各个变量
    print('show feature\n', feature)
    print('show vector name\n', vec.get_feature_names_out())
    print('show vector\n', x_train)

    vec2 = DictVectorizer(sparse=False)
    result = data[['good']]
    y_train = vec2.fit_transform(result.to_dict(orient='records'))
    print('show result vector name\n', vec2.get_feature_names_out())
    print('show result vector\n', y_train)

    # 决策树训练
    # 划分成训练集，交叉集，验证集，不过这里我们数据量不够大，没必要
    # train_x, test_x, train_y, test_y = train_test_split(X_train, Y_train, test_size = 0.3)
    # 训练决策树
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(x_train, y_train)

    # 保存成 dot 文件，后面可以用 dot out.dot -T pdf -o out.pdf 转换成图片
    with open("out.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f, feature_names=vec.get_feature_names_out())


if __name__ == "__main__":
    test_watermelon_decision_tree()
