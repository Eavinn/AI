"""
欠拟合：增加输入特征项，增加网络参数，减少正则化力度
过拟合：数据清洗，增大训练集，采用正则化
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression


def linear_regression():
    """
    线性回归：本质是将损失函数（最小二乘法）降到最小
    正规方程-直接求解，复杂度高，适合小规模数据
    梯度下降-迭代求解，需要制定学习率（默认0.01），适合大规模数据
    岭回归-梯度下降的基础上修改SGD随机梯度下降法为SAG随机平均梯度法，同时L2正则化限制解决过拟合问题。
    """
    boston = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 正规方程
    # estimator = LinearRegression()
    # 梯度下降
    # estimator = SGDRegressor()
    # 岭回归
    estimator = Ridge()

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)
    print("系数：", estimator.coef_)
    print("偏置：", estimator.intercept_)
    print("预测结果为：", y_predict)

    # 均方误差进行模型评估
    print("均方误差为：", mean_squared_error(y_test, y_predict))


def logistic_regression():
    """
    逻辑回归:线性回归的输出是逻辑回归的输入，然后sigmoid映射[0,1]输出预测值，预测值大于阈值则为A类，否则为B类。实际上是分类算法
    损失函数使用的对数似然函数

    """
    column_name = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']
    data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        names=column_name)
    data.replace(to_replace="?", value=np.nan, inplace=True)
    data.dropna(inplace=True)

    x = data[column_name[1:-1]]
    y = data[column_name[-1]]
    print(y.unique())
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    estimator = LogisticRegression()
    estimator.fit(x_train, y_train)

    # 模型保存与加载
    joblib.dump(estimator, "data/LogisticRegression.pkl")
    estimator = joblib.load("data/LogisticRegression.pkl")

    y_predict = estimator.predict(x_test)
    print("系数：", estimator.coef_)
    print("偏置：", estimator.intercept_)
    print("预测结果为：", y_predict)

    print("预测的准确率:", estimator.score(x_test, y_test))

    # 1.分类评估报告
    # 精确率Precision：预测结果为正例样本中真实为正例的比例；
    # 召回率Recall：真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力）
    # F1-score：反映了模型的稳健型
    print(classification_report(y_test, y_predict, labels=[2, 4], target_names=['良性', '恶性']))

    # 2.AUC指标
    # 以TPR(所有真实类别为1的样本中，预测类别为1的比例)为x轴，以FPR(所有真实类别为0的样本中，预测类别为1的比例)为y轴的曲线即ROC曲线，ROC曲线围成的面积即AUC值
    # AUC的范围在[0.5, 1]之间，并且越接近1越好.AUC只能用来评价二分类,适合评价样本不平衡中的分类器性能
    # y_true:每个样本的真实类别，必须为0(反例),1(正例)标记
    y_test = np.where(y_test > 3, 1, 0)
    print("AUC指标：", roc_auc_score(y_test, y_predict))


logistic_regression()
# linear_regression()
