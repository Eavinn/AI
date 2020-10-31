"""
监督学习-有目标值，分为分类和回归; 无监督学习-无目标值,聚类
机器学习基本流程：获取数据-数据集划分-特征工程（转换器流程）-算法（预估器流程）-模型评估
"""

import pickle
import pandas as pd
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier


def knn():
    """
    KNN: 如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。
    距离即欧氏距离, 懒惰算法，对测试样本分类时的计算量大，内存开销大
    :return:
    """
    iris = load_iris()

    aa = pickle.dumps(iris)
    iris = pickle.loads(aa)
    print(type(iris))

    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    # 标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 实例化API
    # k值取很小：容易受到异常点的影响, k值取很大：受到样本均衡的问题
    estimator = KNeighborsClassifier()
    # 网格搜索、交叉验证自动确定k值
    k_dict = {'n_neighbors': range(1, 10)}
    estimator = GridSearchCV(estimator, param_grid=k_dict, cv=5)
    # fit数据进行训练
    estimator.fit(x_train, y_train)
    # 模型评估
    print(estimator.score(x_test, y_test))

    y_predict = estimator.predict(x_test)
    print(y_predict == y_test)

    print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
    print("最好的参数模型：\n", estimator.best_estimator_)
    print("每次交叉验证后的准确率结果：\n", estimator.cv_results_)


def _local_save():
    """数据获取需要外网，本地先保存"""
    news = fetch_20newsgroups(subset="all")
    news = pickle.dumps(news)
    with open('data/fetch_20newsgroups.pkl', 'wb') as f:
        f.write(news)


def bayes():
    """
    朴素贝叶斯，假定了特征与特征之间相互独立的贝叶斯公式，
    贝叶斯公式：P(C|F1,F2,...) * P(F1,F2,...) = P(F1,F2,...|C) * P(C)
    分类准确度高，速度快
    """
    with open('data/fetch_20newsgroups.pkl', 'rb') as f:
        news = pickle.loads(f.read())
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.2)

    transfer = TfidfVectorizer()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # print(transfer.get_feature_names())

    estimator = MultinomialNB()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("预测每篇文章的类别：", y_predict)
    print(estimator.score(x_test, y_test))


def trees():
    """
    决策树：可解释性强，但容易过拟合
    随机森林：训练集随机选择、特征随机选择。准确率高、能够有效地运行在大数据集上，处理具有高维特征的输入样本，而且不需要降维
    """
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt",
                        usecols=['pclass', 'age', 'sex', 'survived'])
    titan["age"].fillna(titan["age"].mean(), inplace=True)
    data = titan[['pclass', 'age', 'sex']]
    target = titan['survived']
    # 字典特征抽取
    transfer = DictVectorizer(sparse=False)
    data = transfer.fit_transform(data.to_dict(orient="record"))
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # 指定决策树选择器和深度
    # estimator = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    # estimator.fit(x_train, y_train)
    # 决策树可视化
    # xport_graphviz(estimator, out_file="./data/tree.dot", feature_names=transfer.get_feature_names())

    # 随机森林
    estimator = RandomForestClassifier()
    # 网格搜索、交叉验证自动确定树木数量、最大深度
    param_dict = {'n_estimators': range(100, 1000, 100), "max_depth": range(5, 30, 5)}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=2)
    estimator.fit(x_train, y_train)
    print("在交叉验证中验证的最好结果：\n", estimator.best_score_)
    print("最好的参数模型：\n", estimator.best_estimator_)

    print(estimator.score(x_test, y_test))


trees()
