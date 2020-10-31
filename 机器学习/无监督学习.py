import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def k_means():
    """
    聚类:随机设置n个点为聚类中心，通过距离划分类型，然后重新计算每个聚类的中心直到与原中心点一样为止。
    """
    df = pd.DataFrame()
    df["A"] = list(range(5)) + list(range(15, 20)) + list(range(30, 38))

    estimator = KMeans(n_clusters=3)
    estimator.fit(df)
    y_predict = estimator.predict(df)

    print(y_predict)
    # 轮廓系数判断模型优劣, 值介于 [-1,1] ，越趋近于1代表内聚度和分离度都相对较优
    print(silhouette_score(df, y_predict))


k_means()
