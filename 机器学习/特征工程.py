import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA


class FeatureExtract(object):
    """特征抽取"""
    def datasets_demo(self):
        """
        对鸢尾花数据集的演示
        :return:
        """
        iris = load_iris()
        # test_size测试集划分比例，random_state随机数种子（相同的种子采样结果相同）
        x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=22)
        print("x_train比例:\n", x_train.shape[0]/iris.data.shape[0])

    def dict_demo(self):
        """
        对字典类型的数据进行特征抽取
        :return:
        """
        data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '深圳', 'temperature': 30}]
        # 是否转为sparse矩阵，sparse矩阵节省内存、处理效率较高
        transfer = DictVectorizer(sparse=False)
        data = transfer.fit_transform(data)
        print(transfer.get_feature_names())
        print("返回的结果:\n", data)
        print(dir(transfer))

    def text_count_demo(self):
        """
        对文本进行特征抽取
        :return:
        """
        en = ["life is short,i like like python", "life is too long,i dislike python"]
        transfer = CountVectorizer()
        data = transfer.fit_transform(en)
        print(transfer.get_feature_names())
        # 利用toarray()进行sparse矩阵转换array数组
        print(data.toarray())

        zh = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
              "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
              "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
        zh_list = []
        for line in zh:
            # jieba处理中文
            text = " ".join(list(jieba.cut(line)))
            zh_list.append(text)
        # Tf-idf打分：如果A词在100份文件中的仅1份文件出现了5次，而本份文件总词语数为100，则tf-idf= 5/100 * lg(100/1) = 0.1
        transfer = TfidfVectorizer(stop_words=["不要", "一种"])
        data = transfer.fit_transform(zh_list)
        print(transfer.get_feature_names())
        print(data.toarray())


class SklearnPreprocessing(object):
    """特征预处理"""
    def __init__(self):
        self.data = pd.DataFrame([[40920, 8.326976, 0.953952, 3],
                                 [14488, 7.153469, 1.673904, 2],
                                 [26052, 1.441871, 0.805124, 1],
                                 [75136, 13.147394, 0.428964, 1],
                                 [38344, 1.669788, 0.134296, 1]],
                                 columns=["milage", "Liters", "Consumtime", "target"])

    def minmax_demo(self):
        """归一化"""
        # 归一化范围feature_range
        transfer = MinMaxScaler(feature_range=(2, 3))
        data = transfer.fit_transform(self.data[['milage', 'Liters', 'Consumtime']])
        print(data)

    def stand_demo(self):
        """标准化,弱化极值的影响"""
        transfer = StandardScaler()
        data = transfer.fit_transform(self.data[['milage', 'Liters', 'Consumtime']])
        print(data)
        print(transfer.var_)


class DataReduction(object):
    """
    降维，分为特征选择和主成分分析。
    特征选择：1.Filter(过滤式)-方差选择法、相关系数，2.Embedded (嵌入式)-决策树、正则化、深度学习
    主成分分析：PCA
    """
    def __init__(self):
        self.data = pd.read_csv("data/factor_returns.csv").iloc[:, 1:-2]
        print(self.data.shape)

    def variance_demo(self):
        """方差选择法，删除低方差特征"""
        # threshold方差阈值
        transfer = VarianceThreshold(threshold=8)
        data = transfer.fit_transform(self.data)
        print(data.shape)

    def pearsonr_demo(self):
        """皮尔逊相关系数计算, |r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关"""
        # pd.plotting.scatter_matrix(self.data)
        # plt.show()
        for idx, column1 in enumerate(self.data.columns[:-1]):
            for column2 in self.data.columns[idx+1:]:
                # pearsonr返回r表示相关系数，p-value表示不相关的概率
                r, _ = pearsonr(self.data[column1], self.data[column2])
                if r >= 0.7:
                    print("指标%s与指标%s之间的相关性大小为%f" % (column1, column2, r))

    def pca_demo(self):
        """PCA降维"""
        data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
        # n_components小数指保留信息占比，整数指定降维到的维数
        transfer = PCA(n_components=0.9)
        res = transfer.fit_transform(data)
        print(res)


class Extremum(object):
    def __init__(self):
        self.data = np.random.normal(0, 1, 10000)
        plt.figure(figsize=(20, 8), dpi=100)

    def change_extremum(self, high, low):
        self.data = np.where(self.data > high, high, self.data)
        self.data = np.where(self.data < low, low, self.data)
        plt.hist(self.data, bins=1000)
        plt.show()

    def tail_handle(self):
        """缩尾处理：将指定分位数区间以外的极值用分位点的值替换掉"""
        self.data = winsorize(self.data, limits=0.025)
        plt.hist(self.data, bins=1000)
        plt.show()

    def percentile(self, up=98, down=2):
        """自定义分位数去极值"""
        up_scale = np.percentile(self.data, up)
        down_scale = np.percentile(self.data, down)
        self.change_extremum(up_scale, down_scale)

    def mad(self):
        """中位数绝对偏差去极值"""
        # median求中位数
        med = np.median(self.data)
        # abs求绝对值
        mad = np.median(abs(self.data - med))
        high = med + 3 * 1.4826 * mad
        low = med - 3 * 1.4826 * mad
        self.change_extremum(high, low)

    def three_sigma(self):
        """正态分布去极值, 不推荐"""
        mean = self.data.mean()
        std = self.data.std()
        high = mean + 3 * std
        low = mean - 3 * std
        self.change_extremum(high, low)


# if __name__ == '__main__':
#     Extremum().three_sigma()
