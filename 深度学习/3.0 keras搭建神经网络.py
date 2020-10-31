"""
sigmoid：f(x) = 1/(1+e^-x)
1. 易造成梯度消失。深层神经网络更新参数时需要从输出层到输入层逐层链式求导，sigmoid函数的导数在0-0.25之间多次相乘导致导数趋近0造成梯度消失
2. 输出非0均值，收敛慢
3. 幂运算复杂，训练时间长

Tanh: f(x) = 1-e^(-2x) / 1+e^(-2x)
1. 易造成梯度消失
2. 输出0均值, 收敛快
3. 幂运算复杂，训练时间长

Relu： f(x) = max(x,0)
1. 在正区间解决了梯度消失问题
2. 输出非0均值，收敛慢
3. 计算速度快，收敛速度快
4. Dead relu

Leaky Relu： f(x) = max(ax, x)
4. 解决Dead relu问题

损失函数：均方误差MSE、交叉熵CE、也可以自定义损失函数
二分类问题：binary_crossentropy
多分类问题：目标集one—hot编码使用categorical_crossentropy，整数编码使用sparse_categorical_crossentropy

建议：
1.首选relu函数
2.学习率设置较小值
3.输入特征标准化， N(0, 1)
4.初始参数中心化， N(0, 2/输入特征个数)

"""

from sklearn import datasets
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

x_train = datasets.load_iris().data
y_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
# ])


class IrisModel(Model):
    """用class搭建神经网络结构"""
    def __init__(self):
        super(IrisModel, self).__init__()
        # 定义网络结构块
        self.d1 = tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs, training=None, mask=None):
        # 调用网络结构块，实现前向传播
        y = self.d1(inputs)
        return y


model = IrisModel()

# 如果神经网络输出前经过了概率分布（如softmax）from_logits需要选False，反之选True。
# 神经网络前向输出为概率分布，目标集是数值，metrics需要选择sparse_categorical_accuracy
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# validation_split划分为测试集的比例， validation_freq验证准确率的迭代次数间隔
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

model.summary()
