import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# a = tf.constant([1, 5], dtype=tf.int16)
# print(a)
# print(a.dtype)
# print(a.shape)
#
# a = np.arange(0, 5)
# b = tf.convert_to_tensor(a, dtype=tf.int64)
# print(b)

print(tf.zeros([2, 3]))
print(tf.ones(4))
print(tf.fill([2, 2], 9))

a = tf.random.normal([10000, 10000], mean=0.5, stddev=1)
# tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择
b = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)

a = tf.constant([[1, 2, 3], [2, 4, 6]], dtype=tf.int16)
a = tf.cast(a, dtype=tf.float64)
print(tf.reduce_min(a))
print(tf.reduce_max(a))
# argmax返回最大值索引
print(tf.argmax(a, axis=1))
print(tf.reduce_mean(a))
print(tf.reduce_mean(a, axis=1))
print(tf.reduce_sum(a, axis=0))


# tf.Variable()将变量标记为可训练，被标记的变量会在反向传播中记录梯度信息
w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))


# tf中的数学计算
# tf.add() 加
# tf.subtract() 减
# tf.multiply() 乘
# tf.divide() 除
# tf.square() 平方
# tf.pow() 次方
# tf.sqrt(张量名, n次方数字) 开方
# tf.matmul 矩阵乘法
a = tf.pow(tf.constant([2, 3]), 3)
print(a)

a = tf.constant([1,2,3,1,1])
b = tf.constant([0,1,3,4,5])
c = tf.where(tf.greater(a,b), a, b)
print(c)



