"""
MP神经元模型：每个输入特征乘以线上权重求和+偏置b --》通过非线性函数--》输出
全连接网路：每个神经元与前后相邻层的每一个神经元都有连接关系
反向传播：从后向前，逐层求损失函数对每层神经元参数的偏导，迭代更新所有参数

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import tensorflow as tf


def iris_classify():
    """利用鸢尾花数据集，实现前向传播、反向传播，可视化loss曲线"""
    iris = load_iris()
    x_data, y_data = iris.data, iris.target
    # 使用相同的随机数种子使特征/标签一一对应
    np.random.seed(116)
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    # tf全局设置随机数种子
    tf.random.set_seed(116)

    x_train = x_data[:-30]
    y_train = y_data[:-30]
    x_test = x_data[-30:]
    y_test = y_data[-30:]

    # 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
    x_train = tf.cast(x_train, tf.float32)
    x_test = tf.cast(x_test, tf.float32)

    # 配对特征值和输出值, 打包
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    # 定义神经网络训练参数
    w = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1))
    b = tf.Variable(tf.random.truncated_normal([3], stddev=0.1))

    epochs = 500
    lr = 0.1
    train_loss_results = []  # 将每轮的loss记录在此列表中，为后续画loss曲线提供数据
    test_acc = []  # 将每轮的acc记录在此列表中，为后续画acc曲线提供数据
    loss_all = 0  # 每轮分4个step，loss_all记录四个step生成的4个loss的和

    ##########################################################################
    m_w, m_b = 0, 0
    beta = 0.9

    v_w, v_b = 0, 0

    beta1, beta2 = 0.9, 0.999
    delta_w, delta_b = 0, 0
    global_step = 0
    ##########################################################################

    # 训练部分
    for epoch in range(epochs):
        for x_train, y_train in train_db:  # batch级别的循环 ，每个step循环一个batch
            global_step += 1
            with tf.GradientTape() as tape:  # with结构记录梯度信息
                y_predict = tf.matmul(x_train, w) + b
                y_predict = tf.nn.softmax(y_predict)
                # 将label转化为one-hot编码
                y_true = tf.one_hot(y_train, depth=3)
                loss = tf.reduce_mean(tf.square(y_predict - y_true))
                loss_all += loss.numpy()
            # 计算loss对各个参数的梯度
            grads = tape.gradient(loss, [w, b])

            # # SDG梯度下降： w1 = w1 - lr * w1_grad    b = b - lr * b_grad
            # w.assign_sub(lr * grads[0])
            # b.assign_sub(lr * grads[1])

            # # SDGM梯度下降，SGD增加一阶动量
            # m_w = beta * m_w + (1 - beta) * grads[0]
            # m_b = beta * m_b + (1 - beta) * grads[1]
            # w.assign_sub(lr * m_w)
            # b.assign_sub(lr * m_b)

            # # Adagrad，SGD增加二阶动量
            # v_w += tf.square(grads[0])
            # v_b += tf.square(grads[1])
            # w.assign_sub(lr * grads[0] / tf.sqrt(v_w))
            # b.assign_sub(lr * grads[1] / tf.sqrt(v_b))

            # # rmsprop，SGD增加二阶动量
            # v_w = beta * v_w + (1 - beta) * tf.square(grads[0])
            # v_b = beta * v_b + (1 - beta) * tf.square(grads[1])
            # w.assign_sub(lr * grads[0] / tf.sqrt(v_w))
            # b.assign_sub(lr * grads[1] / tf.sqrt(v_b))

            # Adam, 结合SDGM的一阶动量和rmsprop的二阶动量
            m_w = beta1 * m_w + (1 - beta1) * grads[0]
            m_b = beta1 * m_b + (1 - beta1) * grads[1]
            v_w = beta2 * v_w + (1 - beta2) * tf.square(grads[0])
            v_b = beta2 * v_b + (1 - beta2) * tf.square(grads[1])

            m_w_correction = m_w / (1 - tf.pow(beta1, int(global_step)))
            m_b_correction = m_b / (1 - tf.pow(beta1, int(global_step)))
            v_w_correction = v_w / (1 - tf.pow(beta2, int(global_step)))
            v_b_correction = v_b / (1 - tf.pow(beta2, int(global_step)))

            w.assign_sub(lr * m_w_correction / tf.sqrt(v_w_correction))
            b.assign_sub(lr * m_b_correction / tf.sqrt(v_b_correction))

        print("epoch {}, loss {}".format(epoch, loss_all/4))
        train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
        loss_all = 0

        # 测试部分
        total_correct, total_number = 0, 0
        for x_test, y_test in test_db:
            y_predict = tf.matmul(x_test, w) + b
            y_predict = tf.nn.softmax(y_predict)
            y_predict = tf.argmax(y_predict, axis=1)
            y_predict = tf.cast(y_predict, dtype=y_test.dtype)

            # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
            correct = tf.cast(tf.equal(y_predict, y_test), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            total_correct += int(correct)
            total_number += x_test.shape[0]

        acc = total_correct / total_number
        test_acc.append(acc)
        print("Test_acc:", acc)
        print("--------------------------")

    # 绘制 loss 曲线
    plt.title('Loss Function Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Loss')  # y轴变量名称
    plt.plot(train_loss_results, label="$Loss$")  # 逐点画出trian_loss_results值并连线，连线图标是Loss
    plt.legend()  # 画出曲线图标
    plt.show()  # 画出图像

    # 绘制 Accuracy 曲线
    plt.title('Acc Curve')  # 图片标题
    plt.xlabel('Epoch')  # x轴变量名称
    plt.ylabel('Acc')  # y轴变量名称
    plt.plot(test_acc, label="$Accuracy$")  # 逐点画出test_acc值并连线，连线图标是Accuracy
    plt.legend()
    plt.show()


def regularization():
    """搭建2层神经网络，并添加正则化"""
    df = pd.read_csv("data/dot.csv")
    x_data = np.array(df[['x1', 'x2']])
    y_data = np.array(df['y_c'])

    x_train = tf.cast(x_data, tf.float32)
    # y_data需要转换为矩阵否则计算损失会错误
    y_train = tf.cast(y_data.reshape(-1, 1), tf.float32)
    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)

    w1 = tf.Variable(tf.random.normal([2, 11]), dtype=tf.float32)
    b1 = tf.Variable(tf.constant(0.01, shape=[11]))
    w2 = tf.Variable(tf.random.normal([11, 1]), dtype=tf.float32)
    b2 = tf.Variable(tf.constant(0.01, shape=[1]))

    # 训练部分
    lr = 0.005
    epochs = 800
    for epoch in range(epochs):
        for x_train, y_train in train_db:
            with tf.GradientTape() as tape:
                h1 = tf.matmul(x_train, w1) + b1
                h1 = tf.nn.relu(h1)
                y = tf.matmul(h1, w2) + b2

                # 求损失函数并L2正则化
                loss_mse = tf.reduce_mean(tf.square(y_train - y))
                loss_regularization = []
                loss_regularization.append(tf.nn.l2_loss(w1))
                loss_regularization.append(tf.nn.l2_loss(w2))
                loss_regularization = tf.reduce_sum(loss_regularization)
                loss = loss_mse + 0.03 * loss_regularization  # REGULARIZER = 0.03
            grads = tape.gradient(loss, [w1, b1, w2, b2])

            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])
            w2.assign_sub(lr * grads[2])
            b2.assign_sub(lr * grads[3])

        if epoch % 20 == 0:
            print('epoch:', epoch, 'loss:', float(loss))

    # 预测部分
    print("*******predict*******")
    # xx在-3到3之间以步长为0.01，yy在-3到3之间以步长0.01,生成间隔数值点
    xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
    # 将xx, yy拉直，并合并配对为二维张量，生成二维坐标点
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid = tf.cast(grid, tf.float32)
    # 将网格坐标点喂入神经网络，进行预测，probs为输出
    probs = []
    for x_predict in grid:
        # 使用训练好的参数进行预测
        h1 = tf.matmul([x_predict], w1) + b1
        h1 = tf.nn.relu(h1)
        y = tf.matmul(h1, w2) + b2  # y为预测结果
        probs.append(y)

    # probs的shape调整成xx的样子
    probs = np.array(probs).reshape(xx.shape)
    Y_c = ['red' if y else 'blue' for y in y_data]
    plt.scatter(df['x1'], df['x2'], color=Y_c)
    # 把坐标xx yy和对应的值probs放入contour函数，给probs值为0.5的所有点上色  plt.show()后 显示的是红蓝点的分界线
    plt.contour(xx, yy, probs, levels=[.5])
    plt.show()


if __name__ == '__main__':
    regularization()
    # iris_classify()

# # y = 4 + 3x1 + 2x2使用tensorflow线性回归
# x1 = np.random.randn(100)
# x2 = np.random.randn(100)
# x = np.array([x1, x2]).T
# y = x1 * 3 + x2 * 2 + 4
# print(x)
# print(y)
#
# # model = tf.keras.Sequential()
# # model.add(tf.keras.layers.Dense(1, input_shape=(1, 2)))
# # print(model.summary())
# #
# # # loss损失函数-mse均方误差
# # model.compile(optimizer='adam', loss='mse')
# #
# # # epochs训练次数
# # model.fit(x, y, epochs=5000)
# #
# # print(model.predict(np.array([[2, 2]])))
#
#
# # 用numpy实现y = 4 + 3x1 + 2x2线性回归
# x = np.column_stack((np.ones(100), x))
# lr = 0.01
# epochs = 5000
# w = np.zeros([1, 3])
# for i in range(5000):
#     dif_y = np.sum(w * x, axis=1) - y
#     aa = 1/len(x) * np.dot(dif_y.reshape(1, -1), x)
#     w = w - lr * aa
# print(w)
