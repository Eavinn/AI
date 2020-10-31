import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.set_printoptions(threshold=np.inf)

(train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
# (train_image, train_label), (test_image, test_label) = tf.keras.datasets.fashion_mnist.load_data()

# 归一化，把输入特征的数值变小更有利于神经网络吸收
train_image, test_image = train_image/255, test_image/255

# 增加通道维度
train_image = train_image.reshape(train_image.shape[0], 28, 28, 1)
# 数据增强，扩充数据集，小数据量上增加模型泛化性
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.15,  # 宽度偏移
    height_shift_range=.15,  # 高度偏移
    horizontal_flip=False,  # 水平翻转
    zoom_range=0.5  # 将图像随机缩放阈量50％
)
image_gen_train.fit(train_image)

# plt.imshow(train_image[0])
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])


# class MnistModel(Model):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.flatten = tf.keras.layers.Flatten()
#         self.d1 = tf.keras.layers.Dense(128, activation="relu")
#         self.d2 = tf.keras.layers.Dense(10, activation="softmax")
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.flatten(inputs)
#         x = self.d1(x)
#         y = self.d2(x)
#         return y
#
#
# model = MnistModel()

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 断点续训, 存取模型
checkpoint_save_path = "data/checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print("---------load model------------")
    model.load_weights(checkpoint_save_path)

    while True:
        # 模型预测
        pre_num = input("输入图片名称进行预测：")
        image_path = 'data/MNIST_FC/{}.png'.format(pre_num)
        img = Image.open(image_path)

        image = plt.imread(image_path)
        plt.set_cmap('gray')
        plt.imshow(image)

        img = img.resize((28, 28), Image.ANTIALIAS)
        img_arr = np.array(img.convert('L'))
        # 图片白底黑字翻转为黑底白字，同时归一化
        img_arr = np.where(img_arr < 200, 255/255.0, 0)
        # 按照batch的格式添加维度(1, 28, 28)
        x_predict = img_arr[tf.newaxis, :]
        result = model.predict(x_predict)
        pred = tf.argmax(result, axis=1)
        tf.print("预测结果: ", pred)

        plt.pause(1)
        plt.close()


# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
#                                                  save_weights_only=True,
#                                                  save_best_only=True)
#
# history = model.fit(image_gen_train.flow(train_image, train_label, batch_size=32),
#                     epochs=20, validation_data=(test_image, test_label), validation_freq=1,
#                     callbacks=[cp_callback])
# model.summary()

# # 参数提取
# with open('data/weights.txt', 'w') as f:
#     for v in model.trainable_variables:
#         f.write(str(v.name) + '\n')
#         f.write(str(v.shape) + '\n')
#         f.write(str(v.numpy()) + '\n')
#
# # 可视化
# _, axes = plt.subplots(1, 2)
# for key in history.history.keys():
#     if key.endswith("loss"):
#         axes[0].plot(history.epoch, history.history.get(key), label=key)
#         axes[0].legend()
#     elif key.endswith("accuracy"):
#         axes[1].plot(history.history.get(key), label=key)
#         axes[1].legend()
# plt.show()
