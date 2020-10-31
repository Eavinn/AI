import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

input_word = "abcde"

# 单时间步数预测
# x_train = np.eye(len(input_word))
# y_train = np.array(list(range(1, len(input_word))) + [0])

# 多时间步数预测
# x_train = np.empty((5, 0))
# for i in range(4):
#     order = list(range(i, 5)) + list(range(i))
#     new_arr = np.eye(5)[order]
#     x_train = np.hstack([x_train, new_arr])
# x_train = x_train.reshape((5, 4, 5))
# y_train = np.array([4, 0, 1, 2, 3])

# 单时间步数增加Embedding预测
# x_train = [0, 1, 2, 3, 4]
# y_train = [1, 2, 3, 4, 0]

# 多时间步数增加Embedding预测
input_word = "abcdefghijklmnopqrstuvwxyz"

x_train = []
y_train = []
training_set_scaled = list(range(len(input_word)))
for i in range(4, 26):
    x_train.append(training_set_scaled[i - 4:i])
    y_train.append(training_set_scaled[i])

print(x_train)


np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

# 使x_train符合SimpleRNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
# 此处整个数据集送入，送入样本数为len(x_train)；输入1个字母出结果，循环核时间展开步数为1; 表示为独热码有5个输入特征，每个时间步输入特征个数为5
# x_train = x_train.reshape(len(x_train), 1, 5)

# 使x_train符合Embedding输入要求：[送入样本数， 循环核时间展开步数]
x_train = np.reshape(x_train, (len(x_train), 4))
y_train = np.array(y_train)

model = tf.keras.Sequential([
    Embedding(26, 2),  # 词汇量大小、编码维度
    SimpleRNN(10),  # 记忆体个数
    Dense(26, activation="softmax")
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_save_path = "./data/checkpoint/rnn_onehot_1pre1.ckpt"

# if os.path.exists(checkpoint_save_path + '.index'):
#     print('-------------load the model-----------------')
#     model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='loss')  # 由于fit没有给出测试集，不计算测试集准确率，根据loss，保存最优模型

history = model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[cp_callback])
model.summary()

############### predict #############

for i in range(5):
    alphabet1 = input("input test alphabet:")
    # alphabet = np.eye(1, 5, k=input_word.index(alphabet1))
    # alphabet = np.reshape(alphabet, (1, 1, 5))
    # alphabet = np.eye(5)[[input_word.index(alph) for alph in alphabet1]]
    # alphabet = np.reshape(alphabet, (1, 4, 5))
    # alphabet = input_word.index(alphabet1)
    # alphabet = np.reshape(alphabet, (1, 1))
    alphabet = [input_word.index(alph) for alph in alphabet1]
    alphabet = np.reshape(alphabet, (1, 4))
    result = model.predict([alphabet])
    pred = int(tf.argmax(result, axis=1))
    tf.print(alphabet1 + '->' + input_word[pred])


