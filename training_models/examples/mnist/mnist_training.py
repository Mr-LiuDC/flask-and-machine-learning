import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''载入并准备好 MNIST 数据集。将样本从整数转换为浮点数'''
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

print("训练集的图片数据维度：", train_images.shape)
print("训练集的标签数据维度：", train_labels.shape)
print("测试集的图片数据维度：", test_images.shape)
print("测试集的标签数据维度：", test_labels.shape)
# 查看前五张图片
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap="binary")
    plt.xlabel(train_labels[i])
plt.show()

'''为训练选择优化器和损失函数'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''训练并验证模型'''
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels, verbose=2)

'''保存训练得到的模型'''
model.save("../trained_models/mnist_trained_model")

'''加载已有的训练模型'''
mnist_dense = tf.keras.models.load_model("../trained_models/mnist_trained_model")
mnist_dense.summary()

'''识别预测手写数字'''
forecast = train_images[1].reshape(1, -1)
res = mnist_dense.predict(forecast)
print(res)
print("预测的数字为：", np.argmax(res))
print("预测正确的概率为：", res.max())
print("实际数字为：", train_labels[1])
