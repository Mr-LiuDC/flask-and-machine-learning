import tensorflow as tf

'''载入并准备好 MNIST 数据集。将样本从整数转换为浮点数'''
mnist = tf.keras.datasets.mnist
(train_image, train_labels), (test_image, test_labels) = mnist.load_data()
train_image, test_image = train_image / 255.0, test_image / 255.0

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
model.fit(train_image, train_labels, epochs=5)
model.evaluate(test_image, test_labels, verbose=2)
