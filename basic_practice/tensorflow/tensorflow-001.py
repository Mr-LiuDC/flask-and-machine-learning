import matplotlib.pyplot as plt
import tensorflow as tf

boston_housing = tf.keras.datasets.boston_housing
(train_x, train_y), (_, _) = boston_housing.load_data(test_split=0)

print(tf.__version__)

plt.figure(figsize=(5, 5))
plt.scatter(train_x[:, 5], train_y)
plt.xlabel("RM")
plt.ylabel("Price($1000's)")
plt.title("5. RM-Price")
plt.show()
