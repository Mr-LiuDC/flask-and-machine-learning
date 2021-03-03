import os

import numpy as np
import tensorflow as tf
from flask import Flask, send_from_directory, render_template, json, request

from training_models.mnist.convolution_training import CNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello World !"


@app.route("/api/mnist", methods=['POST'])
def mnist():
    input_data = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    model_1 = tf.keras.models.load_model('./training_models/trained_models/my_mnist_trained_model.h5')
    pred1 = model_1.predict(input_data.reshape(-1, 28, 28, 1)).flatten().tolist()

    latest = tf.train.latest_checkpoint('./training_models/mnist/ckpt')
    cnn = CNN()
    # 恢复网络权重
    cnn.model.load_weights(latest)
    pred12 = cnn.model.predict(input_data.reshape(-1, 28, 28, 1)).flatten().tolist()
    return json.jsonify(results=[pred1, pred12])


@app.route('/mnist-index.html')
def main():
    return render_template('mnist-index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'static/images/flask.png', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
