from flask import Flask, send_from_directory

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello World !"


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'static/images/flask.png', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
