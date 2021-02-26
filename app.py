from flask import Flask, send_from_directory, render_template

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello World !"


@app.route('/mnist-index.html')
def main():
    return render_template('mnist-index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(app.root_path, 'static/images/flask.png', mimetype='image/vnd.microsoft.icon')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
