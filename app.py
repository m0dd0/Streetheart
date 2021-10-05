from flask import Flask, render_template, request


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        print(request.get_json())
    return ""


if __name__ == "__main__":
    app.run(debug=True)
