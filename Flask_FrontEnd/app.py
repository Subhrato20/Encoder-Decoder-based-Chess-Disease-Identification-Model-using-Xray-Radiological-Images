from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/newpage1")
def newpage1():
    return render_template("newpage1.html")

@app.route("/newpage2")
def newpage2():
    return render_template("newpage2.html")

@app.route("/newpage3")
def newpage3():
    return render_template("newpage3.html")

@app.route("/button1")
def button1():
    return redirect(url_for('newpage1'))

@app.route("/button2")
def button2():
    return redirect(url_for('newpage2'))

# @app.route("/button3)
# def button3():
#     return redirect(url_for('newpage3'))

if __name__ == "__main__":
    app.run(debug=True)
