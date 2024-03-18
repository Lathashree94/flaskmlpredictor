from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load("fish_model.pkl")


@app.route("/")
def index():
    return render_template("input.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]
    print(input_features)
    prediction = model.predict([input_features])[0]
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
