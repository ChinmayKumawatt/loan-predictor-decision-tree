from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load model
model = joblib.load("decision_tree_model.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Gender"]),
            float(request.form["Married"]),
            float(request.form["Dependents"]),
            float(request.form["Education"]),
            float(request.form["Self_Employed"]),
            float(request.form["ApplicantIncome"]),
            float(request.form["CoapplicantIncome"]),
            float(request.form["LoanAmount"]),
            float(request.form["Loan_Amount_Term"]),
            float(request.form["Credit_History"]),
            float(request.form["Property_Area"]),
        ]

        prediction = model.predict([features])[0]

        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)
