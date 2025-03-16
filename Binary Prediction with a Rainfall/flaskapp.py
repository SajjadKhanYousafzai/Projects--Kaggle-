from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with open("rainfall_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define the home route
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        # Get form inputs
        features = [
            float(request.form["pressure"]),
            float(request.form["maxtemp"]),
            float(request.form["temparature"]),
            float(request.form["mintemp"]),
            float(request.form["dewpoint"]),
            float(request.form["humidity"]),
            float(request.form["cloud"]),
            float(request.form["sunshine"]),
            float(request.form["winddirection"]),
            float(request.form["windspeed"])
        ]

        # Convert to NumPy array and reshape for model input
        features_array = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features_array)[0]
        prediction = "Yes, it will rain! ☔" if prediction == 1 else "No, it won't rain. ☀️"

    return render_template("index.html", prediction=prediction)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
