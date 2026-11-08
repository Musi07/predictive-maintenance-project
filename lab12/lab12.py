import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------
# Load Pre-trained Machine Learning Model (Lab 6-9)
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Fallback dummy prediction logic if model file isn't loaded
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    """Renders the main input form page (predict.html or index.html)"""
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Processes input sensor telemetry parameters and outputs failure status"""
    if request.method == 'POST':
        try:
            # 1. Extract 5 operational telemetry parameters from HTML form
            air_temp = float(request.form['air_temp'])
            process_temp = float(request.form['process_temp'])
            rotational_speed = float(request.form['rotational_speed'])
            torque = float(request.form['torque'])
            tool_wear = float(request.form['tool_wear'])

            # Combine features into numpy array shape (1, 5)
            features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])

            # 2. Predict using trained classifier or heuristic fallback
            if model:
                prediction_result = model.predict(features)[0]
            else:
                # Rule-based fallback: Torque > 60 Nm or Tool Wear > 200 min signals high failure risk
                if torque > 60 or tool_wear > 200 or (process_temp - air_temp) > 12:
                    prediction_result = 1
                else:
                    prediction_result = 0

            # 3. Render the output result page
            return render_template(
                'result.html',
                prediction=prediction_result,
                air_temp=air_temp,
                process_temp=process_temp,
                rotational_speed=rotational_speed,
                torque=torque,
                tool_wear=tool_wear
            )

        except Exception as e:
            return f"Error processing input parameters: {str(e)}", 400


if __name__ == '__main__':
    # Run local web server on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)