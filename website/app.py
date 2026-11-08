import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------
# Load Pre-trained Machine Learning Model (Lab 6-9)
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Warning: Could not load model.pkl ({e}).")

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.route('/')
def home():
    """Renders the main input form page (predict.html)"""
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Processes input sensor telemetry parameters and outputs machine status"""
    if request.method == 'POST':
        try:
            # 1. Extract 5 operational telemetry parameters from HTML form
            air_temp = float(request.form['air_temp'])
            process_temp = float(request.form['process_temp'])
            rotational_speed = float(request.form['rotational_speed'])
            torque = float(request.form['torque'])
            tool_wear = float(request.form['tool_wear'])

            # Calculate physical indicator thresholds
            temp_diff = process_temp - air_temp

            # 2. Enforce explicit physical failure thresholds
            # High Torque (>=60), High Tool Wear (>=200), or High Temp Gap (>=12) trigger Failure
            if torque >= 60 or tool_wear >= 200 or temp_diff >= 12:
                prediction_result = 1
            elif model:
                # Use ML classifier for intermediate boundary cases
                features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])
                prediction_result = int(model.predict(features)[0])
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