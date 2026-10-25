import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "ai4i2020_processed.csv")

def train_model():
    if not os.path.exists(data_path):
        return None, []
    df = pd.read_csv(data_path)
    drop_cols = ['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['Machine failure']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)

model, feature_names = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        try:
            air_temp = float(request.form.get('air_temp', 300.0))
            proc_temp = float(request.form.get('proc_temp', 310.0))
            speed = float(request.form.get('speed', 1500))
            torque = float(request.form.get('torque', 40.0))
            tool_wear = float(request.form.get('tool_wear', 100))
            
            temp_diff = proc_temp - air_temp
            power_proxy = speed * torque
            
            input_data = pd.DataFrame([{
                'Air temperature [K]': air_temp,
                'Process temperature [K]': proc_temp,
                'Rotational speed [rpm]': speed,
                'Torque [Nm]': torque,
                'Tool wear [min]': tool_wear,
                'Temp_Diff [K]': temp_diff,
                'Power_Proxy': power_proxy,
                'Type_Encoded': 0
            }])
            
            input_data = input_data.reindex(columns=feature_names, fill_value=0)
            
            probability = model.predict_proba(input_data)[0][1]
            risk_percent = round(probability * 100, 1)
            
            # Risk Categorization & Dynamic Recommendations
            if risk_percent < 20.0:
                status_class = "success"
                status_title = "Optimal Operating Health"
                badge_text = "NORMAL OPERATION"
                emoji = "⚙️"
                advice = "All operating conditions are within nominal safety thresholds. Continue standard operating procedures."
            elif risk_percent < 50.0:
                status_class = "warning"
                status_title = "Elevated Risk Detected"
                badge_text = "MODERATE RISK"
                emoji = "⚠️"
                advice = "Telemetry indicates elevated stress. Schedule a preventive inspection during the next maintenance window."
            else:
                status_class = "danger"
                status_title = "Critical Failure Imminent"
                badge_text = "CRITICAL RISK"
                emoji = "🚨"
                advice = "Immediate attention required! High torque or tool wear detected. Shut down unit for immediate inspection."

            return render_template(
                'result.html', 
                probability=risk_percent,
                status_class=status_class,
                status_title=status_title,
                badge_text=badge_text,
                emoji=emoji,
                advice=advice,
                air_temp=air_temp,
                proc_temp=proc_temp,
                speed=speed,
                torque=torque,
                tool_wear=tool_wear,
                temp_diff=round(temp_diff, 1)
            )
        except Exception as e:
            return f"Error evaluating machine risk: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)