import os
import subprocess
import sys

# Get paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_path = os.path.join(BASE_DIR, "website", "app.py")

if not os.path.exists(app_path):
    raise FileNotFoundError(f"Could not find web app script at: {app_path}")

print("=== Lab 12: Integrated Predictive Maintenance Web Application ===")
print(f"Launching web app from: {app_path}\n")

# Run the Flask web application
try:
    subprocess.run([sys.executable, app_path], check=True)
except KeyboardInterrupt:
    print("\nWeb application stopped successfully.")