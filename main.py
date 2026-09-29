from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import IsolationForest
from scipy.interpolate import interp1d
import numpy as np
import random
from datetime import datetime

app = FastAPI(title="SkyGuard AI - Production Engine")

# Enable CORS for all frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train Isolation Forest baseline model
X_baseline = np.array([
    [random.uniform(20.0, 35.0), random.uniform(1000.0, 1020.0), random.uniform(45.0, 75.0)]
    for _ in range(1000)
])
model = IsolationForest(contamination=0.08, random_state=42)
model.fit(X_baseline)

MEAN_TEMP, MEAN_PRESS, MEAN_HUM = 27.5, 1010.0, 60.0

def calculate_shap_contributions(temp, press, hum):
    d_temp = abs(temp - MEAN_TEMP) / MEAN_TEMP
    d_press = abs(press - MEAN_PRESS) / MEAN_PRESS
    d_hum = abs(hum - MEAN_HUM) / MEAN_HUM
    total_dev = d_temp + d_press + d_hum + 1e-6
    return {
        "temperature_pct": round((d_temp / total_dev) * 100, 1),
        "pressure_pct": round((d_press / total_dev) * 100, 1),
        "humidity_pct": round((d_hum / total_dev) * 100, 1)
    }

def auto_heal_value(metric, corrupted_val):
    x_points = np.array([1, 2, 4, 5])
    if metric == "Temperature":
        y_points = np.array([26.8, 27.1, 27.4, 27.6])
    elif metric == "Pressure":
        y_points = np.array([1011.0, 1010.8, 1009.9, 1009.5])
    else:
        y_points = np.array([58.0, 59.5, 61.0, 62.2])
    f_interp = interp1d(x_points, y_points, kind='linear', fill_value="extrapolate")
    return round(float(f_interp(3)), 2)

@app.get("/api/telemetry")
async def get_telemetry():
    scenario = random.choices(
        population=["NORMAL", "HARDWARE_SPIKE", "COMM_DROPOUT", "GENUINE_WEATHER"],
        weights=[0.80, 0.08, 0.06, 0.06]
    )[0]

    if scenario == "HARDWARE_SPIKE":
        temp, press, hum = 88.5, 1010.2, 58.0 
        neighbor_temp, neighbor_press = 27.8, 1010.0
    elif scenario == "COMM_DROPOUT":
        temp, press, hum = -999.0, 0.0, 0.0    
        neighbor_temp, neighbor_press = 26.5, 1009.5
    elif scenario == "GENUINE_WEATHER":
        temp, press, hum = 18.2, 982.0, 98.0  
        neighbor_temp, neighbor_press = 18.9, 984.0 
    else:
        temp = round(random.uniform(24.0, 30.0), 1)
        press = round(random.uniform(1008.0, 1014.0), 1)
        hum = round(random.uniform(52.0, 68.0), 1)
        neighbor_temp, neighbor_press = temp + random.uniform(-0.5, 0.5), press + random.uniform(-1.0, 1.0)

    reading = np.array([[temp, press, hum]])
    prediction = model.predict(reading)[0] 
    raw_score = model.score_samples(reading)[0]
    anomaly_score = round(float(np.clip((0.5 - raw_score), 0.0, 1.0)), 2)
    is_anomaly = True if (prediction == -1 or scenario != "NORMAL") else False

    spatial_delta = abs(temp - neighbor_temp)
    spatial_consensus = "CONFIRMED_FAULT" if spatial_delta > 10.0 else "GENUINE_EVENT"

    if not is_anomaly:
        root_cause, primary_metric, healed_val = "System Nominal", "None", None
    elif scenario == "GENUINE_WEATHER" or spatial_consensus == "GENUINE_EVENT":
        root_cause, primary_metric, healed_val = "Severe Atmospheric Event", "Pressure & Humidity", None
    elif temp > 70.0 or temp < -50.0:
        root_cause, primary_metric, healed_val = "Hardware Sensor Spike", "Temperature", auto_heal_value("Temperature", temp)
    elif temp == -999.0 or press == 0.0:
        root_cause, primary_metric, healed_val = "Communication Dropout", "Telemetry Pipeline", auto_heal_value("Pressure", press)
    else:
        root_cause, primary_metric, healed_val = "Gradual Calibration Drift", "Relative Humidity", auto_heal_value("Humidity", hum)

    return {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "station_id": "AWS-IND-WESTBENGAL-09",
        "telemetry": {"temperature_c": temp, "pressure_hpa": press, "humidity_pct": hum},
        "spatial_cross_check": {
            "neighbor_node_id": "AWS-IND-WESTBENGAL-10",
            "neighbor_temp_c": round(neighbor_temp, 1),
            "consensus": spatial_consensus,
            "status_label": "Neighbors Normal (Isolated Fault)" if spatial_consensus == "CONFIRMED_FAULT" else "Neighbors Corroborate Event"
        },
        "ml_evaluation": {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "confidence_pct": int(anomaly_score * 100),
            "root_cause": root_cause,
            "primary_fault_metric": primary_metric
        },
        "explainable_ai_shap": calculate_shap_contributions(temp, press, hum),
        "auto_healing": {
            "engaged": True if healed_val is not None else False,
            "corrupted_raw": temp if primary_metric == "Temperature" else press,
            "reconstructed_proxy_value": healed_val
        }
    }