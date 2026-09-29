# SkyGuard AI - Backend Engine

Official backend service for **SkyGuard AI**, developed for Smart India Hackathon (SIH 2026) under Problem Statement **SIH26073** (AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations).

## 🚀 Overview
This backend provides a high-speed telemetry processing pipeline built with **FastAPI**. It handles real-time multivariate parsing of meteorological data (Temperature, Barometric Pressure, and Relative Humidity), executes unsupervised anomaly detection using **Scikit-Learn**, computes feature attribution via **SHAP**, performs spatial verification against neighboring weather nodes, and executes auto-healing data imputation using **SciPy**.

---

## 🛠️ Tech Stack & Libraries
* **Framework:** FastAPI, Uvicorn
* **Machine Learning & Math:** Scikit-Learn (Isolation Forest), SciPy, NumPy[cite: 2]
* **Deployment:** Render Cloud (Free Tier Production)[cite: 2]

---

## 📊 Core Architecture & Features
1. **Multivariate Anomaly Detection:** Uses an optimized Isolation Forest model to flag hardware spikes, frozen values, and transmission dropouts in real-time streams.
2. **Explainable AI (SHAP):** Computes exact percentage weights for which weather parameter drove the anomaly score.
3. **Spatial Disambiguation:** Cross-checks readings against a 15km radius neighbor node to differentiate local sensor faults from regional weather events.
4. **Self-Healing Imputation:** Automatically intercepts corrupted telemetry frames and reconstructs clean proxy values using SciPy linear interpolation.

---
