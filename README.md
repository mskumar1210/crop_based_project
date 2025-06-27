# AI-Based Waste Reduction in Farm-to-Seller Marketplace

##  Overview
This project presents a data-driven, AI-powered solution to reduce food waste and empower farmers by predicting crop demand and grading crop quality using machine learning and deep learning techniques.

The system integrates **LSTM-based demand forecasting** with **CNN-based crop image quality analysis**, helping farmers make informed decisions on crop pricing, quality-based distribution, and market timing.

---

##  Problem Statement
Farmers often:
- Lack real-time market demand visibility
- Face exploitation by intermediaries
- Cannot assess or grade crop quality themselves
- Experience losses due to overproduction and lack of storage/logistics
- Are impacted by seasonal demand, festivals, and unpredictable weather

---

##  Objectives
- Predict future crop demand using historical trends, weather, and festivals
- Classify crop quality from images into Grades A, B, and C
- Recommend fair pricing based on quality
- Reduce post-harvest food waste and inefficiencies

---

##  Tech Stack

| Area | Tools / Technologies |
|------|----------------------|
| Programming | Python |
| ML Libraries | TensorFlow, Keras, Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Image Processing | OpenCV |
| Deployment | FastAPI |
| GUI | Tkinter |
| Forecasting | Facebook Prophet |
| Dataset | Synthetic + Realistic (CSV and Image) |

---

##  Dataset Overview

### 1. `crop_price_demand_dataset.csv`
- Date-wise crop demand (Tons) and price (Rs./kg)
- Historical trends across multiple years

### 2. `weather_data.csv`
- Temperature and rainfall mapped to date
- Critical for seasonal impact on demand

### 3. `festival_holiday_data.csv`
- Dates of festivals/holidays with estimated demand spikes
- Used as a categorical feature in forecasting

### 4. `synthetic_realistic_crop_dataset.zip`
- Image dataset categorized into **A Grade**, **B Grade**, **C Grade**
- Synthetic but visually representative of real crop quality

---

## Features

###  Demand Forecasting (LSTM)
- Uses a sequence of past demand data
- Input: 40 days
- Output: Next 30 days forecast
- Evaluated using RMSE & MAE
- Visualized using `matplotlib`

###  Crop Quality Prediction (CNN)
- Image classification model trained on 3 quality classes
- Classifies uploaded image as A, B, or C Grade
- Returns prediction confidence + business suggestion

###  Integration
- Frontend for farmers using Tkinter to upload or capture crop images
- Backend served via FastAPI to handle model inference

---

