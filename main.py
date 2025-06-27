import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = info removed, 2 = warning+, 3 = error only
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""loading all the dataset"""

# Load datasets
demand_data = pd.read_csv("crop_price_demand_dataset.csv", parse_dates=["Date"])
weather_data = pd.read_csv("weather_data.csv", parse_dates=["Date"])
festival_data = pd.read_csv("festival_holiday_data.csv", parse_dates=["Date"])

print(demand_data.head())
print(weather_data.head())
print(festival_data.head())

"""merge the dataset on"Date"
"""

# Convert 'Date' columns to datetime objects
demand_data['Date'] = pd.to_datetime(demand_data['Date'], format='%d-%m-%Y')
weather_data['Date'] = pd.to_datetime(weather_data['Date'], format='%d-%m-%Y')
festival_data['Date'] = pd.to_datetime(festival_data['Date'])

# Merge demand, weather, and festival datasets
df = demand_data.merge(weather_data, on="Date", how="left")  # Left join to retain demand data
df = df.merge(festival_data, on="Date", how="left")  # Merge festivals

# Display merged dataset
print(df.head())

"""[A] Preprocessing"""

import pandas as pd

# Load crop price data
df_prices = pd.read_csv("crop_price_demand_dataset.csv")

# Load weather data
df_weather = pd.read_csv("weather_data.csv")

# Load festival data (manual)
df_festivals = pd.read_csv("festival_holiday_data.csv")
print(df_prices.head())
print(df_weather.head())
print(df_festivals.head())

"""[B] Clean and Format Data

(1) Removing Missing Values
"""

df_prices.dropna(inplace=True)
df_weather.dropna(inplace=True)
df_festivals.dropna(inplace=True)
print(df_prices.head())
print(df_weather.head())
print(df_festivals.head())

df

"""(2) Convert Date Column"""

df_prices['Date']= pd.to_datetime(df_prices['Date'], format='%d-%m-%Y')
df_weather['Date'] = pd.to_datetime(df_weather['Date'], format='%d-%m-%Y')
df_festivals['Date'] = pd.to_datetime(df_festivals['Date'])
print(df_prices.head())
print(df_weather.head())
print(df_festivals.head())

"""Merge Datasets"""

df=df_prices.merge(df_weather, on="Date", how="left")
df=df.merge(df_festivals, on="Date", how="left")
print(df.head())

"""Normalize Data for AI model"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load datasets, ensure file paths are correct
# If files are in the current directory, simply use the file names.
# Replace with actual file paths if they are elsewhere.
weather_df = pd.read_csv("weather_data.csv")
crop_df = pd.read_csv("crop_price_demand_dataset.csv")

# Merge datasets on "Date"
df = pd.merge(crop_df, weather_df, on="Date", how="inner")

# Rename columns
df.rename(columns={
    "Price (Rs./kg)": "prices",
    "Demand (Tons)": "demand",
    "Temperature (°C)": "temperature",
    "Rainfall (mm)": "rainfall"
}, inplace=True)

# Apply Min-Max Scaling to numerical columns only
scaler = MinMaxScaler()
df[['prices', 'demand', 'temperature', 'rainfall']] = scaler.fit_transform(df[['prices', 'demand', 'temperature', 'rainfall']])

# Print first few rows
print(df.head())

"""Importing Libraries and Loading the data"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load datasets
weather_df = pd.read_csv("weather_data.csv")
crop_df = pd.read_csv("crop_price_demand_dataset.csv")

# Merge datasets on "Date"
df = pd.merge(crop_df, weather_df, on="Date", how="inner")

# Rename columns for consistency
df.rename(columns={
    "Price (Rs./kg)": "prices",
    "Demand (Tons)": "demand",
    "Temperature (°C)": "temperature",
    "Rainfall (mm)": "rainfall"
}, inplace=True)

# Select relevant features
features = ["prices", "demand", "temperature", "rainfall"]

# Scale the data
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Print first few rows
print(df.head())

"""Data Pre-Processing"""

# Normalize data using MinMaxScaler (LSTM performs better with scaled data)
scaler = MinMaxScaler(feature_range=(0, 1))
df["demand"] = scaler.fit_transform(df["demand"].values.reshape(-1, 1))

# Convert time-series data into sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (e.g., past 30 days of data)
seq_length = 40
X, y = create_sequences(df["demand"].values, seq_length)

# Split into training & testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train

"""Build and Train LSTM Models"""

# Define LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

# Compile the model
lstm_model.compile(optimizer="adam", loss="mean_squared_error",metrics=[MeanAbsoluteError(), RootMeanSquaredError()])

# Train the model
history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Plot training loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.show()

"""Make Predictions and Evaluate Model"""

# Make predictions
y_pred = lstm_model.predict(X_test)

# Reverse scaling to original demand values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)

# Plot actual vs predicted demand
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label="Actual Demand")
plt.plot(y_pred_actual, label="Predicted Demand", linestyle="dashed")
plt.xlabel("Days")
plt.ylabel("Demand")
plt.title("Actual vs Predicted Demand")
plt.legend()
plt.show()

sequence_length = 40  # Ensure this matches the LSTM input sequence length
future_inputs = X_test[-1].reshape(seq_length,1)  # (1, sequence_length, num_features)
future_predictions = []

for _ in range(30):
        # Predict next day's demand
    future_predictions.append(lstm_model.predict(future_inputs))

future_predictions

"""predict future demand"""

def predict_future_demand(lstm_model, last_n_days, days_to_predict):
    future_inputs = last_n_days
    future_predictions = []

    for _ in range(days_to_predict):
        prediction = lstm_model.predict(future_inputs.reshape(seq_length, 1))
        future_predictions.append(prediction[0, 0])
        future_inputs = np.roll(future_inputs, -1)
        future_inputs[-1] = prediction[0, 0]

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Predict next 30 days of demand
last_n_days = X_test[-1]  # Last sequence from test set
future_demand = predict_future_demand(lstm_model, last_n_days, 30)

# Plot future demand predictions
plt.figure(figsize=(12, 5))
plt.plot(future_demand, label="Future Predicted Demand", color="red")
plt.xlabel("Days Ahead")
plt.ylabel("Demand")
plt.title("Predicted Demand for Next 30 Days")
plt.legend()
plt.show()


# Import the correct module
from prophet import Prophet
import pandas as pd

# Load the datasets (if you haven't already)
weather_df = pd.read_csv("weather_data.csv")
crop_df = pd.read_csv("crop_price_demand_dataset.csv")

# Merge datasets on "Date" to create df
df = pd.merge(crop_df, weather_df, on="Date", how="inner")

# Sample DataFrame (Ensure your df has correct 'Date' format)
df['Date'] = pd.to_datetime(df['Date'])  # Convert to datetime if not already

# Fix: Use the correct column name 'Price (Rs./kg)'
df_prophet = df[['Date', 'Price (Rs./kg)']].rename(columns={'Date': 'ds', 'Price (Rs./kg)': 'y'})

# Train the model
model = Prophet()
model.fit(df_prophet)

# Predict next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
print (df)

# Plot forecast
model.plot(forecast)

def predict_future_demand(model, last_n_days, days_to_predict, sequence_length, scaler, num_features=4):
    future_inputs = last_n_days.reshape(1, sequence_length, 1)  # Reshape to (1, sequence_length, num_features)
    future_predictions = []
    for _ in range(days_to_predict):
        # Predict next day's demand
        prediction = model.predict(future_inputs)
        future_predictions.append(prediction)
        print("_" * 50)

        # This block needs to be indented to be part of the loop
        # Copy last row and replace demand with prediction
        new_row = future_inputs[0, -1,].copy()
        new_row = prediction[0, 0] # Now prediction is defined in this scope

        # Slide window: drop first row and append new row
        future_inputs = np.roll(future_inputs, shift=-1, axis=1)
        future_inputs[0, -1, :] = new_row  # Update last row


    # Convert predicted values to original scale
    dummy_array = np.zeros((len(future_predictions), num_features))  # Create dummy array
    dummy_array[:, 1] = [p[0,0] for p in future_predictions]  # Set predicted demand in column index 1, convert from list of arrays to 1D array
    future_demand_original = scaler.inverse_transform(dummy_array)[:, 1]  # Inverse transform only demand

    return future_demand_original # Return meaningful values
    plt.plot(future_demand_original)
    print(future_demand_original)

"""extracting the image dataset"""

import zipfile, os
from tensorflow.keras.preprocessing.image import ImageDataGenerator #import ImagedataGenerator

zip_path = zip_path = r"D:\Users\HP\PycharmProjects\PythonProject5\synthetic_realistic_crop_dataset (1).zip"
extract_to = r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Zip file extracted successfully!")

# Load Dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Use the path to the extracted data directory
train_generator = datagen.flow_from_directory(r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images", target_size=(128, 128), batch_size=32, class_mode='categorical', subset='training')
val_generator = datagen.flow_from_directory(r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images", target_size=(128, 128), batch_size=32, class_mode='categorical', subset='validation')

# Define CNN Model
from tensorflow.keras.models import Sequential  # Import Sequential class
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 Classes (A, B, C)
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Predict Quality of New Image
# Predict Quality of New Image
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define CNN Model (same as before)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 Classes (A, B, C)
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Option 1: Use file dialog to select an image
def select_image_with_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Set initial directory to your extracted images folder
    initial_dir = r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images"

    image_path = filedialog.askopenfilename(
        title="Select a crop image",
        initialdir=initial_dir if os.path.exists(initial_dir) else None,
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )

    return image_path


# Option 2: List and select from available images
def list_and_select_image():
    image_dir = r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

    if not os.path.exists(image_dir):
        print(f"Directory does not exist: {image_dir}")
        return None

    # Find all image files
    image_files = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No image files found in the directory")
        return None

    print("Available images:")
    for i, img_path in enumerate(image_files[:10]):  # Show first 10 images
        print(f"{i}: {os.path.basename(img_path)}")

    try:
        choice = int(input(f"Select image (0-{min(9, len(image_files) - 1)}): "))
        if 0 <= choice < len(image_files):
            return image_files[choice]
        else:
            print("Invalid choice")
            return None
    except ValueError:
        print("Invalid input")
        return None


# Option 3: Hardcode a specific image path
def use_specific_image():
    # Replace 'your_image.jpg' with an actual image filename in your directory
    image_path = r"D:\Users\HP\PycharmProjects\PythonProject5\extracted_images\your_image.jpg"

    if os.path.exists(image_path):
        return image_path
    else:
        print(f"Image not found: {image_path}")
        return None


# Choose one of the methods above
print("Choose image selection method:")
print("1. File dialog (GUI)")
print("2. List available images")
print("3. Use specific image path")

try:
    method = int(input("Enter choice (1-3): "))

    if method == 1:
        image_path = select_image_with_dialog()
    elif method == 2:
        image_path = list_and_select_image()
    elif method == 3:
        image_path = use_specific_image()
    else:
        print("Invalid choice")
        image_path = None

except ValueError:
    print("Invalid input, using file dialog")
    image_path = select_image_with_dialog()

# Process the selected image
if image_path and os.path.exists(image_path):
    print(f"Loading image: {image_path}")

    # Load and preprocess the image
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not load image. Check if the file is a valid image.")
    else:
        print(f"Original image shape: {image.shape}")

        # Preprocess the image
        image_resized = cv2.resize(image, (128, 128))
        image_normalized = image_resized / 255.0
        image_batch = image_normalized.reshape(1, 128, 128, 3)

        # Make prediction
        prediction = model.predict(image_batch)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        # Grade information
        grade_info = {
            0: {
                "grade": "A Grade",
                "description": "Highest quality crops. No visible defects. Suitable for premium markets.",
                "feedback": "Suggestion: Sell directly to high-value buyers or export."
            },
            1: {
                "grade": "B Grade",
                "description": "Medium quality crops. Minor defects or uneven.",
                "feedback": "Suggestion: Sell to local markets or consider light processing."
            },
            2: {
                "grade": "C Grade",
                "description": "Low quality crops. Visibly damaged.",
                "feedback": "Suggestion: Use for animal feed, composting, or industrial processing."
            }
        }

        # Display results
        result = grade_info[predicted_class]
        print(f"\n Image Analysis Complete!")
        print(f" Image: {os.path.basename(image_path)}")
        print(f" Predicted Grade: {result['grade']}")
        print(f" Explanation: {result['description']}")
        print(f" Confidence: {confidence * 100:.2f}%")
        print(f" {result['feedback']}")

else:
    print("No image selected or image file not found.")