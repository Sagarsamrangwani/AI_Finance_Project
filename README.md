
# AI Trading Strategy

## Objective
The objective of this project is to develop an AI-driven intraday trading strategy using reinforcement learning and deep learning techniques. The focus is on leveraging the IVV ETF data to create models that predict short-term market movements and optimize trading decisions.

## Team Composition
- Sagar Lal - VR512164
- Gulshan Kumar - VR509655

## Implementation Steps

### 1. Setup Environment
Ensure you have the necessary libraries installed. You can use the following commands to install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow
```

### 2. Data Preprocessing
- Load the dataset using `pandas`.
- Scale the data using `MinMaxScaler` from `scikit-learn` to normalize the input features.
- Split the data into training and testing sets.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
data = pd.read_csv('path_to_your_data.csv')

# Data scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split the data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]
```

### 3. Model Building
- Define the architecture of the neural network using `Sequential` from `tensorflow.keras.models`.
- Use layers such as `Conv1D`, `MaxPooling1D`, `Bidirectional LSTM`, and `Dense`.
- Compile the model with an optimizer like `Adam` and loss function appropriate for regression tasks.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

# Model architecture
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(train_data.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Bidirectional(LSTM(units=50, return_sequences=True)),
    Bidirectional(LSTM(units=50)),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
```

### 4. Training the Model
- Use `EarlyStopping` and `ReduceLROnPlateau` callbacks to optimize the training process.
- Fit the model on the training data and validate it on the testing data.

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train the model
history = model.fit(train_data, train_data, 
                    epochs=100, 
                    batch_size=32, 
                    validation_split=0.2, 
                    callbacks=[early_stopping, reduce_lr])
```

### 5. Evaluation
- Evaluate the model performance using metrics such as `mean_squared_error`, `mean_absolute_error`, and `r2_score`.

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Predictions
predictions = model.predict(test_data)

# Evaluation metrics
mse = mean_squared_error(test_data, predictions)
mae = mean_absolute_error(test_data, predictions)
r2 = r2_score(test_data, predictions)

print(f'MSE: {mse}, MAE: {mae}, R2: {r2}')
```

### 6. Visualization
- Plot the training history and the predictions against the actual values to visualize the performance.

```python
import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Plot predictions vs actual
plt.figure(figsize=(14, 7))
plt.plot(test_data, label='Actual')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
```

### Conclusion
This README provides a comprehensive guide on how to implement the AI Trading Strategy using the provided Jupyter Notebook. Ensure to adjust paths and parameters as needed based on your specific dataset and requirements.
