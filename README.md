# EX-8-Implement-a-SimpleLSTM-using-TensorFlow-Keras.
# REG NO:2305001031
# NAME:V.S.SREE VIVEKA
# AIM
To implement a simple Long Short-Term Memory (LSTM) neural network using TensorFlow–Keras for sequence prediction.
# ALGORITHM
Import necessary libraries
Import TensorFlow, Keras layers, NumPy, and other required modules.

Prepare Dataset

Create an input sequence (e.g., numbers 0–49).

Convert the sequence into input-output pairs suitable for the LSTM model.

Reshape data into 3D format: (samples, timesteps, features).

Build the LSTM Model

Initialize a Sequential model.

Add an LSTM layer with required hidden units.

Add a Dense layer for output prediction.

Compile the Model

Use loss function: Mean Squared Error (MSE).

Use optimizer: Adam.

Train the Model

Fit the model using training data for several epochs.

Evaluate/ Predict Output

Use the trained model to predict the next value in the sequence.
# PROGRAM
```python
# EX-8: Simple LSTM using TensorFlow Keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. Prepare Dataset
sequence = np.array([i for i in range(50)])  # sequence 0–49

# Function to split into samples
def split_sequence(seq, n_steps):
    X, y = [], []
    for i in range(len(seq)):
        end_idx = i + n_steps
        if end_idx > len(seq) - 1:
            break
        X.append(seq[i:end_idx])
        y.append(seq[end_idx])
    return np.array(X), np.array(y)

n_steps = 3
X, y = split_sequence(sequence, n_steps)

# Reshape for LSTM → (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 2. Build the Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))

# 3. Compile the Model
model.compile(optimizer='adam', loss='mse')

# 4. Train the Model
model.fit(X, y, epochs=300, verbose=0)

# 5. Predict Next Value
test_input = np.array([47, 48, 49])
test_input = test_input.reshape((1, n_steps, 1))
predicted_value = model.predict(test_input, verbose=0)

print("Predicted next value after [47, 48, 49]: ", predicted_value)

```
# OUTPUT

<img width="1077" height="71" alt="image" src="https://github.com/user-attachments/assets/b0aa2d02-7fe7-45b9-ae8e-3d515594ab21" />

# RESULT
A simple LSTM network was successfully implemented using TensorFlow–Keras. The model was able to learn a numerical sequence and predict the next value accurately.
