import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Check if file exists
file_path = r"F:\Academics\Research_paper\Programs\Dataset\spr\gmi\training\gridded\tabular\target\target.nc"
print(os.path.exists(file_path))

# Load dataset
dataset = xr.open_dataset(file_path)

# Extract features and target variable
radar_quality_index = dataset['radar_quality_index'].values
valid_fraction = dataset['valid_fraction'].values
precip_fraction = dataset['precip_fraction'].values
snow_fraction = dataset['snow_fraction'].values
hail_fraction = dataset['hail_fraction'].values
convective_fraction = dataset['convective_fraction'].values
stratiform_fraction = dataset['stratiform_fraction'].values
surface_precip = dataset['surface_precip'].values

# Prepare input data
input_data = np.stack([radar_quality_index, valid_fraction, precip_fraction,
                       snow_fraction, hail_fraction, convective_fraction,
                       stratiform_fraction], axis=1)

# Check the shape of input_data
print("Input data shape:", input_data.shape)  # Expected: (samples, 7)

scaler = StandardScaler()
input_data = scaler.fit_transform(input_data.reshape(-1, input_data.shape[-1])).reshape(input_data.shape)

# Reshape input for Conv1D
input_reshaped = np.expand_dims(input_data, axis=2)  # Shape should be (samples, 7, 1)
print("Reshaped input shape:", input_reshaped.shape)  # Expected: (samples, 7, 1)

# Split data
split_index = int(0.8 * len(input_reshaped))
X_train, X_test = input_reshaped[:split_index], input_reshaped[split_index:]
y_train, y_test = surface_precip[:split_index], surface_precip[split_index:]

# Check the shape of training data
print("X_train shape:", X_train.shape)  # Expected: (train_samples, 7, 1)
print("y_train shape:", y_train.shape)  # Expected: (train_samples,)

# Save test data for separate plotting script
#np.save("X_test.npy", X_test)
#np.save("y_test.npy", y_test)
# Save train data for separate plotting script
#np.save("X_train.npy", X_train)
#np.save("y_train.npy", y_train)

# Define model
# Define model
model = models.Sequential([
    Input(shape=(input_data.shape[1], 1)),
    layers.Flatten(input_shape=(input_data.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Display model summary
model.summary()

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])


# Train model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
# Save the trained model to a file
model.save("cnn_model.keras")
# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
