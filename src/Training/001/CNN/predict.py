import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import r2_score

model = tf.keras.models.load_model("cnn_model.keras", custom_objects={'mse': MeanSquaredError()})

# Load test data
X_test = np.load("X_test.npy")  # Assuming you saved this earlier or save it from your main script
y_test = np.load("y_test.npy")  # Assuming you saved this earlier or save it from your main script

# Predict surface precipitation on the test set
y_pred = model.predict(X_test)

predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print(f"R2 Score: {r2}")
# Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Line of perfect prediction
plt.xlabel("Actual Surface Precipitation")
plt.ylabel("Predicted Surface Precipitation")
plt.title("Actual vs Predicted Surface Precipitation")
plt.grid(True)
plt.show()
