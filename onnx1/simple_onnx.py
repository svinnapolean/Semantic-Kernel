# create_onnx_model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import onnxmltools
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Sample training data
X = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X, y)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([None, 1]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open("linear_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model created successfully!")