import tensorflow as tf
from gcpds.DataSet.infrared_thermal_feet import InfraredThermalFeet
"""
def test_tensorflow_gpu():
    # Check for available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("No GPU found. Please ensure that TensorFlow is installed with GPU support.")
    else:
        print(f"GPUs found: {gpus}")

# Run the test function
test_tensorflow_gpu()
"""

dataset = InfraredThermalFeet()

x, y, z = dataset()

print(len(x), len(y), len(z))