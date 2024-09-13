import tensorflow as tf

# List all physical devices
print("Physical devices: ", tf.config.list_physical_devices())

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Print detailed information about GPU(s)
for device in tf.config.list_physical_devices('GPU'):
    print(device)