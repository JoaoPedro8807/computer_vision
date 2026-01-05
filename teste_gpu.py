import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f"Número de GPUs: {len(gpus)}")