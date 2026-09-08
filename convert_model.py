import tensorflow as tf
import tensorflowjs as tfjs

print("Loading Keras model...")
# Loads the model you just trained
model = tf.keras.models.load_model('deepfake_cnn.h5')

print("Converting to TensorFlow.js format...")
# Bypasses the CLI and converts directly
tfjs.converters.save_keras_model(model, './models')

print("Success! Check your models folder.")