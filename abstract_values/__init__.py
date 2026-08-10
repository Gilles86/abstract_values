import os

# The custom encoding models (encoding_models/models.py) are TF-native
# (tf.where, tf.debugging asserts, tf-tensor constants) and crash under a
# JAX keras backend with TracerArrayConversionError during gradient descent.
# ~/.keras/keras.json on the cluster selects jax globally (for other
# projects), so pin TF here — env var beats keras.json, setdefault still
# honors an explicit KERAS_BACKEND override.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
