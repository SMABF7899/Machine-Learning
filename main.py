import numpy as np
import scipy
import pandas
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices())
X = np.arange(1, 101, step=0.1)
X = tf.cast(tf.constant(X), dtype=tf. float32)
