# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch05b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np, tensorflow as tf
from keras import layers, ops
from examples.digits import load
x_train, x_test, y_train, y_test = load(scale=False)
n = layers.Normalization(); n.adapt(tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32))
print("adapt on (x, y): mean equals x mean:", bool(np.allclose(ops.convert_to_numpy(n.mean)[0], x_train.mean(0), atol=1e-4)))
