# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch08b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
x = np.ones((1, 8), dtype="float32")
print("rate 1.0 training:", ops.convert_to_numpy(layers.Dropout(1.0)(x, training=True)))
try: g = layers.GaussianNoise(2.0); print("stddev 2.0 accepted, std", round(float(ops.convert_to_numpy(g(np.zeros((1,100000),"float32"), training=True)).std()),3))
except Exception as e: print("stddev 2.0:", type(e).__name__, str(e)[:100])
