# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch06b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers
from examples.digits import load
x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((64,)), layers.Dense(10)])
T = keras.metrics.SparseTopKCategoricalAccuracy
try:
    m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=[T(k=1), T(k=3)])
    h = m.fit(x_train, y_train, epochs=1, verbose=0); print("keys:", sorted(h.history))
except Exception as e: print(type(e).__name__, str(e).splitlines()[0][:150])
