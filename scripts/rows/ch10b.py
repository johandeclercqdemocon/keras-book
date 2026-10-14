# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch10b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
from examples.digits import load
_, x_test, _, _ = load(images=True); img = x_test[:200]
n = lambda t: ops.convert_to_numpy(t)
t = layers.RandomTranslation(0.25, 0.25)
print("direct call, no training arg, changed:", not np.array_equal(n(t(img)), img))
print("direct call, training=False, changed:", not np.array_equal(n(t(img, training=False)), img))
m = keras.Sequential([keras.Input((8, 8, 1)), layers.RandomTranslation(0.25, 0.25)])
print("model.predict changed:", not np.allclose(m.predict(img, verbose=0), img))
print("model(x) no training arg changed:", not np.allclose(n(m(img)), img))
