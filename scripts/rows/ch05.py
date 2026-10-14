# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch05.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, warnings, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
from examples.digits import load
x_train, x_test, y_train, y_test = load(scale=False)
for ax in (-1, None):
    n = layers.Normalization(axis=ax); n.adapt(x_train); print("axis", ax, "mean shape", tuple(n.mean.shape))
try: layers.Normalization(mean=3.0)
except Exception as e: print("mean only:", first(e))
n = layers.Normalization(mean=2.0, variance=4.0); print("fixed mean 2 var 4 on [4.]:", ops.convert_to_numpy(n(np.array([[4.0]], dtype="float32"))))
n = layers.Normalization(); n.adapt(x_train); z = n(x_test[:3])
inv = layers.Normalization(invert=True); inv.adapt(x_train); back = ops.convert_to_numpy(inv(z))
print("invert round trip max error:", float(np.abs(back - x_test[:3]).max()))
a = np.array([[0.0], [2.0]], dtype="float32"); b = np.array([[10.0], [12.0]], dtype="float32")
n = layers.Normalization(); n.adapt(a); n.adapt(b); print("adapt twice mean:", float(ops.convert_to_numpy(n.mean)[0][0]))
import tensorflow as tf
ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
try: n = layers.Normalization(); n.adapt(ds); print("adapt on (x, y) dataset: accepted, mean shape", tuple(n.mean.shape))
except Exception as e: print("adapt on (x,y) dataset:", first(e))
n = layers.Normalization(); n.adapt(ds.map(lambda x, y: x)); print("adapt on x-only dataset ok, mean shape", tuple(n.mean.shape))
class NoSuper(keras.utils.PyDataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return 2
    def __getitem__(self, i): return self.x[i*32:(i+1)*32], self.y[i*32:(i+1)*32]
keras.utils.set_random_seed(0)
m = keras.Sequential([keras.Input((64,)), layers.Dense(10)]); m.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    try: m.fit(NoSuper(x_train, y_train), epochs=1, verbose=0); print("PyDataset without super().__init__: fit ok, warnings", [str(x.message)[:90] for x in w])
    except Exception as e: print("PyDataset no super:", first(e))
try: keras.utils.PyDataset(foo=1)
except Exception as e: print("PyDataset foo:", first(e))
print("to_categorical [0,2]:", keras.utils.to_categorical([0, 2]).shape, " num_classes=10:", keras.utils.to_categorical([0, 2], num_classes=10).shape)
try: print("to_categorical [0,5] num_classes=3:", keras.utils.to_categorical([0, 5], num_classes=3))
except Exception as e: print("to_categorical too small:", first(e))
print("to_categorical dtype:", keras.utils.to_categorical([1]).dtype)
try: print("to_categorical [-1]:", keras.utils.to_categorical([-1, 1], num_classes=3))
except Exception as e: print("neg:", first(e))
