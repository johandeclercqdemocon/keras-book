# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch10.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
n = lambda t: ops.convert_to_numpy(t)
keras.utils.set_random_seed(0)
i = keras.Input((8, 8, 1))
def shp(l, inp=i):
    try: o = l(inp); return f"{tuple(o.shape)} params {l.count_params()}"
    except Exception as e: return first(e)
print("data_format channels_first on (8,8,1):", shp(layers.Conv2D(16, 3, data_format="channels_first")))
print("dilation_rate 2:", shp(layers.Conv2D(16, 3, dilation_rate=2)))
print("groups 2 on 1 channel:", shp(layers.Conv2D(16, 3, groups=2)))
i4 = keras.Input((8, 8, 4))
print("groups 1 on 4 ch:", shp(layers.Conv2D(16, 3), i4), " groups 2 on 4 ch:", shp(layers.Conv2D(16, 3, groups=2), i4))
try: print("padding causal:", shp(layers.Conv2D(16, 3, padding="causal")))
except Exception as e: print("padding causal:", first(e))
print("strides 2 valid:", shp(layers.Conv2D(16, 3, strides=2)))
try: print("dilation 2 + strides 2:", shp(layers.Conv2D(16, 3, strides=2, dilation_rate=2)))
except Exception as e: print("dil+stride:", first(e))
print("kernel_size 9 valid on 8x8:", shp(layers.Conv2D(16, 9)))
i7 = keras.Input((7, 7, 16))
print("maxpool 2 on 7x7:", shp(layers.MaxPooling2D(2), i7), " strides 1:", shp(layers.MaxPooling2D(2, strides=1), i7), " same:", shp(layers.MaxPooling2D(2, padding="same"), i7))
print("maxpool 3 on 8x8 default strides:", shp(layers.MaxPooling2D(3), keras.Input((8, 8, 1))))
print("global keepdims:", shp(layers.GlobalMaxPooling2D(keepdims=True), i7), " default:", shp(layers.GlobalMaxPooling2D(), i7))
print("global channels_first on (7,7,16):", shp(layers.GlobalMaxPooling2D(data_format="channels_first"), i7))
from examples.digits import load
_, x_test, _, _ = load(images=True)
img = x_test[:200]
for kw in ({"fill_mode": "reflect"}, {"fill_mode": "constant"}, {"fill_mode": "constant", "fill_value": 0.5}, {"fill_mode": "nearest"}):
    keras.utils.set_random_seed(0); t = layers.RandomTranslation(0.25, 0.25, **kw)
    o = n(t(img, training=True)); print("fill", kw, "mean ink", round(float(o.mean()),4), "orig", round(float(img.mean()),4))
keras.utils.set_random_seed(0); t = layers.RandomTranslation(0.125, 0.125, fill_mode="constant"); o = n(t(img, training=True))
print("0.125 bilinear: share of non-grid values", round(float((~np.isin(np.round(o*16,4), np.arange(17))).mean()),4))
keras.utils.set_random_seed(0); t = layers.RandomTranslation(0.125, 0.125, fill_mode="constant", interpolation="nearest"); o = n(t(img, training=True))
print("0.125 nearest: share of non-grid values", round(float((~np.isin(np.round(o*16,4), np.arange(17))).mean()),4))
print("inference unchanged:", bool(np.array_equal(n(layers.RandomTranslation(0.25, 0.25)(img)), img)))
try: layers.RandomTranslation(1.5, 0.1); print("factor 1.5 accepted")
except Exception as e: print("factor 1.5:", first(e))
keras.utils.set_random_seed(0)
a = n(layers.RandomTranslation(0.25, 0.25, seed=3)(img, training=True)); b = n(layers.RandomTranslation(0.25, 0.25, seed=3)(img, training=True)); print("same seed equal:", bool(np.array_equal(a, b)))
print("pixel max of digits images:", float(img.max()))
