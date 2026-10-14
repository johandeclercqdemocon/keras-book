# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch11b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, warnings, keras, numpy as np
from keras import layers
warnings.simplefilter("ignore")
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
try: keras.Sequential([keras.Input((5, 16)), layers.GRU(8), layers.GRU(4)]); print("stacked GRU without return_sequences: accepted")
except Exception as e: print("stacked GRU:", first(e))
a = layers.GRU(32, reset_after=True); a.build((None, 5, 16)); b = layers.GRU(32, reset_after=False); b.build((None, 5, 16))
try: b.set_weights(a.get_weights()); print("set_weights across forms: accepted")
except Exception as e: print("set_weights across forms:", first(e))
e = layers.Embedding(10, 4, input_length=5); print("input_length=5, input of 3 ids:", tuple(e(np.array([[1, 2, 3]])).shape))
