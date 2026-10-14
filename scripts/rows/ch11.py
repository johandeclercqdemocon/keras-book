# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch11.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
n = lambda t: ops.convert_to_numpy(t)
def tryp(label, fn):
    try: print(label, fn())
    except Exception as e: print(label, first(e))
keras.utils.set_random_seed(0)
e = layers.Embedding(10, 4)
tryp("Embedding id 10 of 10:", lambda: n(e(np.array([[10]]))))
tryp("Embedding id -1:", lambda: n(e(np.array([[-1]]))))
w = n(layers.Embedding(1000, 16)(np.arange(1000)[None]) ) if False else None
ee = layers.Embedding(1000, 16); ee.build((None,)); v = n(ee.embeddings); print("uniform init min/max:", round(float(v.min()),4), round(float(v.max()),4))
tryp("Embedding weights=[ones]:", lambda: n(layers.Embedding(3, 2, weights=[np.ones((3, 2))])(np.array([[2]]))).tolist())
tryp("Embedding input_length=5:", lambda: layers.Embedding(10, 4, input_length=5))
em = layers.Embedding(10, 4, mask_zero=True); print("mask:", n(em.compute_mask(np.array([[3, 0, 5]]))).tolist())
# GRU
x = np.random.default_rng(0).normal(size=(2, 5, 16)).astype("float32")
def gru(**kw): keras.utils.set_random_seed(0); return layers.GRU(32, **kw)
print("return_sequences:", tuple(gru(return_sequences=True)(x).shape), "default:", tuple(gru()(x).shape))
o = gru(return_state=True)(x); print("return_state: list of", len(o), "output == state:", bool(np.allclose(n(o[0]), n(o[1]))))
a = n(gru()(x)); b = n(gru(go_backwards=True)(x)); c = n(gru()(x[:, ::-1])); print("go_backwards differs from forward:", not np.allclose(a, b), " equals forward on reversed input:", bool(np.allclose(b, c, atol=1e-6)))
s1 = n(gru(return_sequences=True, go_backwards=True)(x)); s2 = n(gru(return_sequences=True)(x[:, ::-1])); print("go_backwards sequences in reversed time order:", bool(np.allclose(s1, s2, atol=1e-6)))
for ra in (True, False):
    g = gru(reset_after=ra); g(x); print("reset_after", ra, "params", g.count_params())
tryp("stateful without fixed batch:", lambda: keras.Sequential([keras.Input((5, 16)), layers.GRU(4, stateful=True)])(x).shape)
st = keras.Sequential([keras.Input((5, 16), batch_size=2), layers.GRU(4, stateful=True)])
r1 = n(st(x)); r2 = n(st(x)); print("stateful: second call differs:", not np.allclose(r1, r2))
tryp("use_cudnn=True on CPU:", lambda: tuple(gru(use_cudnn=True)(x).shape))
g = gru(dropout=0.5); print("dropout inference deterministic:", bool(np.allclose(n(g(x)), n(g(x)))), " training differs:", not np.allclose(n(g(x, training=True)), n(g(x, training=True))))
tryp("GRU activation relu ok:", lambda: tuple(gru(activation="relu")(x).shape))
# Conv1D
x1 = np.random.default_rng(1).normal(size=(1, 10, 3)).astype("float32")
keras.utils.set_random_seed(0); c = layers.Conv1D(2, 3, padding="causal"); y1 = n(c(x1)); x2 = x1.copy(); x2[0, 7] += 5; y2 = n(c(x2))
print("causal: shape", y1.shape, "outputs before step 7 unchanged:", bool(np.allclose(y1[0, :7], y2[0, :7])), "step 7 changed:", not np.allclose(y1[0, 7], y2[0, 7]))
keras.utils.set_random_seed(0); c = layers.Conv1D(2, 3, padding="same"); y1 = n(c(x1)); y2 = n(c(x2)); print("same: output at step 6 changed by step 7:", not np.allclose(y1[0, 6], y2[0, 6]))
print("valid len:", tuple(layers.Conv1D(2, 3)(x1).shape), " dilation 2:", tuple(layers.Conv1D(2, 3, dilation_rate=2)(x1).shape), " strides 2:", tuple(layers.Conv1D(2, 3, strides=2)(x1).shape))
# TextVectorization
TV = layers.TextVectorization; data = ["the cat sat", "the dog sat on the mat", "a cat"]
t = TV(max_tokens=5); t.adapt(data); print("max_tokens=5 vocab:", t.get_vocabulary())
t = TV(); t.adapt(["The Cat!"]); print("standardize default 'The Cat!':", t.get_vocabulary())
t = TV(standardize=None); t.adapt(["The Cat!"]); print("standardize None:", t.get_vocabulary())
t = TV(split="character"); t.adapt(["abc"]); print("split character:", t.get_vocabulary())
t = TV(ngrams=2); t.adapt(data); print("ngrams 2 vocab size:", len(t.get_vocabulary()), "vs", len((lambda u: (u.adapt(data), u.get_vocabulary())[1])(TV())))
t = TV(output_mode="multi_hot"); t.adapt(data); print("multi_hot shape:", tuple(t(["a cat"]).shape))
t = TV(); t.adapt(data); print("no output_sequence_length: batch lengths", tuple(t(["a cat"]).shape), tuple(t(["the dog sat on the mat"]).shape))
t = TV(vocabulary=["cat", "dog"]); print("vocabulary given:", t.get_vocabulary(), n(t(["dog cat bird"])).tolist())
t = TV(ragged=True); t.adapt(data); print("ragged type:", type(t(["a cat", "the dog sat"])).__name__)
tryp("output_sequence_length with multi_hot:", lambda: TV(output_mode="multi_hot", output_sequence_length=4))
