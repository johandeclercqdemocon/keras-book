# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch12.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers, ops
def first(e): return f"{type(e).__name__}: {str(e).splitlines()[0][:130]}"
n = lambda t: ops.convert_to_numpy(t)
def tryp(label, fn):
    try: print(label, fn())
    except Exception as e: print(label, first(e))
rng = np.random.default_rng(2); q = rng.normal(size=(1, 5, 2, 8)).astype("float32")
D = ops.dot_product_attention
tryp("3-D query:", lambda: tuple(D(q[:, :, 0], q[:, :, 0], q[:, :, 0]).shape))
mask = np.ones((1, 2, 5, 5), bool); mask[..., 0] = False
bias = np.where(mask, 0.0, -1e9).astype("float32")
a, b = n(D(q, q, q, mask=mask)), n(D(q, q, q, bias=bias)); print("mask equals -1e9 bias:", bool(np.allclose(a, b, atol=1e-5)))
def wmax(**kw):
    s = np.einsum("bthd,bshd->bhts", q, q) * (kw.get("scale", 1/np.sqrt(8)))
    w = np.exp(s - s.max(-1, keepdims=True)); w /= w.sum(-1, keepdims=True); return round(float(w.max(-1).mean()), 3)
print("mean largest weight: default scale", wmax(), " scale=1.0", wmax(scale=1.0))
print("scale=1.0 changes output:", not np.allclose(n(D(q, q, q)), n(D(q, q, q, scale=1.0))))
q2 = q.copy(); q2[:, 4] += 1
c1, c2 = n(D(q, q, q, is_causal=True)), n(D(q2, q2, q2, is_causal=True)); print("is_causal: positions 0-3 unchanged:", bool(np.allclose(c1[:, :4], c2[:, :4])))
tryp("flash_attention=True on CPU:", lambda: tuple(D(q, q, q, flash_attention=True).shape))
tryp("attn_logits_soft_cap=1.0:", lambda: bool(np.allclose(n(D(q, q, q, attn_logits_soft_cap=1.0)), n(D(q, q, q)))))
# MHA
keras.utils.set_random_seed(0)
x = np.random.default_rng(1).normal(size=(1, 6, 16)).astype("float32")
def mha(**kw): keras.utils.set_random_seed(0); return layers.MultiHeadAttention(**{"num_heads": 2, "key_dim": 8, **kw})
m = mha(); m(x, x); print("MHA 2x8 on 16: params", m.count_params())
m = mha(key_dim=3); print("key_dim 3: output", tuple(m(x, x).shape), "params", m.count_params())
m = mha(value_dim=4); m(x, x); print("value_dim 4: params", m.count_params())
print("output_shape 4:", tuple(mha(output_shape=4)(x, x).shape))
m = mha(use_gate=True); m(x, x); print("use_gate params:", m.count_params())
m = mha(use_bias=False); m(x, x); print("use_bias False params:", m.count_params())
tryp("flash_attention True ctor+call:", lambda: tuple(mha(flash_attention=True)(x, x).shape))
x2 = x.copy(); x2[:, 4] += 1.0
for sw in (None, 1, 2):
    try:
        m = mha(sliding_window=sw) if sw else mha(); a, b = n(m(x, x)), n(m(x2, x2))
        print("sliding_window", sw, "positions moved by token 4:", [int(i) for i in range(6) if not np.allclose(a[:, i], b[:, i], atol=1e-6)])
    except Exception as e: print("sliding_window", sw, first(e))
m = mha(dropout=0.5); print("dropout: inference stable", bool(np.allclose(n(m(x, x)), n(m(x, x)))), " training differs", not np.allclose(n(m(x, x, training=True)), n(m(x, x, training=True))))
m = mha(); print("key None equals key=value:", bool(np.allclose(n(m(x, x)), n(m(x, x, key=x)))))
vm = np.array([[True, True, True, True, False, False]])
_, s = m(x, x, value_mask=vm, return_attention_scores=True); print("value_mask: weight on last 2 keys", float(n(s)[..., 4:].max()))
am = np.ones((1, 6, 6), bool); am[:, :, 0] = False
_, s = m(x, x, attention_mask=am, return_attention_scores=True); print("attention_mask: weight on key 0", float(n(s)[..., 0].max()))
k = np.random.default_rng(3).normal(size=(1, 9, 16)).astype("float32"); o, s = m(x, k, return_attention_scores=True); print("query 6, value 9: out", tuple(o.shape), "scores", tuple(s.shape))
tryp("query_mask accepted:", lambda: tuple(m(x, x, query_mask=vm).shape))
# LayerNormalization
r = np.array([[0.0, 0.0632, -0.0632, 0.0]], "float32"); print("row variance", float(r.var()))
o = n(layers.LayerNormalization()(r)); print("LN default eps: output std", round(float(o.std()), 3))
o = n(layers.LayerNormalization(epsilon=1e-6)(r)); print("LN eps 1e-6: output std", round(float(o.std()), 3))
for kw in ({}, {"center": False}, {"scale": False}):
    l = layers.LayerNormalization(**kw); l(x); print("LN", kw, "params", l.count_params())
l = layers.LayerNormalization(axis=[1, 2]); l(x); print("LN axis [1,2] params", l.count_params(), "gamma shape", tuple(l.gamma.shape))
