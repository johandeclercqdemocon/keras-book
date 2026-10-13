# Chapter 2 — Tensors and ops

> **Build** — a layer-normalization function written only with `keras.ops`, checked against NumPy, and a set of four measured rules for number types and random seeds.
>
> **Theory kept** — a tensor is an array that a back end owns. A function that uses only `keras.ops` runs on every back end.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0. The code is New.
>
> **Run this first** — `uv run python -m examples.ch02_tensors_and_ops`, which is [`examples/ch02_tensors_and_ops.py`](../examples/ch02_tensors_and_ops.py).

[Chapter 1](01-the-first-model.md) used layers and did not look at what flows between them. This chapter does. A layer takes a tensor and returns a tensor. You must know what a tensor is, which functions can work on it, and which number types and random numbers it uses.

## A tensor belongs to a back end

NumPy arrays belong to NumPy. The tensors in a Keras model belong to the back end. With the TensorFlow back end they are `tf.Tensor` objects. With the JAX back end they are JAX arrays.

The module `keras.ops` is the common interface. Its functions accept NumPy arrays and Python lists. They return tensors of the active back end.

```python
import keras
import numpy as np
from keras import ops

ones = ops.ones((2, 3))
print(type(ones).__name__, ones.dtype, ops.shape(ones))

array = np.arange(6, dtype="float32").reshape(2, 3)
tensor = ops.convert_to_tensor(array)
print(ops.is_tensor(array), ops.is_tensor(tensor))
print(type(ops.convert_to_numpy(tensor)).__name__)
```

```text
EagerTensor <dtype: 'float32'> (2, 3)
False True
ndarray
```

The first line shows the class of the tensor on this back end. The second line shows that a NumPy array is not a tensor. The third line shows the way back: `ops.convert_to_numpy` returns a NumPy array. Use it whenever you print, plot or compare a result.

## Write once with `keras.ops`

Most NumPy functions have a twin in `keras.ops`, with the same name and the same arguments. The next block writes layer normalization with `ops` only. Layer normalization subtracts the mean of each row and divides by the standard deviation of the row. A small constant `eps` keeps the division safe.

The block compares the result with the same formula in NumPy.

```python
import keras
import numpy as np
from keras import ops

def layer_norm(x, eps=1e-6):
    mean = ops.mean(x, axis=-1, keepdims=True)
    variance = ops.var(x, axis=-1, keepdims=True)
    return (x - mean) / ops.sqrt(variance + eps)

x = np.random.default_rng(0).normal(size=(4, 5)).astype("float32")
reference = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-6)
result = ops.convert_to_numpy(layer_norm(ops.convert_to_tensor(x)))

print("max difference below 1e-6:", bool(np.abs(result - reference).max() < 1e-6))
print("row means near zero:", bool(np.abs(result.mean(-1)).max() < 1e-6))
print("row variances near one:", bool(np.abs(result.var(-1) - 1).max() < 1e-4))
```

```text
max difference below 1e-6: True
row means near zero: True
row variances near one: True
```

The function `layer_norm` contains no back end code. This book ran it on TensorFlow only. The three functions `ops.mean`, `ops.var` and `ops.sqrt` are part of `keras.ops`, so the same code is valid on the other back ends. Chapter 15 says what that claim is worth. A function that calls `tf.reduce_mean` is valid on one back end only. This is the reason to prefer `keras.ops` inside a custom layer ([Chapter 3](03-layers-and-weights.md)).

## Number types decide what a number is

A tensor has a dtype. The default is `float32`, and `keras.config.floatx()` returns it. Two other types appear often: `float16`, which has less precision, and the 8-bit integers, which wrap.

```python
import keras
import numpy as np
from keras import ops

half = ops.cast(0.1, "float16")
stored = float(ops.convert_to_numpy(half))
print("0.1 as float16:", stored, f"(error {0.1 - stored:.2e})")

total = ops.add(ops.cast([200], "uint8"), ops.cast([100], "uint8"))
print("uint8 200 + 100:", ops.convert_to_numpy(total))

big = ops.cast([70000.0], "float16")
print("float16 70000:", ops.convert_to_numpy(big), "largest finite:", np.finfo("float16").max)
```

```text
0.1 as float16: 0.0999755859375 (error 2.44e-05)
uint8 200 + 100: [44]
float16 70000: [inf] largest finite: 6.55e+04
```

The value 0.1 becomes 0.0999755859375 in 16 bits. The error is 2.44e-05. A sum of `uint8` values wraps: 200 + 100 is 300, and 300 modulo 256 is 44. No error is raised. The `float16` type has a largest finite value of 65504, so 70000 becomes infinity.

These three results are the reasons for care when you train with 16-bit numbers or quantize a model to 8 bits. [Chapter 15](15-performance-and-reproducibility.md) measures what mixed precision costs and what it saves.

## Random numbers need a seed policy

Keras has its own random functions in `keras.random`. They use the seed that you give them, or a global seed that you set with `keras.utils.set_random_seed`. That call sets the seed for Python, NumPy and the back end at the same time. The block below checks this claim in its first lines.

The two seed styles behave in different ways. The block below shows the three cases that surprise people. It also confirms that `Dropout` owns a seed generator.

```python
import keras
import numpy as np
from keras import ops

import random

keras.utils.set_random_seed(5)
python_a, numpy_a = random.random(), np.random.rand()
keras.utils.set_random_seed(5)
python_b, numpy_b = random.random(), np.random.rand()
print("python and numpy restart together:", python_a == python_b and numpy_a == numpy_b)
print("Dropout owns a seed generator:", hasattr(keras.layers.Dropout(0.5), "seed_generator"))

def same(a, b):
    return bool(np.array_equal(ops.convert_to_numpy(a), ops.convert_to_numpy(b)))

keras.utils.set_random_seed(1)
first = keras.random.normal((3,))
keras.utils.set_random_seed(1)
second = keras.random.normal((3,))
print("global seed set twice, same values:", same(first, second))

generator = keras.random.SeedGenerator(7)
draw_a = keras.random.normal((3,), seed=generator)
draw_b = keras.random.normal((3,), seed=generator)
print("SeedGenerator, two draws equal:", same(draw_a, draw_b))

draw_c = keras.random.normal((3,), seed=7)
draw_d = keras.random.normal((3,), seed=7)
print("integer seed, two draws equal:", same(draw_c, draw_d))
```

```text
python and numpy restart together: True
Dropout owns a seed generator: True
global seed set twice, same values: True
SeedGenerator, two draws equal: False
integer seed, two draws equal: True
```

Setting the global seed again restarts the stream, so the two values are equal. A `SeedGenerator` is a stateful object. It advances on each use, so two draws differ and the whole sequence is still repeatable. An integer seed is stateless. The same integer gives the same numbers on every call.

The third result is the trap. A custom layer that calls `keras.random.normal(shape, seed=7)` inside `call` adds the same noise to every batch. Use a `SeedGenerator`, which is what the built-in `Dropout` layer does.

## Four rules

The chapter reduces to four rules. Each rule has a measurement above.

1. Convert with `ops.convert_to_numpy` before you print or plot a tensor.
2. Write custom code with `keras.ops` so that it runs on every back end.
3. Name the dtype when the range or the precision matters. Do not rely on the default.
4. Use `set_random_seed` once at the start of a script. Use a `SeedGenerator` inside a layer.

## How it works

A `keras.ops` function does not compute anything itself. It looks at the active back end, calls the matching function of that back end and returns a back end tensor. This is why the same code runs on three back ends, and why the type of the result is not a NumPy array.

The random functions need one more idea. A back end random function is a pure function of a seed and a shape. The same seed gives the same numbers.

For a program to draw different numbers on each call, something must change the seed between calls. In Keras, a `SeedGenerator` holds that state. Its state is a variable of two numbers: the seed and a counter.

The next block shows the state before and after two draws, and the difference between a generator and an integer seed.

```python
import keras
from keras import ops

generator = keras.random.SeedGenerator(7)
print("state at start      :", ops.convert_to_numpy(generator.state).tolist())
first = keras.random.normal((2,), seed=generator)
print("state after one draw:", ops.convert_to_numpy(generator.state).tolist())
second = keras.random.normal((2,), seed=generator)
print("state after two     :", ops.convert_to_numpy(generator.state).tolist())

again = keras.random.SeedGenerator(7)
repeat = keras.random.normal((2,), seed=again)
print("new generator with seed 7 repeats the first draw:",
      bool((ops.convert_to_numpy(repeat) == ops.convert_to_numpy(first)).all()))
print("a draw does not change an integer seed:", keras.random.normal((1,), seed=7).shape == (1,))
```

```text
state at start      : [7, 0]
state after one draw: [7, 1]
state after two     : [7, 2]
new generator with seed 7 repeats the first draw: True
a draw does not change an integer seed: True
```

The counter goes up by one on each draw, and the seed stays at 7. The new generator starts at counter 0, so it repeats the first draw.

The invariant is: *the numbers are a function of the seed and the counter.* A generator moves the counter, and an integer seed never moves it. This is the cause of the trap in this chapter. The call `seed=7` inside a layer gives the same noise on every call.

## Reference

```python
import keras
from keras import ops
from examples.signatures import show

show(("ops.convert_to_tensor", ops.convert_to_tensor), ("ops.cast", ops.cast),
     ("ops.convert_to_numpy", ops.convert_to_numpy), ("random.normal", keras.random.normal),
     keras.random.SeedGenerator, ("utils.set_random_seed", keras.utils.set_random_seed))
```

```text
ops.convert_to_tensor(x, dtype=None, sparse=None, ragged=None)
ops.cast(x, dtype)
ops.convert_to_numpy(x)
random.normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None)
SeedGenerator(seed=None, name=None, **kwargs)
utils.set_random_seed(seed)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `convert_to_tensor(x)` | required | NumPy array or list to a back end tensor | — |
| `convert_to_tensor(dtype)` | `None` | type of the result | — |
| `cast(x, dtype)` | required | change the type | `uint8` 200 + 100 gives 44, `float16` 70000 gives `inf` (Chapter 2) |
| `convert_to_numpy(x)` | required | tensor to NumPy array | print and plot need it |
| `random.normal(shape)` | required | tensor of random numbers | — |
| `random.normal(seed)` | `None` | `None`: the global generator that `set_random_seed` sets; integer: stateless; `SeedGenerator`: stateful | an integer seed gives equal draws on every call |
| `SeedGenerator(seed)` | `None` | start state of a stream of seeds | two draws differ, a new generator repeats the stream |
| `set_random_seed(seed)` | required | seed Python, NumPy and the back end together | call it once at the start of a script |

## What you have now

You have a layer-normalization function that runs on any Keras back end and that agrees with NumPy. You also have the four rules above.

[Chapter 3](03-layers-and-weights.md) puts such a function inside a layer and shows where the layer keeps its weights. Exercises for this chapter are in [exercises/02.md](../exercises/02.md), and the worked solutions are in [exercises/solutions/02.md](../exercises/solutions/02.md).
