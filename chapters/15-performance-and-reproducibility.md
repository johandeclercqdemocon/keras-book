# Chapter 15 — Performance and reproducibility

> **Build** — a test that counts how many different results the same training gives, a measurement of how much one setting narrows them, a rule for layer order that decides the start weights, and four measured facts about `steps_per_execution`, `jit_compile`, mixed precision and the back end.
>
> **Theory kept** — a result is reproducible when the same inputs give the same bits. Seeds are necessary and not sufficient. The order of floating-point operations must also be fixed.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0, CPU only. The code is New.
>
> **Run this first** — `uv run python -m examples.ch15_performance_and_reproducibility`, which is [`examples/ch15_performance_and_reproducibility.py`](../examples/ch15_performance_and_reproducibility.py).

This book printed numbers to four digits and compared them exactly. That is only fair if the same code gives the same numbers on every run. This chapter shows when it does not. It measures one setting that makes runs closer, and it says what that setting did not fix.

Speed is the second topic. This book ran on a CPU with no accelerator, so the chapter does not report timings. It reports what each speed option does to the result, and what it did not do on this machine.

## The same code, many results

Setting a seed is not enough. The fingerprint of a run in this section is a hash of the bytes of all weights, and the last training loss. The program is a small CNN that trains for 40 epochs. It is the module [`examples/determinism_probe.py`](../examples/determinism_probe.py).

The next block runs the probe in 8 separate processes at the same time. It does this once with the TensorFlow setting `TF_DETERMINISTIC_OPS` on and once with it off. It prints the number of distinct weight hashes and the spread of the final loss. The output depends on the machine and on the load, so the checker does not compare it. The text block holds what the run that wrote this chapter printed.

```python
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

def run_probe(flag):
    env = {**os.environ, "TF_DETERMINISTIC_OPS": flag}
    result = subprocess.run([sys.executable, "-m", "examples.determinism_probe"],
                            capture_output=True, text=True, env=env)
    digest, loss = result.stdout.strip().splitlines()[-1].split()
    return digest, float(loss)

for flag, label in (("1", "on "), ("0", "off")):
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(run_probe, [flag] * 8))
    losses = [loss for _, loss in runs]
    print(f"flag {label}: 8 runs, distinct weight hashes {len({d for d, _ in runs})}, "
          f"loss spread {max(losses) - min(losses):.2e}")
```

```text
# output varies
flag on : 8 runs, distinct weight hashes 2, loss spread 1.00e-08
flag off: 8 runs, distinct weight hashes 2, loss spread 1.16e-05
```

Read the result with care. The setting did not make the run bit-exact on this machine: 8 runs gave 2 different sets of weights with it on, and 2 with it off. The setting did make the difference small.

The spread of the final loss fell from 1.16e-05 to 1.00e-08, which is about 1,000 times. The chapter did not find the source of the remaining difference. A separate test with one TensorFlow thread also gave 2 hashes in 8 runs.

What this means for you: compare two runs with a tolerance and not with `==`. The checker of this book compares printed values that are rounded to 3 or 4 decimals, and a spread of 1e-8 is far below that.

The next block checks the part that is firm. It runs the probe 4 times with the setting on and tests that the final losses agree to 1e-6.

```python
import os
import subprocess
import sys

env = {**os.environ, "TF_DETERMINISTIC_OPS": "1"}
losses = []
for _ in range(4):
    result = subprocess.run([sys.executable, "-m", "examples.determinism_probe"],
                            capture_output=True, text=True, env=env)
    losses.append(float(result.stdout.strip().splitlines()[-1].split()[1]))
print("runs:", len(losses))
print("losses agree to 1e-6:", max(losses) - min(losses) < 1e-6)
```

```text
runs: 4
losses agree to 1e-6: True
```

## How it works

A floating-point sum depends on the order of the additions. In 32-bit numbers, `(a + b) + c` can differ from `a + (b + c)`. A multi-threaded reduction adds the parts in the order in which the threads finish, and that order can change from run to run. The `TF_DETERMINISTIC_OPS` setting asks TensorFlow to use implementations with a fixed order where they exist. This book measured only the effect of the setting, shown in the previous section.

The next block shows the first fact. It does not show the cause of the differences in training, which this book did not observe directly.

```python
import numpy as np

a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
print("(a + b) + c =", (a + b) + c)
print("a + (b + c) =", a + (b + c))
```

```text
(a + b) + c = 1.0
a + (b + c) = 0.0
```

The two orders give 1.0 and 0.0. The invariant is this: a result is bit-reproducible only if every random number and every order of addition is fixed. The seed fixes the random numbers. The setting fixes the order of the operations that it covers. The previous section shows that this was not enough for exact equality on this machine.

The setting has a second effect that surprised the author. With determinism on, a `fit` call without a seed failed in several blocks of this book. The message was: `Random ops require a seed to be set when determinism is enabled`. The shuffle of the training data is the random operation.

A script that worked before can fail with this message. The fix is a call to `keras.utils.set_random_seed` at the top. Exercise 2 reproduces the error.

## The order of construction decides the start weights

The start weights of a layer come from a random generator that Keras advances each time it makes an initializer. So the order in which you create layers decides which numbers each layer gets. The two lines below build the same network. They do not give the same weights.

```python
import keras
import numpy as np
from keras import layers

def nested():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8,))
    outputs = layers.Dense(3)(layers.Dense(5)(inputs))
    return keras.Model(inputs, outputs)

def in_order():
    keras.utils.set_random_seed(0)
    inputs = keras.Input((8,))
    hidden = layers.Dense(5)(inputs)
    return keras.Model(inputs, layers.Dense(3)(hidden))

first, second = nested(), in_order()
print("first layer kernels equal:", np.array_equal(first.get_weights()[0], second.get_weights()[0]))
```

```text
first layer kernels equal: False
```

In the nested line, Python creates the outer `Dense(3)` before it calls the inner expression, so the outer layer takes the first random numbers. The two models have the same shapes and different weights. When you compare two runs, build the layers in the same order, and write the order down.

## Four speed options, measured for their effect

The next block trains the same model with the default, with `steps_per_execution=4`, with `jit_compile=True`, and under a mixed-precision policy. It prints the final loss of 3 epochs and, for the first two, the largest difference of the weights.

```python
import warnings

import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def train(**compile_options):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.compile("adam", loss, **compile_options)
        history = model.fit(x_train, y_train, epochs=3, batch_size=32, shuffle=False, verbose=0).history
    flat = np.concatenate([w.ravel() for w in model.get_weights()])
    return flat, history["loss"][-1], [str(c.message)[:58] for c in caught]

base, base_loss, _ = train()
fused, fused_loss, _ = train(steps_per_execution=4)
jitted, jitted_loss, notes = train(jit_compile=True)
print("default             loss", round(base_loss, 4))
print("steps_per_execution loss", round(fused_loss, 4), " weight difference", float(np.abs(base - fused).max()))
print("jit_compile=True    loss", round(jitted_loss, 4), " weight difference", float(np.abs(base - jitted).max()))
print("jit message:", notes[0] if notes else None)

keras.mixed_precision.set_global_policy("mixed_float16")
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
first = model.layers[0]
print("mixed policy: compute", first.compute_dtype, " variables", first.variable_dtype,
      " output", model(x_test[:2]).dtype)
keras.utils.set_random_seed(0)
model.compile("adam", loss)
mixed_loss = model.fit(x_train, y_train, epochs=3, batch_size=32, shuffle=False, verbose=0).history["loss"][-1]
print("mixed precision loss", round(mixed_loss, 4), " optimizer:", type(model.optimizer).__name__)
keras.mixed_precision.set_global_policy("float32")
```

```text
default             loss 1.0979
steps_per_execution loss 1.0979  weight difference 0.0
jit_compile=True    loss 1.0979  weight difference 0.0
jit message: Model doesn't support `jit_compile=True`. Proceeding with 
mixed policy: compute float16  variables float32  output <dtype: 'float16'>
mixed precision loss 1.1083  optimizer: LossScaleOptimizer
```

Read the lines in order. `steps_per_execution=4` runs four steps in one call, and it gives the same weights: the difference is 0. The option changes the cost of the loop and not the result. `jit_compile=True` printed a message that the model does not support it. Keras ran without it, so the weights are equal for that reason, and this chapter did not measure any speed gain.

Mixed precision changed the result. The compute type is `float16`, the variables stay `float32`, and the optimizer is wrapped in a loss-scaling optimizer. The loss after 3 epochs is different from the default. The difference is the price of the 16-bit numbers of [Chapter 2](02-tensors-and-ops.md). Whether the speed pays for it depends on hardware that this book did not have.

## The other back ends

Keras 3 runs on JAX and on PyTorch. This book installed neither, so no result in it comes from them. The next block shows what Keras says when you ask.

```python
import os
import subprocess
import sys

for backend in ("jax", "torch"):
    result = subprocess.run([sys.executable, "-c", "import keras"],
                            capture_output=True, text=True,
                            env={**os.environ, "KERAS_BACKEND": backend})
    print(backend, "->", result.stderr.strip().splitlines()[-1])
```

```text
jax -> ModuleNotFoundError: No module named 'jax'
torch -> ModuleNotFoundError: No module named 'torch'
```

The statement "this code runs on every back end" is a property of the functions that the book used, and not a measurement. Run the blocks on your back end before you rely on it. A custom `train_step` with `tf.GradientTape` ([Chapter 9](09-custom-training.md)) will not run on another back end.

## A checklist

1. Call `keras.utils.set_random_seed` once, at the top.
2. Set `TF_DETERMINISTIC_OPS=1` to make runs closer, and compare them with a tolerance.
3. Build layers in a fixed order, and do not nest constructors that you compare.
4. Change one option at a time: the chapter showed an option that changes nothing and an option that changes the result.
5. State the hardware and the back end next to every number.

## Reference

```python
import keras
from examples.signatures import show

show(("Model.compile", keras.Model.compile),
     ("utils.set_random_seed", keras.utils.set_random_seed),
     ("mixed_precision.set_global_policy", keras.mixed_precision.set_global_policy))
```

```text
Model.compile(optimizer='rmsprop', loss=None, loss_weights=None, metrics=None,
    weighted_metrics=None, run_eagerly=False, steps_per_execution=1,
    jit_compile='auto', auto_scale_loss=True)
utils.set_random_seed(seed)
mixed_precision.set_global_policy(policy)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `compile(steps_per_execution)` | `1` | steps that run in one call | 4 gave the same weights as 1 |
| `compile(jit_compile)` | `'auto'` | compile the step with XLA | `True` printed that the model does not support it, on this machine |
| `compile(run_eagerly)` | `False` | run the step in Python | — |
| `set_random_seed(seed)` | required | seeds Python, NumPy and the back end | with determinism on, a missing seed is an error |
| `set_global_policy(policy)` | required | default compute and variable types | `'mixed_float16'` changed the loss |

## What you have now

You have a count of distinct results and a setting that narrows the differences. You also have two sources of difference that you can control: the order of layer construction and the seed. The cause of the remaining difference is open. The list of what this book did not measure is in Chapter 16: speed, and any back end except TensorFlow.

[Chapter 16](16-field-guide.md) collects the book into a table from symptom to cause. Exercises for this chapter are in [exercises/15.md](../exercises/15.md), and the worked solutions are in [exercises/solutions/15.md](../exercises/solutions/15.md).
