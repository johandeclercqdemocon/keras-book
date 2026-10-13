# Chapter 1 — The first model

> **Build** — a model with one weight and one bias that learns the line y = 2x + 1, and a small table that shows how the optimizer and the learning rate decide whether it learns at all.
>
> **Theory kept** — a model is a function with weights. Training changes the weights to make a number, the loss, smaller.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0, CPU only. The code is New.
>
> **Run this first** — `uv run python -m examples.ch01_the_first_model`, which is [`examples/ch01_the_first_model.py`](../examples/ch01_the_first_model.py).

Keras is a library to build neural networks and to train them. You describe a model with layers. Keras does the rest: it creates the weights, computes the gradients and updates the weights. This book teaches Keras 3, the current major version.

The aim of this chapter is small. You will build the smallest model that can learn something, and you will read every number that Keras prints about it. Later chapters reuse each call from this chapter.

## What Keras 3 is

Keras 3 is a front end. The same Keras code can run on three back ends: TensorFlow, JAX and PyTorch. The back end does the tensor arithmetic. Keras defines the layers, the models, the losses and the training loop.

This book runs on the TensorFlow back end, on a CPU. Every number in the book comes from that setup. The first block prints the back end and the default number type.

```python
import keras

print(keras.backend.backend(), keras.config.floatx(), keras.config.image_data_format())
```

```text
tensorflow float32 channels_last
```

The three values are the back end, the default floating-point type and the default layout of image tensors. All three are settings. The back end comes from the environment variable `KERAS_BACKEND`, and you must set it before you import Keras. [Chapter 15](15-performance-and-reproducibility.md) returns to the back ends and says what this book did not test on the other two.

## The smallest model

The task is to learn a straight line. The data are 64 points on y = 2x + 1, with x between -1 and 1. A model with one weight and one bias can represent this line exactly. Training must find the values 2 and 1.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
y = 2 * x + 1

model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse")
history = model.fit(x, y, epochs=100, batch_size=16, verbose=0)

kernel, bias = model.layers[0].get_weights()
print("first epoch loss:", round(history.history["loss"][0], 4))
print("last epoch loss: ", round(history.history["loss"][-1], 6))
print("weight:", kernel.ravel(), "bias:", bias)
print("prediction at x=3:", model.predict(np.array([[3.0]], dtype="float32"), verbose=0).ravel())
```

```text
first epoch loss: 1.8592
last epoch loss:  0.0
weight: [2.] bias: [1.]
prediction at x=3: [7.]
```

The model found the line. The weight is 2, the bias is 1, and the prediction at x = 3 is 7. This is a fit to noise-free data, so the loss can reach zero.

## The five calls

Every Keras program in this book uses the same five calls. The block above contains all of them.

1. `keras.Input((1,))` declares the shape of one sample. The batch size is not part of it.
2. `keras.layers.Dense(1)` is a layer. It computes `x @ kernel + bias` with one output unit.
3. `keras.Sequential([...])` stacks the layers into a model.
4. `model.compile(optimizer=..., loss=...)` chooses how to learn. It does not change the weights.
5. `model.fit(x, y, ...)` runs the training loop and returns a `History` object.

Two more calls read the result. `model.evaluate(x, y)` returns the loss on data. `model.predict(x)` returns the output of the model. Both calls work on batches, so the input must have a batch axis as its first axis.

## Read the summary

`model.summary()` prints a table of the layers and a count of the parameters. The next block builds the same model and prints the table. The argument `line_length` keeps the table narrow.

```python
import keras

keras.utils.set_random_seed(0)

model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse")
model.summary(line_length=60)
```

```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Layer (type)             ┃ Output Shape      ┃   Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ dense (Dense)            │ (None, 1)         │         2 │
└──────────────────────────┴───────────────────┴───────────┘
 Total params: 2 (8.00 B)
 Trainable params: 2 (8.00 B)
 Non-trainable params: 0 (0.00 B)
```

The `None` in the output shape is the batch axis. Its size is not fixed. The `Dense` layer has 2 parameters: one kernel value and one bias value. Each is a 32-bit float, so the table reports 8 bytes.

After `fit`, the same table shows a fourth line, `Optimizer params`. For this model it reports 6. Adam keeps a moving average of the gradient and of the squared gradient for each of the 2 weights, which is 4 values. It also keeps a step counter and a learning rate, which is 2 more. The optimizer state is part of the model when you save it. [Chapter 14](14-saving-loading-export.md) shows the cost.

## What `History` holds

`fit` returns a `History` object. Its `history` attribute is a dictionary. Each key is a metric name, and each value is a list with one number for each epoch.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)
x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.Adam(0.1), loss="mse")
history = model.fit(x, 2 * x + 1, epochs=5, batch_size=16, verbose=0)

print(list(history.history), len(history.history["loss"]))
print("steps per epoch:", int(np.ceil(64 / 16)))
print("optimizer iterations:", int(model.optimizer.iterations.numpy()))
```

```text
['loss'] 5
steps per epoch: 4
optimizer iterations: 20
```

The history has one key, `loss`, and five values. The loss for an epoch is the average of the loss of every batch in that epoch. It is not the loss of the final weights. This is why the first value in the earlier block was 1.8592 and not the loss of the random start.

One epoch is one pass through the data. With 64 samples and a batch size of 16, one epoch has 4 steps. After 5 epochs the optimizer makes 20 updates. The count of updates, not the count of epochs, is what the learning rate acts on.

## The optimizer and the learning rate decide the result

The model and the data did not change between the runs below. Only the optimizer did. Each run trains for 100 epochs from the same seed, and the block prints the final loss.

```python
import keras
import numpy as np

x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
y = 2 * x + 1

def final_loss(optimizer):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
    model.compile(optimizer=optimizer, loss="mse")
    history = model.fit(x, y, epochs=100, batch_size=16, verbose=0)
    return history.history["loss"][-1]

runs = {
    "SGD  lr=0.01": keras.optimizers.SGD(0.01),
    "SGD  lr=0.1": keras.optimizers.SGD(0.1),
    "SGD  lr=1.5": keras.optimizers.SGD(1.5),
    "Adam lr=0.1": keras.optimizers.Adam(0.1),
    "Adam lr=0.001": keras.optimizers.Adam(0.001),
}
for name, optimizer in runs.items():
    print(f"{name:14s} final loss {final_loss(optimizer):.3g}")
```

```text
SGD  lr=0.01   final loss 0.00538
SGD  lr=0.1    final loss 1.31e-13
SGD  lr=1.5    final loss nan
Adam lr=0.1    final loss 0
Adam lr=0.001  final loss 1.27
```

Read the table from the top. SGD with a learning rate of 0.01 is slow but it works. A rate of 0.1 is much better. A rate of 1.5 is too large, and each step overshoots, so the loss grows until it becomes `nan`.

Adam with its default rate of 0.001 is too slow for this problem. After 100 epochs the loss is still 1.27. Adam with a rate of 0.1 reaches zero.

Keras did not raise an error for the `nan` run. The run finished and `fit` returned a history. A loss of `nan` is a result, not an exception. [Chapter 7](07-callbacks-and-training-control.md) shows the callback that stops a run when this happens.

## Three mistakes and what Keras says

Shape errors are the most common first failure. The block below makes three of them on purpose. It prints the first line of each message.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)

x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
y = 2 * x + 1
model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")

def attempt(label, xs, ys):
    try:
        model.fit(xs, ys, epochs=1, verbose=0)
        print(f"{label}: accepted")
    except Exception as error:
        print(f"{label}: {type(error).__name__}: {str(error).splitlines()[0][:60]}")

attempt("rank-1 x      ", x.ravel(), y.ravel())
attempt("3 features    ", np.zeros((64, 3), dtype="float32"), y)
attempt("10 targets    ", x, y[:10])
```

```text
rank-1 x      : accepted
3 features    : ValueError: Exception encountered when calling Sequential.call().
10 targets    : ValueError: Data cardinality is ambiguous. Make sure all arrays contain
```

The first call is the surprise. A rank-1 array of 64 numbers is accepted for a model that expects shape `(1,)`. Keras adds the missing axis. This is convenient here and wrong in other places, so give your arrays explicit shapes.

The second and third calls fail with a `ValueError`. The second message names the layer. The third message names the cause.

## How it works

`fit` is a loop. In each step it does four things: a forward pass, a loss, a gradient and an update. You can write one step with a pencil, and Keras does the same arithmetic.

For the line model, the output is `w * x + b`. The loss is the mean of `(w * x + b - y)²`. The gradient of the loss with respect to `w` is `2 * mean(error * x)`. The gradient with respect to `b` is `2 * mean(error)`. SGD subtracts the learning rate times the gradient from each weight.

The next block does one step by hand and one step with `fit`. It uses one batch that holds all 64 samples, so one epoch is one step. It compares the weights.

```python
import keras
import numpy as np

x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
y = 2 * x + 1

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(optimizer=keras.optimizers.SGD(0.1), loss="mse")
w0, b0 = (a.copy() for a in model.layers[0].get_weights())

error = w0[0, 0] * x + b0[0] - y
gradient_w, gradient_b = 2 * np.mean(error * x), 2 * np.mean(error)
by_hand = [w0[0, 0] - 0.1 * gradient_w, b0[0] - 0.1 * gradient_b]

model.fit(x, y, epochs=1, batch_size=64, shuffle=False, verbose=0)
w1, b1 = model.layers[0].get_weights()
print("before: w = %.4f  b = %.4f" % (w0[0, 0], b0[0]))
print("gradient: dw = %.4f  db = %.4f" % (gradient_w, gradient_b))
print("by hand: w = %.4f  b = %.4f" % tuple(by_hand))
print("fit:     w = %.4f  b = %.4f" % (w1[0, 0], b1[0]))
print("equal:", bool(np.allclose(by_hand, [w1[0, 0], b1[0]], atol=1e-6)))

default = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
default.compile(optimizer="sgd", loss="mse")
default.fit(x, y, epochs=1, verbose=0)
print("steps in one epoch with the default batch_size:", int(default.optimizer.iterations.numpy()))
```

```text
before: w = 0.0543  b = 0.0000
gradient: dw = -1.3383  db = -2.0000
by hand: w = 0.1882  b = 0.2000
fit:     w = 0.1882  b = 0.2000
equal: True
steps in one epoch with the default batch_size: 2
```

The two results agree. The invariant is simple: one step of `fit` is one gradient step on one batch. The last line shows the default batch size. With `batch_size=None`, Keras uses 32 samples, so 64 samples give 2 steps.

The mechanism explains two results of this chapter. A learning rate of 1.5 gives a step that is larger than the distance to the minimum. Each step overshoots by more than the step before, and the loss grows to `nan`. A batch size of 16 gives 4 steps in one epoch. Each of the 4 gradients is a noisier estimate than the gradient of all 64 samples.

## Reference

The next block prints the signature of each function that this chapter uses. The signatures come from the installed Keras. The text under the block is what the library printed.

```python
import keras
from keras import layers
from examples.signatures import show

show(("Sequential", keras.Sequential.__init__), ("Input", keras.Input), layers.Dense,
     ("Model.compile", keras.Model.compile), ("Model.fit", keras.Model.fit),
     ("Model.predict", keras.Model.predict), ("Model.evaluate", keras.Model.evaluate))
```

```text
Sequential(layers=None, trainable=True, name=None)
Input(shape=None, batch_size=None, dtype=None, sparse=None, ragged=None,
    batch_shape=None, name=None, tensor=None, optional=False)
Dense(units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, lora_rank=None,
    lora_alpha=None, quantization_config=None, **kwargs)
Model.compile(optimizer='rmsprop', loss=None, loss_weights=None, metrics=None,
    weighted_metrics=None, run_eagerly=False, steps_per_execution=1,
    jit_compile='auto', auto_scale_loss=True)
Model.fit(x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0,
    steps_per_epoch=None, validation_steps=None, validation_batch_size=None,
    validation_freq=1)
Model.predict(x, batch_size=None, verbose='auto', steps=None, callbacks=None)
Model.evaluate(x=None, y=None, batch_size=None, verbose='auto',
    sample_weight=None, steps=None, callbacks=None, return_dict=False,
    **kwargs)
```

The parameters that matter most in this chapter:

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `Input(shape)` | `None` | shape of one sample, without the batch axis | a rank-1 array is accepted and Keras adds the axis (Chapter 1) |
| `Dense(units)` | required | number of outputs; the kernel has shape `(features, units)` | — |
| `Dense(activation)` | `None` | function applied to the output | `None` gives raw scores, which a loss may need to know about (Chapter 6) |
| `compile(optimizer)` | `'rmsprop'` | rule that turns a gradient into a step | `SGD(1.5)` gave `nan`; `Adam(0.001)` was too slow for the line (Chapter 1) |
| `compile(loss)` | `None` | number that training makes smaller | — |
| `fit(epochs)` | `1` | passes through the data | the number of steps is `epochs × ceil(n / batch_size)` |
| `fit(batch_size)` | `None` | samples in one step; `None` gives 32 | 4 steps in one epoch at 16, 2 at the default of 32 |
| `fit(verbose)` | `'auto'` | progress output | — |
| `fit(shuffle)` | `True` | shuffle the arrays at each epoch | ignored for a `tf.data.Dataset` (Chapter 5) |
| `predict(batch_size)` | `None` | samples in one forward pass | — |
| `evaluate(return_dict)` | `False` | return a dictionary and not a list | a list has the loss first, then the metrics in the order of `compile` |

## What you have now

You built a model, trained it and read its summary, its history and its parameter count. You also saw that the learning rate can decide the result and that `nan` does not raise an error.

[Chapter 2](02-tensors-and-ops.md) goes below the layers. It shows the tensors that the back end creates and the `keras.ops` functions that work on every back end. Exercises for this chapter are in [exercises/01.md](../exercises/01.md), and the worked solutions are in [exercises/solutions/01.md](../exercises/solutions/01.md).
