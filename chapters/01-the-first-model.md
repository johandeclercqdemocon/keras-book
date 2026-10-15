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

The data are `float32` because the weights of the model are `float32`, the default type of Keras. The call `keras.utils.set_random_seed(0)` fixes the random start values of the weight, so that your run prints the same numbers. `get_weights` returns the weights of a layer as NumPy arrays. [Chapter 3](03-layers-and-weights.md) has its block. [Chapter 2](02-tensors-and-ops.md) explains it. The optimizers `Adam` and `SGD` have their blocks in [Chapter 6](06-losses-metrics-optimizers.md).

## The five calls

Every Keras program in this book uses the same five calls. The block above contains all of them.

1. `keras.Input((1,))` declares the shape of one sample. The batch size is not part of it.
2. `keras.layers.Dense(1)` is a layer. It computes `x @ kernel + bias` with one output unit.
3. `keras.Sequential([...])` stacks the layers into a model.
4. `model.compile(optimizer=..., loss=...)` chooses how to learn. It does not change the weights.
5. `model.fit(x, y, ...)` runs the training loop and returns a `History` object.

The five calls have the signatures that follow. Each table has one row for each parameter. The last column gives a result that a run of this book measured, or `—` when no run measured one.

#### `keras.Input`

```text
keras.Input(
    shape=None,
    batch_size=None,
    dtype=None,
    sparse=None,
    ragged=None,
    batch_shape=None,
    name=None,
    tensor=None,
    optional=False,
)
```

The function returns a symbolic tensor that describes one input of the model.

The table under `keras.Input` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `keras.layers.Dense`

```text
keras.layers.Dense(
    units,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    lora_rank=None,
    lora_alpha=None,
    quantization_config=None,
    **kwargs,
)
```

The layer computes `activation(x @ kernel + bias)`. The kernel has the shape `(features, units)`.

The table under `keras.layers.Dense` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `keras.Sequential`

```text
keras.Sequential(
    layers=None,
    trainable=True,
    name=None,
)
```

The class makes a model from a list of layers. The output of each layer is the input of the next.

The table under `keras.Sequential` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `keras.Model.compile`

```text
keras.Model.compile(
    optimizer='rmsprop',
    loss=None,
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile='auto',
    auto_scale_loss=True,
)
```

The method chooses how the model learns. It does not change a weight.

The table under `keras.Model.compile` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `keras.Model.fit`

```text
keras.Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose='auto',
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)
```

The method runs the training loop and returns a `History` object.

The table under `keras.Model.fit` in the Reference section of this chapter gives each parameter and what goes wrong with it.

Two more calls read the result. `model.evaluate(x, y)` returns the loss on data. `model.predict(x)` returns the output of the model. Both calls work on batches, so the input must have a batch axis as its first axis.

#### `keras.Model.predict`

```text
keras.Model.predict(
    x,
    batch_size=None,
    verbose='auto',
    steps=None,
    callbacks=None,
)
```

The method returns the output of the model for each sample, as a NumPy array.

The table under `keras.Model.predict` in the Reference section of this chapter gives each parameter and what goes wrong with it.

#### `keras.Model.evaluate`

```text
keras.Model.evaluate(
    x=None,
    y=None,
    batch_size=None,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
    **kwargs,
)
```

The method returns the loss, and the metrics of `compile`, on the data.

The table under `keras.Model.evaluate` in the Reference section of this chapter gives each parameter and what goes wrong with it.

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

#### `keras.Model.summary`

```text
keras.Model.summary(
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
)
```

The method prints the layers, their output shapes and the count of parameters.

The table under `keras.Model.summary` in the Reference section of this chapter gives each parameter and what goes wrong with it.

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

| Callable | Introduced in | Contract |
|---|---|---|
| `keras.Input` | The five calls | the shape of one sample, without the batch axis |
| `keras.layers.Dense` | The five calls | `activation(x @ kernel + bias)` |
| `keras.Sequential` | The five calls | a model from a list of layers |
| `keras.Model.compile` | The five calls | the optimizer, the loss and the metrics. No weight changes |
| `keras.Model.fit` | The five calls | the training loop. It returns a `History` |
| `keras.Model.predict` | The five calls | the output for each sample, as a NumPy array |
| `keras.Model.evaluate` | The five calls | the loss and the metrics on data |
| `keras.Model.summary` | Read the summary | a table of layers and parameter counts |

### Parameters

##### `keras.Input`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `shape` | `None` | the shape of one sample, without the batch axis | a rank-1 array of 64 numbers is accepted for `shape=(1,)`. Keras adds the axis (this chapter) |
| `batch_size` | `None` | a fixed batch size | `Input((1,), batch_size=16)` still accepted `fit` with `batch_size=32`, with no error |
| `dtype` | `None` | the type of the input. With `None`, the default float type, `float32` | with `dtype="int32"`, the input 0.7 becomes 0, and the prediction is 0.0 |
| `sparse` | `None` | accept sparse tensors. TensorFlow and JAX back ends only | — |
| `ragged` | `None` | accept ragged tensors. TensorFlow back end only | — |
| `batch_shape` | `None` | the shape with the batch axis, such as `(None, 1)` | with `shape` too, it raises `ValueError: You cannot pass both shape and batch_shape at the same time` |
| `name` | `None` | the name of the input | an unnamed input gets a name such as `keras_tensor_4`, and an exported model shows that name ([Chapter 14](14-saving-loading-export.md)) |
| `tensor` | `None` | an existing tensor to wrap | — |
| `optional` | `False` | with `True`, the input accepts `None` | — |

##### `keras.layers.Dense`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `units` | required | the number of outputs | — |
| `activation` | `None` | the function on the output. With `None`, the output is `x @ kernel + bias` | a loss that expects probabilities gets raw scores ([Chapter 6](06-losses-metrics-optimizers.md)) |
| `use_bias` | `True` | add the bias vector | with `False`, the line model has 1 parameter, and the loss stops at 1.0004 after 100 epochs: it cannot make the offset 1 |
| `kernel_initializer` | `'glorot_uniform'` | the start values of the kernel | with `'zeros'` in a hidden layer of 4 `tanh` units, the kernel was still `[0, 0, 0, 0]` after 50 epochs. No unit learned |
| `bias_initializer` | `'zeros'` | the start values of the bias | — |
| `kernel_regularizer` | `None` | a penalty on the kernel, added to the loss | `L2(0.1)` pulled the weight of the line to 1.5486 and not 2. The reported loss includes the penalty ([Chapter 8](08-overfitting-and-regularization.md)) |
| `bias_regularizer` | `None` | a penalty on the bias | — |
| `activity_regularizer` | `None` | a penalty on the output of the layer | — |
| `kernel_constraint` | `None` | a rule that the kernel must obey after each update | `NonNeg()` on the data y = -2x + 1 kept the weight at 0, and the loss stopped at 1.3767 |
| `bias_constraint` | `None` | a rule for the bias | — |
| `lora_rank` | `None` | train two small matrices of this rank, and freeze the kernel | with `lora_rank=2`, a layer from 16 to 8 has 56 trainable and 128 frozen values, against 136 trainable without it |
| `lora_alpha` | `None` | scale the low-rank update by `lora_alpha / lora_rank` | — |
| `quantization_config` | `None` | settings to store the weights with fewer bits | — |
| `**kwargs` | — | the arguments of every layer, such as `name`, `dtype` and `trainable` | a misspelled name raises `ValueError: Unrecognized keyword arguments passed to Dense: {'foo': 1}` |

##### `keras.Sequential`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `layers` | `None` | the layers, in order | without a `keras.Input`, the model has 0 weights until its first call, and `summary` shows `0 (unbuilt)` |
| `trainable` | `True` | with `False`, `fit` changes no weight | `fit` runs with no error, and the loss stays at 2.3019 for every epoch |
| `name` | `None` | the name of the model. With `None`, `sequential`, `sequential_1` and so on | — |

##### `keras.Model.compile`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `optimizer` | `'rmsprop'` | the rule that turns a gradient into a step | `SGD(1.5)` gave `nan`, and `Adam(0.001)` was too slow for the line (this chapter) |
| `loss` | `None` | the number that training makes smaller | with `None`, `fit` raises `ValueError: No loss to compute` |
| `loss_weights` | `None` | the weight of each loss of a model with several outputs | — |
| `metrics` | `None` | more numbers to report, such as `["mae"]` | each metric adds a key to `history`: `['loss', 'mae']` |
| `weighted_metrics` | `None` | metrics that use `sample_weight` | — |
| `run_eagerly` | `False` | with `True`, run the Python code of each step, with no graph | the first epoch gave the same loss as the graph, 2.2735. The speed was not measured |
| `steps_per_execution` | `1` | the number of steps in one call of the compiled function | with 4, `on_train_batch_end` ran 1 time in an epoch of 4 steps. The optimizer still made 4 steps |
| `jit_compile` | `'auto'` | compile the step with XLA | on the TensorFlow CPU setup of this book, `'auto'` became `False` |
| `auto_scale_loss` | `True` | scale the loss for mixed precision ([Chapter 15](15-performance-and-reproducibility.md)) | — |

##### `keras.Model.fit`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | `None` | the inputs: an array, a tensor, a `tf.data.Dataset` or a `PyDataset` | — |
| `y` | `None` | the targets. Leave it out when `x` gives the targets | 10 targets for 64 inputs raise `ValueError: Data cardinality is ambiguous` (this chapter) |
| `batch_size` | `None` | the samples in one step. With `None`, 32 | 4 steps in one epoch at 16, and 2 at the default (this chapter) |
| `epochs` | `1` | the number of the last epoch | the number of steps is `epochs × ceil(n / batch_size)` |
| `verbose` | `'auto'` | 0 is silent, 1 is a progress bar, 2 is one line for each epoch | — |
| `callbacks` | `None` | objects that Keras calls during training ([Chapter 7](07-callbacks-and-training-control.md)) | — |
| `validation_split` | `0.0` | the part of the data to keep for validation | it takes the last samples, before any shuffle. With 0.25 on the sorted line data, `val_loss` was 2.1647, the loss of the last 16 samples. The first 16 gave 0.3968 |
| `validation_data` | `None` | a separate `(x, y)` for validation | with `validation_split` too, `validation_data` wins: `val_loss` was the loss on `validation_data` |
| `shuffle` | `True` | shuffle the arrays at each epoch | it has no effect on a `tf.data.Dataset` ([Chapter 5](05-data-pipelines.md)) |
| `class_weight` | `None` | a weight for each class, as a dict | — |
| `sample_weight` | `None` | a weight for each sample | the 32 samples with x > 0 got y = 10 and the weight 0. `fit` found w = 2.0 and b = 1.0. With no weights it found w = 7.65 and b = 4.99 |
| `initial_epoch` | `0` | the number of the first epoch | with `initial_epoch=3` and `epochs=5`, `fit` runs 2 epochs, numbered 3 and 4 |
| `steps_per_epoch` | `None` | the steps in one epoch. With `None`, all the batches | with 2, two epochs made 4 updates and not 8 |
| `validation_steps` | `None` | the batches of validation data for each check | — |
| `validation_batch_size` | `None` | the batch size of the validation data | — |
| `validation_freq` | `1` | check the validation data every `n` epochs | with 2, 4 epochs gave 4 values of `loss` and 2 of `val_loss` |

##### `keras.Model.predict`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the inputs, with the batch axis first | a rank-1 array gave shape `(2, 1)`. After this call, every later `predict` of the same model raised `ValueError: Cannot take the length of shape with unknown rank`, also with the correct shape |
| `batch_size` | `None` | the samples in one forward pass. With `None`, 32 | — |
| `verbose` | `'auto'` | the progress output | — |
| `steps` | `None` | the number of batches to predict | with `batch_size=16` and `steps=1`, 16 of 64 rows. With `batch_size=None` and `steps=1`, all 64 rows |
| `callbacks` | `None` | callbacks during the prediction | — |

##### `keras.Model.evaluate`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | `None` | the inputs | — |
| `y` | `None` | the targets | — |
| `batch_size` | `None` | the samples in one step. With `None`, 32 | — |
| `verbose` | `'auto'` | the progress output | — |
| `sample_weight` | `None` | a weight for each sample | — |
| `steps` | `None` | the number of batches to use | with `batch_size=16` and `steps=1`, the result is the loss of the first 16 samples only |
| `callbacks` | `None` | callbacks during the evaluation | — |
| `return_dict` | `False` | return a dict and not a list | the list has the loss first, then the metrics in the order of `compile`: `[2.3019, 1.2412]` |
| `**kwargs` | — | no keyword is accepted here | `foo=1` raises `ValueError: Arguments not recognized: {'foo': 1}` |

##### `keras.Model.summary`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `line_length` | `None` | the width of the table. With `None`, 76 characters for this model | with 60 and `show_trainable=True`, the header `Trainable` is cut to `Tr…` |
| `positions` | `None` | the relative positions of the columns | — |
| `print_fn` | `None` | the function that prints. With `None`, `print` | the docstring says that Keras calls it for each line. It called it 1 time, with the whole table in one string |
| `expand_nested` | `False` | show the layers of a nested model | — |
| `show_trainable` | `False` | add a column that says if each layer trains | — |
| `layer_range` | `None` | the first and the last layer to show | — |


## What you have now

You built a model, trained it and read its summary, its history and its parameter count. You also saw that the learning rate can decide the result and that `nan` does not raise an error.

[Chapter 2](02-tensors-and-ops.md) goes below the layers. It shows the tensors that the back end creates and the `keras.ops` functions that work on every back end. Exercises for this chapter are in [exercises/01.md](../exercises/01.md), and the worked solutions are in [exercises/solutions/01.md](../exercises/solutions/01.md).
