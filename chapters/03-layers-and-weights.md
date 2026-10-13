# Chapter 3 — Layers and weights

> **Build** — a `MyDense` layer that gives the same output as `keras.layers.Dense`, a layer that counts its own calls, and a table of what each kind of weight does in `fit` and `predict`.
>
> **Theory kept** — a layer is a container for weights and a function `call`. Weights come into existence at the first call, not at construction.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0. The code is New.
>
> **Run this first** — `uv run python -m examples.ch03_layers_and_weights`, which is [`examples/ch03_layers_and_weights.py`](../examples/ch03_layers_and_weights.py).

Every model in Keras is a tree of layers. A `Dense` layer, a `Conv2D` layer and a whole `Sequential` model are all objects of the class `keras.layers.Layer` or of a subclass. If you know the layer contract, you can read any Keras model and write any custom part.

The contract has four parts: the constructor, `build`, `call` and `get_config`. This chapter writes a layer with all four and then measures what the framework does with it.

## A layer in four methods

The first block writes `MyDense`. It is the same computation as `keras.layers.Dense` with no activation: `x @ kernel + bias`.

```python
import keras
from keras import ops

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")

    def call(self, x):
        return ops.matmul(x, self.kernel) + self.bias

    def get_config(self):
        return {**super().get_config(), "units": self.units}
```

The constructor stores the arguments that do not depend on the input. It must call `super().__init__(**kwargs)`, so that `name`, `dtype` and `trainable` work. The method `build` receives the shape of the input and creates the weights with `add_weight`. The method `call` computes the output. The method `get_config` returns the arguments that rebuild the layer.

The `build` method is the part that surprises people. The next block shows that a new layer has no weights.

```python
import keras
import numpy as np
from keras import ops

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="zeros", name="bias")

    def call(self, x):
        return ops.matmul(x, self.kernel) + self.bias

layer = MyDense(3)
print("before the first call: built =", layer.built, "weights =", len(layer.weights))

x = np.ones((2, 4), dtype="float32")
y = layer(x)
print("after the first call:  built =", layer.built, "weights =", len(layer.weights))
print("kernel shape", tuple(layer.kernel.shape), "bias shape", tuple(layer.bias.shape))
print("output shape", tuple(y.shape))
```

```text
before the first call: built = False weights = 0
after the first call:  built = True weights = 2
kernel shape (4, 3) bias shape (3,)
output shape (2, 3)
```

The kernel has the shape `(4, 3)` because the first input had 4 features. The layer cannot know this number in advance. This is the reason for lazy weights: the same layer class works on any input width, and Keras fixes the width at the first call.

## The same numbers as `Dense`

A custom layer is correct when it matches a reference. The next block copies the weights of `MyDense` into a built-in `Dense` layer and compares the outputs.

```python
import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)

class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="glorot_uniform", name="kernel")
        self.bias = self.add_weight(shape=(self.units,), initializer="random_normal", name="bias")

    def call(self, x):
        return ops.matmul(x, self.kernel) + self.bias

x = np.random.default_rng(0).normal(size=(5, 4)).astype("float32")
mine = MyDense(3)
reference = layers.Dense(3)
mine(x)
reference(x)
reference.set_weights(mine.get_weights())

difference = np.abs(ops.convert_to_numpy(mine(x)) - ops.convert_to_numpy(reference(x))).max()
print("max difference:", float(difference))
print("weight order in get_weights:", [w.shape for w in mine.get_weights()])
```

```text
max difference: 0.0
weight order in get_weights: [(4, 3), (3,)]
```

The difference is zero. The order of `get_weights` is the order in which `add_weight` was called: kernel first, bias second. This order matters because `set_weights` takes a plain list. Two layers that create the same weights in a different order are not interchangeable. [Chapter 14](14-saving-loading-export.md) returns to this point.

## Trainable and non-trainable weights

A weight is either trainable or not. The optimizer updates trainable weights. It does not update the others. The two groups are lists on every layer: `trainable_weights` and `non_trainable_weights`.

Batch normalization is the standard example. It has two trainable weights, `gamma` and `beta`. It has two non-trainable weights, the moving mean and the moving variance. The optimizer does not touch the last two. The layer updates them itself during training.

```python
import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(0)

x = np.ones((2, 4), dtype="float32")
norm = layers.BatchNormalization()
norm(x)
print("trainable:    ", [w.path for w in norm.trainable_weights])
print("non-trainable:", [w.path for w in norm.non_trainable_weights])

dense = layers.Dense(3)
dense(x)
print("before freeze:", len(dense.trainable_weights), len(dense.non_trainable_weights))
dense.trainable = False
print("after freeze: ", len(dense.trainable_weights), len(dense.non_trainable_weights))
```

```text
trainable:     ['batch_normalization/gamma', 'batch_normalization/beta']
non-trainable: ['batch_normalization/moving_mean', 'batch_normalization/moving_variance']
before freeze: 2 0
after freeze:  0 2
```

Setting `trainable = False` moves every weight of the layer to the non-trainable list. This is how you freeze a layer. [Chapter 13](13-transfer-learning.md) uses it.

The list changes at once. The training loop does not change at once. Keras builds the training function at the first `fit` and keeps it. A change of `trainable` after that point has no effect until you call `compile` again. The next block measures four cases. It reports whether the weights of layer `a` changed during one more epoch.

```python
import keras
import numpy as np
from keras import layers

x = np.random.default_rng(0).normal(size=(64, 4)).astype("float32")
y = x.sum(1, keepdims=True)

def run(freeze_first, new_state, recompile):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((4,)), layers.Dense(8, name="a"), layers.Dense(1, name="b")])
    model.get_layer("a").trainable = not freeze_first
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
    before = model.get_layer("a").get_weights()[0].copy()
    model.get_layer("a").trainable = new_state
    if recompile:
        model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
    return not np.array_equal(before, model.get_layer("a").get_weights()[0])

for label, freeze_first, new_state, recompile in [
    ("freeze,   no recompile", False, False, False),
    ("freeze,   recompile   ", False, False, True),
    ("unfreeze, no recompile", True, True, False),
    ("unfreeze, recompile   ", True, True, True),
]:
    print(label, "-> layer a changed:", run(freeze_first, new_state, recompile))
```

```text
freeze,   no recompile -> layer a changed: True
freeze,   recompile    -> layer a changed: False
unfreeze, no recompile -> layer a changed: False
unfreeze, recompile    -> layer a changed: True
```

Two of the four cases are wrong in the way that matters. Without `compile`, a frozen layer keeps training and an unfrozen layer stays frozen. Keras raises no error and prints no warning. The rule is simple: call `compile` after every change of `trainable`.

## A weight that the optimizer never touches

A non-trainable weight is also a place to keep state. The next layer counts how many times Keras calls it. It does not change its input.

The layer keeps two counters, one for calls in training mode and one for calls in inference mode. The block puts the layer in a model and runs `predict`, `fit` and `evaluate`. After each call it prints both counters.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)

class Counter(keras.layers.Layer):
    def build(self, input_shape):
        self.train_calls = self.add_weight(shape=(), initializer="zeros", trainable=False, name="train")
        self.infer_calls = self.add_weight(shape=(), initializer="zeros", trainable=False, name="infer")

    def call(self, x, training=None):
        (self.train_calls if training else self.infer_calls).assign_add(1.0)
        return x

    def read(self):
        return int(self.train_calls.numpy()), int(self.infer_calls.numpy())

counter = Counter()
model = keras.Sequential([keras.Input((4,)), counter, keras.layers.Dense(1)])
model.compile(optimizer="sgd", loss="mse")
x = np.ones((100, 4), dtype="float32")
y = np.ones((100, 1), dtype="float32")

print("(train, infer) at start:", counter.read())
model.predict(x, batch_size=32, verbose=0)
print("after predict, 100 samples in batches of 32:", counter.read())
model.fit(x, y, batch_size=25, epochs=2, verbose=0)
print("after fit, 2 epochs of 4 steps:", counter.read())
model.evaluate(x, y, batch_size=50, verbose=0)
print("after evaluate, 2 batches:", counter.read())
```

```text
(train, infer) at start: (0, 0)
after predict, 100 samples in batches of 32: (0, 4)
after fit, 2 epochs of 4 steps: (8, 4)
after evaluate, 2 batches: (8, 6)
```

The counters measure the batch loop. `predict` makes 4 inference calls, because 100 samples in batches of 32 give 4 batches. `fit` makes 8 training calls, because 2 epochs of 4 steps give 8 steps. `evaluate` makes 2 inference calls. The layer sees batches, not samples. Any statistic that the layer keeps is a statistic over batches.

## A layer that behaves in two ways

Some layers act differently in training and in inference. `Dropout` is the best known. A custom layer gets this information through the argument `training` of `call`.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)

class AddOne(keras.layers.Layer):
    def call(self, x, training=None):
        return x + 1.0 if training else x

zeros = np.zeros((1, 2), dtype="float32")
layer = AddOne()
print("direct call:         ", layer(zeros).numpy())
print("direct, training=True:", layer(zeros, training=True).numpy())

model = keras.Sequential([keras.Input((2,)), AddOne()])
print("model.predict:        ", model.predict(zeros, verbose=0))
print("model(x, training):   ", model(zeros, training=True).numpy())
```

```text
direct call:          [[0. 0.]]
direct, training=True: [[1. 1.]]
model.predict:         [[0. 0.]]
model(x, training):    [[1. 1.]]
```

The default is inference. The previous section measured that `fit` runs in training mode and that `predict` and `evaluate` run in inference mode. When you call a model yourself, you must pass `training=True` if you want the training behavior.

## Losses that a layer adds

A layer can add a term to the total loss. A weight regularizer does this. The term appears in `layer.losses` after the layer has weights.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)

layer = keras.layers.Dense(
    2, kernel_regularizer=keras.regularizers.L2(0.01), kernel_initializer="ones")
print("before the first call:", layer.losses)

layer(np.ones((1, 5), dtype="float32"))
print("after the first call: ", [round(float(v), 4) for v in layer.losses])
print("expected: 0.01 * (5 * 2 weights of value 1) =", 0.01 * 5 * 2)
```

```text
before the first call: []
after the first call:  [0.1]
expected: 0.01 * (5 * 2 weights of value 1) = 0.1
```

The list is empty until the weights exist. After that the value is the penalty that `fit` adds to the loss at every step. It is 0.01 times the sum of the squared weights. [Chapter 8](08-overfitting-and-regularization.md) measures what this does to a model.

## How it works

When you call a layer, you call `Layer.__call__`, not your `call`. `__call__` does the bookkeeping first: it converts the inputs, it checks whether the layer is built, and it calls `build` once if it is not. Only then does it call `call`. This is why you never call `build` yourself, and why `call` has no code to create weights.

The next block records the order of events. It also shows what a built layer does with an input of a new width.

```python
import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(0)

events = []

class Traced(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        events.append("__init__")

    def build(self, input_shape):
        events.append(f"build{tuple(input_shape)}")

    def call(self, x):
        events.append("call")
        return x

layer = Traced()
layer(np.ones((2, 3), dtype="float32"))
layer(np.ones((2, 3), dtype="float32"))
layer(np.ones((5, 3), dtype="float32"))
print(events)

dense = layers.Dense(2)
dense(np.ones((1, 3), dtype="float32"))
try:
    dense(np.ones((1, 4), dtype="float32"))
except ValueError as error:
    print("a built Dense with a new width:", str(error).splitlines()[0][:70])
```

```text
['__init__', 'build(2, 3)', 'call', 'call', 'call']
a built Dense with a new width: Input 0 with name 'None' of layer 'dense' is incompatible with the lay
```

`build` runs once, with the shape of the first input. It does not run again for a batch of 5. The invariant is: *the first call fixes the weights.* Every later input must fit them. The `Dense` layer was built for 3 features, so an input of 4 features fails. `Traced` has no weights, so no shape can fail.

The mechanism also explains the counter of this chapter. Keras calls `call` once for each batch, in training and in inference, and the layer sees one batch at a time.

## Reference

```python
import keras
from keras import layers
from examples.signatures import show

show(("Layer", keras.layers.Layer.__init__), ("Layer.add_weight", keras.layers.Layer.add_weight),
     ("Layer.build", keras.layers.Layer.build), ("Layer.get_config", keras.layers.Layer.get_config),
     layers.BatchNormalization)
```

```text
Layer(activity_regularizer=None, trainable=True, dtype=None, autocast=True,
    name=None, **kwargs)
Layer.add_weight(*args, shape=None, initializer=None, dtype=None,
    trainable=True, autocast=True, regularizer=None, constraint=None,
    aggregation='none', overwrite_with_gradient=False, name=None)
Layer.build(input_shape)
Layer.get_config()
BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True,
    scale=True, beta_initializer='zeros', gamma_initializer='ones',
    moving_mean_initializer='zeros', moving_variance_initializer='ones',
    beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
    gamma_constraint=None, renorm=False, renorm_clipping=None,
    renorm_momentum=0.99, synchronized=False, **kwargs)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `Layer(name)` | `None` | name of the layer; `None` gives a name from the class | `MyDense` became `my_dense` |
| `Layer(trainable)` | `True` | the optimizer updates the weights | set to `False` after the first `fit` and Keras ignores it until `compile` (Chapter 3) |
| `Layer(dtype)` | `None` | type of the weights and the computation | — |
| `add_weight(shape)` | `None` | shape of the new weight | — |
| `add_weight(initializer)` | `None` | how the start values are made | `None` leaves the choice to Keras; name the initializer so the choice is visible |
| `add_weight(trainable)` | `True` | `False` puts the weight in `non_trainable_weights` | a counter layer used this and the optimizer did not touch it |
| `add_weight(regularizer)` | `None` | adds a penalty term to `layer.losses` | the term appears only after the layer is built |
| `build(input_shape)` | required | create the weights from the first input shape | runs once |
| `call(x, training=None)` | — | the computation | `training` is `True` inside `fit` only |
| `BatchNormalization(momentum)` | `0.99` | speed of the moving mean and variance | the two moving values are non-trainable weights |

## What you have now

You have a custom layer that matches `Dense` and a counter that exposes the batch loop. You also know how lazy weights, trainable weights and the `training` argument work. The next chapter puts layers together in three different ways and measures that the three give the same model.

Exercises for this chapter are in [exercises/03.md](../exercises/03.md), and the worked solutions are in [exercises/solutions/03.md](../exercises/solutions/03.md). Continue with [Chapter 4](04-three-ways-to-build.md).
