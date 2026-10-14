# Chapter 9 — Custom losses, metrics and training steps

> **Build** — a custom loss, a custom metric and a custom `train_step`, each checked against a built-in reference, and a portable way to change the loss of a model on every back end.
>
> **Theory kept** — `fit` is a loop of four things: a forward pass, a loss, a gradient and an update. Keras lets you replace each of them. Replace the smallest part that does the job.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch09_custom_training`, which is [`examples/ch09_custom_training.py`](../examples/ch09_custom_training.py).

The built-in pieces cover most work. Sometimes you need a loss that Keras does not have, a metric that the task defines, or a gradient that you must see. Keras gives a hook for each. This chapter uses each hook. It checks each one against the built-in version of the same thing.

Then you know that the hook is right before you change it.

The rule for the whole chapter is: **first make the custom version equal to the built-in version, then change it.** An equal result proves that the loop is right. A changed result is then the change and nothing else.

## A custom loss function

A loss is a function of `y_true` and `y_pred` that returns one value for each sample. Keras averages it over the batch. The next block writes cross-entropy by hand with `keras.ops`. It trains one model with the built-in loss and one with the hand-written loss. Both runs use the same seed and the same batch order.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

x_train, x_test, y_train, y_test = load()

def my_cross_entropy(y_true, y_pred):
    y_true = ops.cast(y_true, "int32")
    log_probability = ops.log_softmax(y_pred, axis=-1)
    picked = ops.take_along_axis(log_probability, ops.expand_dims(y_true, -1), axis=-1)
    return -ops.mean(picked, axis=-1)

def train(loss):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile("adam", loss, metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, shuffle=False, verbose=0)
    return model, history.history["loss"]

built_in, losses_a = train(keras.losses.SparseCategoricalCrossentropy(from_logits=True))
custom, losses_b = train(my_cross_entropy)
print("built-in loss per epoch:", [round(v, 4) for v in losses_a])
print("custom loss per epoch  :", [round(v, 4) for v in losses_b])
difference = max(float(np.abs(a - b).max()) for a, b in zip(built_in.get_weights(), custom.get_weights()))
print("largest weight difference below 1e-6:", difference < 1e-6)
```

```text
built-in loss per epoch: [2.0879, 1.5542, 1.0979]
custom loss per epoch  : [2.0879, 1.5542, 1.0979]
largest weight difference below 1e-6: True
```

The two runs give the same losses and the same weights, to within rounding of 32-bit numbers. The custom function is now trustworthy, and you can change it.

Note two details of the function. It uses only `keras.ops`, so it runs on every back end. It returns one value for each sample, and Keras does the averaging.

## A custom loss class

A function has no parameters. For a loss with a parameter, subclass `keras.losses.Loss`. The class below is cross-entropy with label smoothing. The smoothing moves a share `epsilon` of the target probability from the true class to all classes equally.

```python
import keras
import numpy as np
from keras import ops

class SmoothedCrossEntropy(keras.losses.Loss):
    def __init__(self, epsilon=0.1, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.num_classes = num_classes

    def call(self, y_true, y_pred):
        target = ops.one_hot(ops.cast(y_true, "int32"), self.num_classes)
        target = target * (1.0 - self.epsilon) + self.epsilon / self.num_classes
        return -ops.sum(target * ops.log_softmax(y_pred, axis=-1), axis=-1)

    def get_config(self):
        return {**super().get_config(), "epsilon": self.epsilon, "num_classes": self.num_classes}

rng = np.random.default_rng(0)
logits = rng.normal(size=(32, 10)).astype("float32")
labels = rng.integers(0, 10, size=(32,))

mine = SmoothedCrossEntropy(0.1)(labels, logits)
reference = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)(
    keras.utils.to_categorical(labels, 10), logits)
print("custom class  :", round(float(ops.convert_to_numpy(mine)), 5))
print("built-in class:", round(float(ops.convert_to_numpy(reference)), 5))
print("config:", SmoothedCrossEntropy(0.2).get_config()["epsilon"])
```

```text
custom class  : 2.56721
built-in class: 2.56721
config: 0.2
```

The class matches the built-in loss with label smoothing on 32 random samples. The method `get_config` matters for saving. A loss class without `get_config` loads, but its arguments return to their defaults ([Chapter 14](14-saving-loading-export.md) measures this).

#### `keras.losses.Loss`

```text
keras.losses.Loss(
    name=None,
    reduction='sum_over_batch_size',
    dtype=None,
)
```

The class is the base of a loss. A subclass writes `call(y_true, y_pred)`, which returns one value for each sample, and passes `**kwargs` to the base.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `name` | `None` | the name. With `None`, the class name in snake case | — (a class `Sq` got the name `sq`) |
| `reduction` | `'sum_over_batch_size'` | how the values of the samples combine | for the squared errors 1, 4, 9 and 16, the default gave 7.5 and `"sum"` gave 30.0. `None` returns the values of the samples. `"mean"` is also accepted |
| `dtype` | `None` | the type of the computation | — |

## A custom metric

A metric keeps state across batches. The loss of an epoch is an average of batch values. A metric such as balanced accuracy cannot be built that way, because it is an average over classes of per-class recall. The metric must keep a count for each class and combine the counts at the end.

A `keras.metrics.Metric` has four methods. `update_state` takes one batch. `result` returns the value. `reset_state` clears the counts at the start of an epoch. The state lives in variables made with `add_variable`.

```python
import keras
import numpy as np
from keras import layers, ops
from sklearn.metrics import balanced_accuracy_score
from examples.digits import load

class BalancedAccuracy(keras.metrics.Metric):
    def __init__(self, num_classes=10, name="balanced_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.hits = self.add_variable(shape=(num_classes,), initializer="zeros", name="hits")
        self.counts = self.add_variable(shape=(num_classes,), initializer="zeros", name="counts")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.cast(ops.reshape(y_true, (-1,)), "int32")
        predicted = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        hit = ops.cast(ops.equal(predicted, y_true), "float32")
        onehot = ops.one_hot(y_true, self.num_classes)
        self.hits.assign(self.hits + ops.sum(onehot * ops.expand_dims(hit, -1), axis=0))
        self.counts.assign(self.counts + ops.sum(onehot, axis=0))

    def result(self):
        return ops.mean(self.hits / ops.maximum(self.counts, 1.0))

    def reset_state(self):
        self.hits.assign(ops.zeros_like(self.hits))
        self.counts.assign(ops.zeros_like(self.counts))

x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy", BalancedAccuracy()])
history = model.fit(x_train, y_train, epochs=3, batch_size=32,
                    validation_data=(x_test, y_test), verbose=0).history

predicted = model.predict(x_test, verbose=0).argmax(axis=1)
print("history keys     :", sorted(history))
print("Keras metric     :", round(history["val_balanced_accuracy"][-1], 5))
print("scikit-learn     :", round(float(balanced_accuracy_score(y_test, predicted)), 5))
print("plain accuracy   :", round(history["val_accuracy"][-1], 5))
```

```text
history keys     : ['accuracy', 'balanced_accuracy', 'loss', 'val_accuracy', 'val_balanced_accuracy', 'val_loss']
Keras metric     : 0.89342
scikit-learn     : 0.89342
plain accuracy   : 0.89333
```

The custom metric agrees with the scikit-learn function to five digits. It also differs from plain accuracy, because the 450 test images are not perfectly balanced across classes. On a data set with a rare class the two numbers can differ much more. Balanced accuracy is the one that shows a model that ignores the rare class.

#### `keras.metrics.Metric`

```text
keras.metrics.Metric(
    dtype=None,
    name=None,
)
```

The class is the base of a metric. A subclass writes `update_state`, `result` and `reset_state`, and keeps its state in variables made with `add_variable`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `dtype` | `None` | the type of the state and of the result | — |
| `name` | `None` | the key in `history`. With `None`, the class name in snake case | — |

A state in a plain Python attribute does not work. A metric that counted the batches in `self.n += 1.0` reported 2.0 for the first epoch and 0.0 for the second. Each epoch had 43 batches. `fit` runs a compiled function, and the Python line runs only while Keras traces it. The same count in a variable from `add_variable` gave 1,347 samples in each epoch, which is correct.

#### `keras.metrics.Metric.add_variable`

```text
keras.metrics.Metric.add_variable(
    shape,
    initializer,
    dtype=None,
    aggregation='sum',
    name=None,
)
```

The method creates a variable of the state and tracks it, so that `reset_state` and saving can find it.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `shape` | required | the shape of the state | — |
| `initializer` | required | the start values, such as `"zeros"` | there is no default, unlike `Layer.add_weight`: without it, the call raises `TypeError: Metric.add_variable() missing 1 required positional argument: 'initializer'` |
| `dtype` | `None` | the type. With `None`, the type of the metric | — |
| `aggregation` | `'sum'` | how to combine copies on several devices. The default is `'sum'`, and for a layer weight it is `'none'` | — |
| `name` | `None` | the name of the variable | — |

## A custom `train_step`

The method `train_step` is the loop body of `fit`. It receives one batch and returns a dictionary of metric values. If you override it, you write the forward pass, the gradient and the update.

This is the one place where Keras 3 code depends on the back end. The gradient comes from the back end: `tf.GradientTape` on TensorFlow, `jax.grad` on JAX, and `loss.backward()` on PyTorch. The block below is for the TensorFlow back end.

It adds one thing to the built-in step: the global norm of the gradient, logged as a metric. Before it adds anything, the block checks that the step reproduces the built-in `fit`.

```python
import keras
import numpy as np
import tensorflow as tf
from keras import layers
from examples.digits import load

class GradNormModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norm = keras.metrics.Mean(name="grad_norm")

    @property
    def metrics(self):
        return super().metrics + [self.grad_norm]

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x=x, y=y, y_pred=y_pred)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply(gradients, self.trainable_variables)
        self.grad_norm.update_state(tf.linalg.global_norm(gradients))
        batch = tf.cast(tf.shape(x)[0], "float32")
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss, sample_weight=batch)
            elif metric is not self.grad_norm:
                metric.update_state(y, y_pred)
        return {metric.name: metric.result() for metric in self.metrics}

def make(model_class):
    keras.utils.set_random_seed(0)
    inputs = keras.Input((64,))
    hidden = layers.Dense(64, activation="relu")(inputs)
    model = model_class(inputs, layers.Dense(10)(hidden))
    model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    return model

x_train, x_test, y_train, y_test = load()
reference, custom = make(keras.Model), make(GradNormModel)
history_a = reference.fit(x_train, y_train, epochs=3, batch_size=32, shuffle=False, verbose=0).history
history_b = custom.fit(x_train, y_train, epochs=3, batch_size=32, shuffle=False, verbose=0).history

print("reference loss:", [round(v, 4) for v in history_a["loss"]])
print("custom loss   :", [round(v, 4) for v in history_b["loss"]])
print("weights equal :", all(np.array_equal(a, b) for a, b in zip(reference.get_weights(), custom.get_weights())))
print("history keys  :", sorted(history_b))
print("gradient norm per epoch:", [round(v, 3) for v in history_b["grad_norm"]])
```

```text
reference loss: [2.0879, 1.5542, 1.0979]
custom loss   : [2.0879, 1.5542, 1.0979]
weights equal : True
history keys  : ['accuracy', 'grad_norm', 'loss']
gradient norm per epoch: [0.901, 0.946, 0.915]
```

The custom loop gives the same losses and the same weights as the built-in loop, bit for bit. The extra key `grad_norm` is the new information: the average global norm of the gradient in each epoch.

Two details in the block cost a debugging session each. You must weight the loss metric by the batch size. The last batch has 3 images, and an unweighted mean gives a different epoch loss.

The second detail is the order of layer construction in `make`. [Chapter 15](15-performance-and-reproducibility.md) shows how that order decides the initial weights.

The step used two methods of the model and one method of the optimizer. The built-in step also calls `compute_metrics`.

#### `keras.Model.compute_loss`

```text
keras.Model.compute_loss(
    x=None,
    y=None,
    y_pred=None,
    sample_weight=None,
    training=True,
)
```

The method returns the loss of one batch: the loss of `compile` plus the losses of the layers, such as an L2 penalty.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | `None` | the inputs of the batch | — |
| `y` | `None` | the targets | — |
| `y_pred` | `None` | the output of the model for `x` | it is not computed for you: without it, the call raises `ValueError: Attempt to convert a value (None) with an unsupported type` |
| `sample_weight` | `None` | a weight for each sample | the default reduction divides by the batch size, not by the sum of the weights. With the weight 0 on 16 of 32 samples, the cross-entropy part was half the loss of the 16 samples alone |
| `training` | `True` | the mode of the call | — (with `training=False`, the L2 penalty of 1.674 was still in the loss: 4.2105 in both modes) |

#### `keras.Model.compute_metrics`

```text
keras.Model.compute_metrics(
    x,
    y,
    y_pred,
    sample_weight=None,
)
```

The method updates the metrics of `compile` with one batch and returns their values as a dict, with the key `loss` first.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `x` | required | the inputs of the batch | — |
| `y` | required | the targets | — |
| `y_pred` | required | the output of the model | — |
| `sample_weight` | `None` | a weight for each sample | — |

The `loss` value in the dict comes from the loss tracker, and `compute_metrics` does not update it. After a reset of the metrics, the call returned `loss` 0.0 and the correct accuracy.

#### `keras.optimizers.Optimizer.apply`

```text
keras.optimizers.Optimizer.apply(
    grads,
    trainable_variables=None,
)
```

The method changes each variable with its gradient and adds 1 to `optimizer.iterations`.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `grads` | required | one gradient for each variable, in the same order | a list that is too short is accepted with no error: 1 gradient for 2 variables changed the first variable and left the second at 1.0 |
| `trainable_variables` | `None` | the variables. With `None`, the variables of the first call | in a first call, `None` raises `ValueError: When passing grads without variables, the optimizer must already be built` |

## A portable change: override `compute_loss`

A `train_step` that uses `tf.GradientTape` runs on one back end only. If the change that you want is a change of the loss, there is a portable hook. Override `compute_loss`. Keras calls it in the training step of every back end.

The block below adds a penalty on the size of the outputs to a `Sequential` model. It then checks the new loss against a hand calculation.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

class OutputPenalty(keras.Sequential):
    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None, training=True):
        base = super().compute_loss(x=x, y=y, y_pred=y_pred, sample_weight=sample_weight, training=training)
        return base + 0.01 * ops.mean(ops.square(y_pred))

x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
model = OutputPenalty([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile("adam", loss, metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), verbose=0)

logits = model.predict(x_test, verbose=0)
by_hand = float(loss(y_test, logits)) + 0.01 * float(np.mean(logits ** 2))
reported = model.evaluate(x_test, y_test, verbose=0)[0]
print("loss by hand:", round(by_hand, 4))
print("evaluate    :", round(reported, 4))
print("equal       :", abs(by_hand - reported) < 1e-4)
```

```text
loss by hand: 0.9281
evaluate    : 0.9281
equal       : True
```

The reported loss is the cross-entropy plus the penalty. The same class runs on JAX and PyTorch, because the hook is part of the Keras `Model` class. This book ran it on TensorFlow only.

## Which hook to use

Use the smallest hook that does the job:

1. A different loss: write a function or a `Loss` class.
2. A different number to read: write a `Metric`.
3. An extra term in the loss that uses the model: override `compute_loss`.
4. A change in how the gradient is used, such as a second optimizer: override `train_step`, and accept that it ties the code to one back end. Gradient accumulation does not need it: it is the optimizer argument `gradient_accumulation_steps` ([Chapter 6](06-losses-metrics-optimizers.md)).

## How it works

The optimizer is a small program with its own state. `optimizer.apply(gradients, variables)` reads each gradient, updates its internal numbers and changes the variables. In `train_step`, this one call replaces the update step of `fit`.

Adam keeps two numbers for each weight: a moving average of the gradient `m` and a moving average of its square `v`. At step `t` the update is `lr × m̂ / (√v̂ + ε)`, where `m̂` and `v̂` are corrected for the start. At the first step, `m̂` equals the gradient `g` and `v̂` equals `g²`.

The update is then `lr × g / (|g| + ε)`, which is close to `lr × sign(g)`. The size of the gradient has almost no effect on the first step.

The next block checks this with `optimizer.apply` on one variable.

```python
import keras
import numpy as np
from keras import ops

for gradient in (0.5, 50.0):
    weight = keras.Variable(np.array([1.0], dtype="float32"))
    optimizer = keras.optimizers.Adam(learning_rate=0.1)
    optimizer.apply([ops.convert_to_tensor(np.array([gradient], dtype="float32"))], [weight])
    change = 1.0 - float(ops.convert_to_numpy(weight)[0])
    expected = 0.1 * gradient / (abs(gradient) + 1e-7)
    print(f"gradient {gradient:5.1f}: weight moved by {change:.4f}, formula {expected:.4f}")

weight = keras.Variable(np.array([1.0], dtype="float32"))
sgd = keras.optimizers.SGD(learning_rate=0.1)
sgd.apply([ops.convert_to_tensor(np.array([50.0], dtype="float32"))], [weight])
print("SGD with gradient 50.0: weight moved by", round(1.0 - float(ops.convert_to_numpy(weight)[0]), 4))
```

```text
gradient   0.5: weight moved by 0.1000, formula 0.1000
gradient  50.0: weight moved by 0.1000, formula 0.1000
SGD with gradient 50.0: weight moved by 5.0
```

A gradient of 0.5 and a gradient of 50 move the Adam weight by the same amount: 0.1. SGD moves it by 5.

The invariant is this: Adam steps by a fixed size at the start, and SGD steps in proportion to the gradient. This is why Adam hid the scale problem of Chapter 5. It is also why Adam forgives a bad learning rate more than SGD does.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `keras.losses.Loss` | A custom loss class | the base of a loss: `call` returns one value for each sample |
| `keras.metrics.Metric` | A custom metric | the base of a metric: state in variables, not in Python attributes |
| `keras.metrics.Metric.add_variable` | A custom metric | a tracked state variable. The initializer is required |
| `keras.Model.compute_loss` | A custom `train_step` | the loss of `compile` plus the losses of the layers |
| `keras.Model.compute_metrics` | A custom `train_step` | update the metrics and return their values |
| `keras.optimizers.Optimizer.apply` | A custom `train_step` | one update of the variables with their gradients |

## What you have now

You have a custom loss, a custom metric and a custom step. Each one equals a built-in reference before it changes anything. The habit is the lesson: check the loop against the built-in first.

[Chapter 10](10-convolutional-networks.md) starts the part on architectures, with images. Exercises for this chapter are in [exercises/09.md](../exercises/09.md), and the worked solutions are in [exercises/solutions/09.md](../exercises/solutions/09.md).
