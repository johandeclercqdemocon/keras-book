# Chapter 6 — Losses, metrics and optimizers

> **Build** — a digit classifier with 97% accuracy on 450 test images, and four measured choices: the pairing of label format and loss, the `from_logits` flag, the optimizer, and gradient clipping.
>
> **Theory kept** — the loss is the number that training lowers. A metric is a number that people read. The two are different objects and they need not agree.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch06_losses_metrics_optimizers`, which is [`examples/ch06_losses_metrics_optimizers.py`](../examples/ch06_losses_metrics_optimizers.py).

The `compile` call takes three choices: a loss, a list of metrics and an optimizer. Each choice has a default form and a set of ways to be wrong. This chapter measures the ways to be wrong on the digits data of [Chapter 5](05-data-pipelines.md).

## A baseline

The model is the same in every block of this chapter: 64 inputs, a hidden layer of 64 units with a ReLU, and 10 outputs. The outputs are raw scores. They are not probabilities. People call raw scores logits.

```python
import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile(
    optimizer="adam",
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                    validation_data=(x_test, y_test), verbose=0)
print("train loss      :", round(history.history["loss"][-1], 4))
print("test loss       :", round(history.history["val_loss"][-1], 4))
print("test accuracy   :", round(history.history["val_accuracy"][-1], 4))
print("history keys    :", sorted(history.history))
```

```text
train loss      : 0.1283
test loss       : 0.1547
test accuracy   : 0.9711
history keys    : ['accuracy', 'loss', 'val_accuracy', 'val_loss']
```

The model reaches an accuracy of 0.9711 on the test images: 437 of 450 are correct. The history has four keys. Every key that starts with `val_` is the same quantity measured on the validation data.

## The label format and the loss

Class labels come in two formats. A label can be an integer, such as 7. It can also be a one-hot vector with 10 values and a 1 at position 7. The two formats need two loss classes: `SparseCategoricalCrossentropy` for integers and `CategoricalCrossentropy` for one-hot vectors.

They compute the same number. The next block checks this claim.

```python
import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()

def run(loss, y_train, y_test):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile("adam", loss, metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=20, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0)
    return round(h.history["val_loss"][-1], 4), round(h.history["val_accuracy"][-1], 4)

integer = run(keras.losses.SparseCategoricalCrossentropy(from_logits=True), y_train, y_test)
one_hot = run(keras.losses.CategoricalCrossentropy(from_logits=True),
              keras.utils.to_categorical(y_train, 10), keras.utils.to_categorical(y_test, 10))
print("integer labels :", integer)
print("one-hot labels :", one_hot)
print("equal          :", integer == one_hot)
```

```text
integer labels : (0.1547, 0.9711)
one-hot labels : (0.1547, 0.9711)
equal          : True
```

The result is the same to four digits. Use integer labels. They are smaller, and there is no step that can build the vectors wrongly. The metric name `"accuracy"` also adapts: Keras picks the sparse or the dense version from the shape of the labels.

## The `from_logits` flag

The last layer of the model can end with a softmax, so that the outputs are probabilities. Or it can end with no activation, so that the outputs are logits. The loss must know which one it gets. The flag `from_logits` tells it.

There are three combinations that people write. The next block trains the three and prints the test loss and accuracy.

```python
import warnings

import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()

def run(last_activation, from_logits):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"),
                              layers.Dense(10, activation=last_activation)])
    model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=from_logits),
                  metrics=["accuracy"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        h = model.fit(x_train, y_train, epochs=20, batch_size=32,
                      validation_data=(x_test, y_test), verbose=0)
    return round(h.history["val_loss"][-1], 4), round(h.history["val_accuracy"][-1], 4), len(caught) > 0

for label, activation, flag in [
    ("logits,        from_logits=True ", None, True),
    ("softmax,       from_logits=False", "softmax", False),
    ("softmax,       from_logits=True ", "softmax", True),
    ("logits,        from_logits=False", None, False),
]:
    loss, accuracy, warned = run(activation, flag)
    print(f"{label}  val loss {loss:.4f}  val accuracy {accuracy:.4f}  warning: {warned}")
```

```text
logits,        from_logits=True   val loss 0.1547  val accuracy 0.9711  warning: False
softmax,       from_logits=False  val loss 0.1547  val accuracy 0.9711  warning: False
softmax,       from_logits=True   val loss 0.1547  val accuracy 0.9711  warning: True
logits,        from_logits=False  val loss 2.1798  val accuracy 0.5711  warning: False
```

The first two rows are the two correct pairings. The third row is the one that older tutorials call a bug: a softmax output with `from_logits=True`. In Keras 3.15.1 the result is the same as the correct pairing to four digits, and Keras gives a warning. The warning says that the output came from a softmax and does not represent logits.

The fourth row is the one that harms. The model outputs raw scores, and the loss treats them as probabilities. The model trains. It gives no error and no warning. The test accuracy is 0.5711 instead of 0.9711.

This number is unstable. In an earlier run of the same setup, without `TF_DETERMINISTIC_OPS`, the accuracy was 0.7089. Do not trust one number from a run that is broken in this way.

The rule is: if the last layer has no activation, set `from_logits=True`. A missing flag is silent.

## The optimizer

The optimizer decides how the gradient becomes a step. The next block trains the same model with five optimizers and reports the test accuracy after 1, 5 and 10 epochs.

```python
import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

optimizers = {
    "SGD 0.1": lambda: keras.optimizers.SGD(0.1),
    "SGD 0.1 momentum 0.9": lambda: keras.optimizers.SGD(0.1, momentum=0.9),
    "RMSprop 0.001": lambda: keras.optimizers.RMSprop(0.001),
    "Adam 0.001": lambda: keras.optimizers.Adam(0.001),
    "AdamW 0.001": lambda: keras.optimizers.AdamW(0.001),
}
for name, make in optimizers.items():
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile(make(), loss, metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=10, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0)
    acc = h.history["val_accuracy"]
    print(f"{name:22s} epoch 1 {acc[0]:.3f}   epoch 5 {acc[4]:.3f}   epoch 10 {acc[9]:.3f}")
```

```text
SGD 0.1                epoch 1 0.716   epoch 5 0.940   epoch 10 0.962
SGD 0.1 momentum 0.9   epoch 1 0.913   epoch 5 0.969   epoch 10 0.960
RMSprop 0.001          epoch 1 0.693   epoch 5 0.924   epoch 10 0.951
Adam 0.001             epoch 1 0.693   epoch 5 0.927   epoch 10 0.951
AdamW 0.001            epoch 1 0.693   epoch 5 0.924   epoch 10 0.951
```

After the first epoch the optimizers are far apart. SGD with momentum is at 0.913 and the adaptive optimizers are at 0.693. After 10 epochs they are within 0.011 of each other, which is 5 images of 450. Do not rank them by the last column.

The first column is a real difference. Momentum keeps a running sum of past gradients, so it moves faster in a direction that stays the same. The practical statement is limited. On this small problem, momentum SGD reaches 90% sooner. On other problems the order changes.

AdamW is Adam with weight decay that is applied to the weights directly. On this problem the decay is too small to show.

## The learning rate schedule

A schedule changes the learning rate during training. Pass a schedule object in place of a number. The next block shows the values of a cosine decay and trains with it.

```python
import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
steps = 10 * (len(x_train) // 32)
schedule = keras.optimizers.schedules.CosineDecay(0.01, decay_steps=steps)
print("rate at step 0, middle, end:",
      [round(float(schedule(s)), 5) for s in (0, steps // 2, steps)])

def run(learning_rate):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile(keras.optimizers.Adam(learning_rate),
                  keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=10, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0)
    return round(h.history["val_accuracy"][-1], 4)

print("constant 0.01   :", run(0.01))
print("cosine from 0.01:", run(schedule))
```

```text
rate at step 0, middle, end: [0.01, 0.005, 0.0]
constant 0.01   : 0.9733
cosine from 0.01: 0.9689
```

The schedule works as described: the rate starts at 0.01 and ends at 0. On this problem it does not help. The constant rate is 0.0044 higher, which is 2 images of 450. A schedule pays back on longer runs and larger models. Do not add one without a measurement.

## Gradient clipping

A single bad batch can produce a gradient that is far larger than the rest. Clipping limits the size of the step. Keras has three clipping arguments on every optimizer: `clipnorm`, `clipvalue` and `global_clipnorm`.

The next block uses a learning rate that is too large on purpose. It shows which of the three arguments rescues the run.

```python
import keras
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()

def run(**clipping):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile(keras.optimizers.SGD(2.0, **clipping),
                  keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    h = model.fit(x_train, y_train, epochs=5, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0)
    return round(h.history["val_accuracy"][-1], 3)

print("no clipping          :", run())
print("clipnorm=1.0         :", run(clipnorm=1.0))
print("clipvalue=0.1        :", run(clipvalue=0.1))
print("global_clipnorm=1.0  :", run(global_clipnorm=1.0))
```

```text
no clipping          : 0.1
clipnorm=1.0         : 0.1
clipvalue=0.1        : 0.942
global_clipnorm=1.0  : 0.96
```

Without clipping the run fails: the accuracy is 0.1, which is chance for 10 classes. `clipnorm` does not help in this run. The argument `clipvalue` helps in part, and `global_clipnorm` rescues the run.

The three arguments act in different ways. `clipnorm` rescales the gradient of each weight array alone. `clipvalue` limits each number of the gradient to the given range. `global_clipnorm` rescales all gradients together by one factor, so the direction of the update stays the same. The measurement shows the effect and not the reason. On this problem only `global_clipnorm` rescued the run fully.

## How it works

Cross-entropy is the negative log of the probability that the model gave to the right class. With raw scores, a softmax makes the probabilities. The two steps combine into one formula that is stable in 32-bit numbers: `loss = logsumexp(scores) - scores[label]`. This is what `from_logits=True` computes.

With `from_logits=False`, Keras assumes that the numbers are probabilities already. It clips each value to the range from `epsilon` to `1 - epsilon`, takes the log, and then applies the same stable formula as before. If the numbers are raw scores, the result has no meaning.

A negative score becomes `epsilon`, and the model cannot see how negative it was. The result is still a number, and the optimizer lowers it.

The next block computes both losses by hand and compares them with Keras.

```python
import keras
import numpy as np
from keras import ops

rng = np.random.default_rng(0)
scores = rng.normal(size=(8, 10)).astype("float32")
labels = rng.integers(0, 10, size=(8,))
rows = np.arange(8)

shifted = scores - scores.max(axis=1, keepdims=True)
log_sum_exp = np.log(np.exp(shifted).sum(axis=1)) + scores.max(axis=1)
correct = float(np.mean(log_sum_exp - scores[rows, labels]))

epsilon = keras.backend.epsilon()
clipped = np.clip(scores, epsilon, 1 - epsilon)
wrong = float(np.mean(np.log(clipped.sum(axis=1)) - np.log(clipped[rows, labels])))

with_logits = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, scores)
without = keras.losses.SparseCategoricalCrossentropy(from_logits=False)(labels, scores)
print("from_logits=True : by hand %.4f, Keras %.4f" % (correct, float(ops.convert_to_numpy(with_logits))))
print("from_logits=False: by hand %.4f, Keras %.4f" % (wrong, float(ops.convert_to_numpy(without))))
```

```text
from_logits=True : by hand 3.0124, Keras 3.0124
from_logits=False: by hand 13.2655, Keras 13.2655
```

Both formulas match Keras. The second one is a number, and it is not cross-entropy. The invariant is this: the loss defines the direction of every step. A wrong flag gives a valid loss for the wrong problem. The optimizer follows it without a complaint.

This is the reason that the model in this chapter trained to 0.5711 and not to 0.9711.

## Reference

```python
import keras
from examples.signatures import show

show(keras.losses.SparseCategoricalCrossentropy, keras.optimizers.SGD, keras.optimizers.Adam,
     keras.optimizers.schedules.CosineDecay, keras.metrics.SparseTopKCategoricalAccuracy)
```

```text
SparseCategoricalCrossentropy(from_logits=False, ignore_class=None,
    reduction='sum_over_batch_size', axis=-1,
    name='sparse_categorical_crossentropy', dtype=None)
SGD(learning_rate=0.01, momentum=0.0, nesterov=False, weight_decay=None,
    clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False,
    ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None,
    gradient_accumulation_steps=None, name=None, **kwargs)
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    amsgrad=False, weight_decay=None, clipnorm=None, clipvalue=None,
    global_clipnorm=None, use_ema=False, ema_momentum=0.99,
    ema_overwrite_frequency=None, loss_scale_factor=None,
    gradient_accumulation_steps=None, name=None, **kwargs)
CosineDecay(initial_learning_rate, decay_steps, alpha=0.0, name='CosineDecay',
    warmup_target=None, warmup_steps=0)
SparseTopKCategoricalAccuracy(k=5, name='sparse_top_k_categorical_accuracy',
    dtype=None, from_sorted_ids=False)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `SparseCategoricalCrossentropy(from_logits)` | `False` | the output is raw scores, not probabilities | `False` on raw scores gave 0.5711 and no warning |
| `SparseCategoricalCrossentropy(reduction)` | `'sum_over_batch_size'` | how the per-sample values combine | — |
| `SGD(learning_rate)` | `0.01` | size of the step | 1.5 gave `nan` on the line (Chapter 1) |
| `SGD(momentum)` | `0.0` | share of the previous step that carries over | 0.9 reached 0.913 after one epoch, against 0.716 |
| `Adam(learning_rate)` | `0.001` | base size of the step | too small for the line in 100 epochs |
| `Adam(epsilon)` | `1e-07` | floor of the divisor in the update | — |
| `clipnorm`, `clipvalue`, `global_clipnorm` | `None` | three kinds of gradient clipping | at a rate of 2.0 only `global_clipnorm` rescued the run |
| `CosineDecay(decay_steps)` | required | steps over which the rate falls to `alpha` times the start | — |
| `SparseTopKCategoricalAccuracy(k)` | `5` | the label counts if it is in the best `k` | the default of 5 is high for 10 classes |

## What you have now

You have a classifier that reaches 0.9711, and a short list of measured choices. Use integer labels and `from_logits=True`. Do not rank optimizers by a single final number. Add a schedule or clipping only when a run shows the need.

[Chapter 7](07-callbacks-and-training-control.md) shows how to stop a run, save the best weights and react to what happens during training. Exercises for this chapter are in [exercises/06.md](../exercises/06.md), and the worked solutions are in [exercises/solutions/06.md](../exercises/solutions/06.md).
