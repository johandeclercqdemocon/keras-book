# Chapter 5 — Data pipelines

> **Build** — three input pipelines for one data set, a `Normalization` layer that travels inside the model, and four measured traps: a validation split that sees one class, an optimizer that cannot cope with raw pixels, a constant pixel that explodes, and a pipeline that ignores `shuffle`.
>
> **Theory kept** — the training loop sees batches. How the batches are made decides what the model learns. Scale, order and leakage are properties of the pipeline, not of the model.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch05_data_pipelines`, which is [`examples/ch05_data_pipelines.py`](../examples/ch05_data_pipelines.py).

A model is only as good as the batches that reach it. Most first failures in a Keras project are not model failures. They are pipeline failures: the wrong scale, the wrong order, or a statistic that leaked from the test data.

This chapter uses one data set for the next four chapters. It is the digits data that ships with scikit-learn. It has 1,797 images of 8 × 8 pixels with values from 0 to 16. The data are on your disk, so nothing here needs a network.

## The data

The module [`examples/digits.py`](../examples/digits.py) loads the data and makes one fixed split. The split is stratified: each class has the same share in the training set and in the test set. The first block looks at the raw data.

```python
import numpy as np
from sklearn.datasets import load_digits

data = load_digits()
x, y = data.data, data.target
print("shape:", x.shape, "pixel range:", x.min(), "to", x.max())
print("images per class:", np.bincount(y).tolist())
print("first 12 labels:", y[:12].tolist())

from examples.digits import load
x_train, x_test, y_train, y_test = load()
print("train:", x_train.shape, "test:", x_test.shape)
```

```text
shape: (1797, 64) pixel range: 0.0 to 16.0
images per class: [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]
first 12 labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
train: (1347, 64) test: (450, 64)
```

There are 1,797 images with 64 pixels each. The classes are balanced: each digit has between 174 and 183 images. The labels in the file repeat the pattern 0 to 9, so they are not sorted. This detail matters in the next section.

## The validation split takes the last rows

`fit` has an argument `validation_split`. It holds back the last fraction of the arrays as validation data. It does this before it shuffles. If the class order of the rows is sorted, the validation set contains only the last classes.

The next block sorts the same data by label and trains the same model twice. The first run uses the sorted rows. The second run shuffles the rows first.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
order = np.argsort(y_train, kind="stable")
shuffled = np.random.default_rng(0).permutation(len(y_train))

def validation_accuracy(x, y):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    history = model.fit(x, y, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    return history.history["accuracy"][-1], history.history["val_accuracy"][-1]

for name, rows in [("sorted by class", order), ("shuffled", shuffled)]:
    train_acc, val_acc = validation_accuracy(x_train[rows], y_train[rows])
    print(f"{name:16s} train accuracy {train_acc:.3f}  validation accuracy {val_acc:.3f}")
```

```text
sorted by class  train accuracy 0.974  validation accuracy 0.015
shuffled         train accuracy 0.945  validation accuracy 0.904
```

With sorted rows, the validation set holds the last classes. The model never saw them in training, so it scores close to zero. The training accuracy is high and the validation accuracy is near zero. This looks like extreme overfitting. It is a pipeline fault.

Shuffle the rows before you call `fit` with `validation_split`. Better, make the validation set yourself with a stratified split, as `examples/digits.py` does.

## The scale of the input

The next block trains the same model on the same data with three input treatments: raw pixels from 0 to 16, pixels divided by 16, and a `Normalization` layer inside the model. The layer learns the mean and the variance of each pixel with `adapt`. It then subtracts the mean and divides by the standard deviation.

Each treatment runs with two optimizers.

```python
import keras
from keras import layers
from examples.digits import load

x_raw, xt_raw, y_train, y_test = load(scale=False)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def run(optimizer, treatment):
    keras.utils.set_random_seed(0)
    first = []
    x_train, x_test = x_raw, xt_raw
    if treatment == "divide by 16":
        x_train, x_test = x_raw / 16, xt_raw / 16
    elif treatment == "Normalization":
        norm = layers.Normalization()
        norm.adapt(x_raw)
        first = [norm]
    model = keras.Sequential([keras.Input((64,)), *first,
                              layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile(optimizer, loss, metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                        validation_data=(x_test, y_test), verbose=0)
    return history.history["val_accuracy"][-1]

for optimizer_name, make in [("SGD(0.1)", lambda: keras.optimizers.SGD(0.1)),
                             ("Adam(0.001)", lambda: keras.optimizers.Adam(0.001))]:
    scores = [run(make(), t) for t in ("raw", "divide by 16", "Normalization")]
    print(f"{optimizer_name:12s} raw {scores[0]:.3f}  divide by 16 {scores[1]:.3f}  Normalization {scores[2]:.3f}")
```

```text
SGD(0.1)     raw 0.296  divide by 16 0.962  Normalization 0.964
Adam(0.001)  raw 0.962  divide by 16 0.951  Normalization 0.958
```

Read the table by rows. For SGD, raw pixels give an accuracy of 0.296 and both scaled versions give 0.96. For Adam, all three are between 0.951 and 0.962. The test set has 450 images, and one image is 0.0022. The three Adam results differ by 5 images at most, so Adam shows no effect of scale here.

The reason is how the two optimizers use the gradient. SGD steps by the gradient, so a large input gives a large step. Adam divides each step by a running estimate of the size of the gradient, so it removes most of the scale. Adam hides the scale problem. It does not remove it. A model with a different layer, a different loss or a different optimizer can show it again.

## A constant pixel can explode

The `Normalization` layer divides by a standard deviation. A pixel that is constant in the training data has a standard deviation of zero. What does the layer do with it?

The next block adapts the layer on the training images and looks for the largest output on the test images.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

x_train, x_test, y_train, y_test = load(scale=False)
constant = np.where(x_train.std(axis=0) == 0)[0]
print("pixels that are constant in training:", constant.tolist())
print("test images with a non-zero value there:",
      int((x_test[:, constant] != 0).any(axis=1).sum()))

norm = layers.Normalization()
norm.adapt(x_train)
out = ops.convert_to_numpy(norm(x_test))
other = np.delete(out, constant, axis=1)
print("largest output on the other pixels:", round(float(np.abs(other).max()), 1))
print("largest output on the constant pixels:", float(np.abs(out[:, constant]).max()))
```

```text
pixels that are constant in training: [0, 24, 32, 39]
test images with a non-zero value there: 2
largest output on the other pixels: 88.6
largest output on the constant pixels: 10000000.0
```

Four pixels are constant in the training set. In the test set, two images have a value of 1 in one of them. The layer divides that difference by a number close to zero, and the output is 10,000,000. The next layer receives a value more than 100,000 times larger than the largest of the other inputs. It does not raise an error, and the value is finite.

The cause is in the data, not in Keras. A pixel that never changes carries no information, so do not give it to the layer. Check the minimum of the standard deviation before you adapt. Remove constant features, or add a floor to the variance.

## Three ways to feed `fit`

Until now the data were NumPy arrays. Arrays are the simplest input, and they fit in memory. When the data do not fit, or when each batch needs work, there are two other inputs.

A `keras.utils.PyDataset` is a Python class with `__len__` and `__getitem__`. Keras asks it for batch `i`, and the class returns the arrays. A `tf.data.Dataset` is a pipeline object of TensorFlow. Keras 3 accepts it on every back end.

The next block trains the same model through all three inputs for the same 10 epochs with batch size 32.

```python
import warnings

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class Digits(keras.utils.PyDataset):
    def __init__(self, x, y, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.x, self.y, self.batch_size = x, y, batch_size
        self.rng = np.random.default_rng(0)
        self.order = self.rng.permutation(len(x))

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        rows = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        return self.x[rows], self.y[rows]

    def on_epoch_end(self):
        self.rng.shuffle(self.order)

def fresh_model():
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile("adam", loss, metrics=["accuracy"])
    return model

validation = (x_test, y_test)
pydataset = Digits(x_train, y_train, 32)
print("batches per epoch:", len(pydataset), "last batch shape:", pydataset[len(pydataset) - 1][0].shape)

h = fresh_model().fit(x_train, y_train, epochs=10, batch_size=32, validation_data=validation, verbose=0)
print("NumPy arrays :", round(h.history["val_accuracy"][-1], 3))
h = fresh_model().fit(pydataset, epochs=10, validation_data=validation, verbose=0)
print("PyDataset    :", round(h.history["val_accuracy"][-1], 3))

pipeline = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(len(x_train), seed=0).batch(32).prefetch(tf.data.AUTOTUNE))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    h = fresh_model().fit(pipeline, epochs=10, validation_data=validation, verbose=0)
print("tf.data      :", round(h.history["val_accuracy"][-1], 3))
print("warning:", str(caught[0].message)[:62])
```

```text
batches per epoch: 43 last batch shape: (3, 64)
NumPy arrays : 0.951
PyDataset    : 0.949
tf.data      : 0.944
warning: `shuffle=True` was passed, but will be ignored since the data
```

The three accuracies are within 0.007 of each other, which is 3 images of 450. The pipelines are equal for this purpose. The last batch of the `PyDataset` has 3 images, because 1,347 is not a multiple of 32. Keras uses the short batch.

The warning is useful. `fit` shuffles arrays by default, with `shuffle=True`. It cannot shuffle a `tf.data.Dataset`, so it ignores the argument and says so. The pipeline must shuffle for itself, with `.shuffle(buffer_size)`. A pipeline without a shuffle sends the batches in file order in every epoch.

## How it works

`Normalization.adapt` is not training. It makes one pass over the data and stores three numbers for each feature: the mean, the variance and the count. They are weights of the layer, but they are not trainable, and no gradient touches them. The layer then computes `(x - mean) / max(sqrt(variance), epsilon)`, where `epsilon` is the Keras back end constant.

The constant is the cause of the 10,000,000 in this chapter. A pixel with zero variance has a standard deviation of 0, so the divisor becomes `epsilon`, which is 1e-7. A value of 1 in that pixel gives 1 / 1e-7 = 10,000,000.

The next block checks the formula against the layer, and it prints the constant.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

x_train, x_test, _, _ = load(scale=False)
layer = layers.Normalization()
layer.adapt(x_train)

mean, variance = x_train.mean(axis=0), x_train.var(axis=0)
epsilon = keras.backend.epsilon()
by_hand = (x_test - mean) / np.maximum(np.sqrt(variance), epsilon)
result = ops.convert_to_numpy(layer(x_test))

print("epsilon:", epsilon)
print("stored mean equals the NumPy mean:", bool(np.allclose(ops.convert_to_numpy(layer.mean)[0], mean, atol=1e-5)))
print("formula equals the layer:", bool(np.allclose(result, by_hand, rtol=1e-4, atol=1e-4)))
print("largest value of the formula:", float(np.abs(by_hand).max()))
```

```text
epsilon: 1e-07
stored mean equals the NumPy mean: True
formula equals the layer: True
largest value of the formula: 10000000.0
```

The invariant is this: after `adapt`, the layer is a fixed affine map. The statistics come from the data that you give to `adapt`. They stay fixed for every later call.

Two consequences follow. Adapt on the training set only, or the test set leaks into the model. Check for zero variance before you adapt, because the divisor has a floor and not a guard.

## Reference

```python
import keras
from keras import layers
from examples.signatures import show

show(layers.Normalization, ("Normalization.adapt", layers.Normalization.adapt),
     ("PyDataset", keras.utils.PyDataset.__init__), ("utils.to_categorical", keras.utils.to_categorical),
     ("Model.fit", keras.Model.fit))
```

```text
Normalization(axis=-1, mean=None, variance=None, invert=False, **kwargs)
Normalization.adapt(data)
PyDataset(workers=1, use_multiprocessing=False, max_queue_size=10)
utils.to_categorical(x, num_classes=None)
Model.fit(x=None, y=None, batch_size=None, epochs=1, verbose='auto',
    callbacks=None, validation_split=0.0, validation_data=None, shuffle=True,
    class_weight=None, sample_weight=None, initial_epoch=0,
    steps_per_epoch=None, validation_steps=None, validation_batch_size=None,
    validation_freq=1)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `Normalization(axis)` | `-1` | the axis that keeps its own statistics | the default gives one mean for each pixel here |
| `Normalization(mean, variance)` | `None` | fixed statistics, so no `adapt` is needed | — |
| `adapt(data)` | required | computes mean, variance and count | a constant pixel gives 1e7 on test data |
| `PyDataset(workers)` | `1` | worker threads or processes for loading | — |
| `PyDataset(max_queue_size)` | `10` | batches that wait in the queue | — |
| `fit(validation_split)` | `0.0` | fraction of the last rows held back | sorted rows gave a validation accuracy of 0.015 |
| `fit(shuffle)` | `True` | shuffle each epoch | ignored for `tf.data.Dataset` |
| `to_categorical(num_classes)` | `None` | width of the one-hot vector | `None` takes the largest label plus 1 |

A `PyDataset` subclass must define `__len__` and `__getitem__`. It can define `on_epoch_end`. The base class takes only the three loading arguments that the table lists.

## What you have now

You have three input pipelines that give the same result. You also have four traps, each with a number. Sorted rows give a validation accuracy of 0.015. SGD on raw pixels gives an accuracy of 0.296. A constant pixel gives an output of 10,000,000. A `shuffle` argument can do nothing and say so only in a warning.

[Chapter 6](06-losses-metrics-optimizers.md) uses this data to choose a loss, a metric and an optimizer. Exercises for this chapter are in [exercises/05.md](../exercises/05.md), and the worked solutions are in [exercises/solutions/05.md](../exercises/solutions/05.md).
