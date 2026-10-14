# Chapter 8 — Overfitting and regularization

> **Build** — a table that compares dropout, an L2 penalty, batch normalization and input noise on a model that memorizes 200 images, a learning curve over six training-set sizes, and a measurement of what the validation loss contains when a penalty is present.
>
> **Theory kept** — overfitting is a gap between the score on the training data and the score on new data. A regularizer shrinks the gap. It does not always raise the score on new data.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch08_overfitting_and_regularization`, which is [`examples/ch08_overfitting_and_regularization.py`](../examples/ch08_overfitting_and_regularization.py).

A model overfits when it learns the training examples and not the rule behind them. The signal is a gap: the accuracy on the training data is high, and the accuracy on data that the model did not see is lower. [Chapter 7](07-callbacks-and-training-control.md) used `EarlyStopping` to stop at the point where the gap starts to hurt. This chapter asks what else you can do.

The setup is the one of Chapter 7. The training set is 200 images, and the model has two hidden layers of 256 units. This model has about 85,000 weights for 200 examples. It can memorize the set.

## Four regularizers, one table

The next block trains the same model with each regularizer and prints the training accuracy, the test accuracy and the gap between them. The four regularizers are:

- **Dropout** sets a random half of the activations to zero at each training step.
- **An L2 penalty** adds the sum of the squared weights, times a factor, to the loss.
- **Batch normalization** rescales the activations of each batch.
- **Gaussian noise** adds random noise to the input at each training step.

Every setting uses the same seed and 60 epochs.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
rows = np.random.default_rng(0).choice(len(x_train), 200, replace=False)
x_small, y_small = x_train[rows], y_train[rows]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def build(dropout=0.0, l2=0.0, batch_norm=False, noise=0.0):
    keras.utils.set_random_seed(0)
    regularizer = keras.regularizers.L2(l2) if l2 else None
    stack = [keras.Input((64,))]
    if noise:
        stack.append(layers.GaussianNoise(noise))
    for _ in range(2):
        stack.append(layers.Dense(256, kernel_regularizer=regularizer))
        if batch_norm:
            stack.append(layers.BatchNormalization())
        stack.append(layers.Activation("relu"))
        if dropout:
            stack.append(layers.Dropout(dropout))
    stack.append(layers.Dense(10))
    model = keras.Sequential(stack)
    model.compile("adam", loss, metrics=["accuracy"])
    return model

def run(**options):
    model = build(**options)
    h = model.fit(x_small, y_small, epochs=60, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0).history
    return h

settings = {
    "baseline": {},
    "dropout 0.5": dict(dropout=0.5),
    "L2 1e-3": dict(l2=1e-3),
    "batch norm": dict(batch_norm=True),
    "noise 0.3": dict(noise=0.3),
}
print(f"{'setting':14s} {'train acc':>9s} {'val acc':>8s} {'gap':>6s}")
for name, options in settings.items():
    h = run(**options)
    train, val = h["accuracy"][-1], h["val_accuracy"][-1]
    print(f"{name:14s} {train:9.3f} {val:8.3f} {train - val:6.3f}")
```

```text
setting        train acc  val acc    gap
baseline           1.000    0.933  0.067
dropout 0.5        0.980    0.940  0.040
L2 1e-3            1.000    0.929  0.071
batch norm         1.000    0.927  0.073
noise 0.3          0.935    0.933  0.002
```

Read the table by columns. The baseline reaches a training accuracy of 1.000 and a test accuracy of 0.933, and the gap is 0.067. Dropout 0.5 lowers the gap to 0.040. Input noise lowers it to 0.002. The L2 penalty and batch normalization leave the gap near the baseline or slightly above.


Now read the test accuracy column alone. It runs from 0.927 to 0.940. The test set has 450 images, and one image is 0.0022. The whole spread is 6 images.

All four settings are within 3 images of the baseline. Input noise has the smallest gap and the same test accuracy as the baseline. It lowered the training accuracy to 0.935 and did not raise the test accuracy.

A regularizer shrinks the gap. It does not always move the test score. Judge a regularizer by the test score, not by the gap.

The block used three callables that have their blocks here. `BatchNormalization` has its block in [Chapter 3](03-layers-and-weights.md).

#### `keras.layers.Dropout`

```text
keras.layers.Dropout(
    rate,
    noise_shape=None,
    seed=None,
    **kwargs,
)
```

In training, the layer sets a random share `rate` of its inputs to zero and multiplies the others by `1 / (1 - rate)`. In inference, it returns its input.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `rate` | required | the share of inputs to set to zero, from 0 to 1 | — (with 0.5 on 10,000 ones, 0.4968 were kept, each with the value 2.0, and the mean stayed 0.9936). `rate=1.0` is accepted and gives only zeros. 1.5 raises `ValueError` |
| `noise_shape` | `None` | the shape of the mask. A 1 shares one value along that axis | with `(4, 1)` on a batch of 4 rows of 6, each row was all zeros or all 2.0 |
| `seed` | `None` | the seed of the generator of the layer | two layers with `seed=1` drew the same mask. The second call of one layer drew a new mask |
| `**kwargs` | — | the arguments of `Layer`, such as `name` | — |

#### `keras.regularizers.L2`

```text
keras.regularizers.L2(
    l2=0.01,
)
```

The regularizer adds `l2 * sum(w ** 2)` to the loss. It does not divide by 2.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `l2` | `0.01` | the factor | `L2()` on 10 weights of value 1 gave 0.1. A negative factor raises `ValueError: Invalid value for argument l2: expected a non-negative float` |

#### `keras.layers.GaussianNoise`

```text
keras.layers.GaussianNoise(
    stddev,
    seed=None,
    **kwargs,
)
```

In training, the layer adds noise from a normal distribution with the mean 0. In inference, it returns its input.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `stddev` | required | the standard deviation of the noise | the value must be from 0 to 1: 2.0 raises `ValueError`. On raw pixels from 0 to 16, scale the input first. 0.3 gave a measured standard deviation of 0.3001 in training and no noise in inference |
| `seed` | `None` | the seed of the generator of the layer | — |
| `**kwargs` | — | the arguments of `Layer`, such as `name` | — |

## More data beats every regularizer

The lever that always works is more data. The next block trains one small model on growing subsets of the training images and prints both accuracies.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

print(f"{'training images':>15s} {'train acc':>10s} {'test acc':>9s}")
for n in (50, 100, 200, 400, 800, 1347):
    rows = np.random.default_rng(0).choice(len(x_train), n, replace=False)
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    model.compile("adam", loss, metrics=["accuracy"])
    h = model.fit(x_train[rows], y_train[rows], epochs=60, batch_size=32,
                  validation_data=(x_test, y_test), verbose=0).history
    print(f"{n:15d} {h['accuracy'][-1]:10.3f} {h['val_accuracy'][-1]:9.3f}")
```

```text
training images  train acc  test acc
             50      0.940     0.720
            100      0.970     0.829
            200      0.975     0.916
            400      0.998     0.938
            800      0.996     0.973
           1347      0.997     0.973
```

The test accuracy rises from 0.720 with 50 images to 0.916 with 200 images and 0.973 with 800 images. No setting of the previous table moved the test accuracy by more than 0.007 at 200 images. Going from 200 to 800 images moved it by 0.057.

The curve also tells you what to do next. Between 800 and 1,347 images the test accuracy does not change, at 0.973. At that size, more of the same data does not help this model. Below 400 images the curve is steep, and collecting data is the best use of time.

## The validation loss contains the penalty

One detail makes it hard to compare runs with and without a penalty. A regularizer adds a term to the loss, and `evaluate` reports the total. The next block trains a model with a strong L2 penalty and splits the reported loss into its two parts.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
rows = np.random.default_rng(0).choice(len(x_train), 200, replace=False)
x_small, y_small = x_train[rows], y_train[rows]

keras.utils.set_random_seed(0)
model = keras.Sequential([
    keras.Input((64,)),
    layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.L2(1e-2)),
    layers.Dense(10),
])
model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_small, y_small, epochs=60, batch_size=32, verbose=0)

reported = model.evaluate(x_test, y_test, verbose=0)[0]
logits = model.predict(x_test, verbose=0)
plain = float(keras.losses.sparse_categorical_crossentropy(y_test, logits, from_logits=True).numpy().mean())
penalty = float(sum(float(np.asarray(w)) for w in model.losses))
print("reported by evaluate:", round(reported, 4))
print("cross-entropy alone :", round(plain, 4))
print("penalty             :", round(penalty, 4))
print("sum of the last two :", round(plain + penalty, 4))
```

```text
reported by evaluate: 0.5433
cross-entropy alone : 0.3435
penalty             : 0.1998
sum of the last two : 0.5433
```

The reported loss is 0.5433. The cross-entropy alone is 0.3435, and the penalty is 0.1998. The sum of the two parts is the reported value. The penalty depends on the weights and not on the data. It adds the same amount to the loss of every batch of test data.

Two runs with different penalties have losses that cannot be compared. Compare the accuracy, or compute the cross-entropy alone as in the block. This also applies to `EarlyStopping(monitor="val_loss")`. The value that it watches contains the penalty. Watch `val_accuracy` or a metric without the penalty when you use a strong regularizer.

## What to do in practice

Use the evidence of the tables in this order.

1. Check the pipeline for leakage and for a bad split ([Chapter 5](05-data-pipelines.md)). A large gap is often a data fault.
2. Add data if you can. The learning curve says how much it is worth.
3. Use early stopping with `restore_best_weights=True`. It costs no extra weights, and it directly targets the best epoch.
4. Try one regularizer at a time, and judge it by the test score with a stated bound. One image of 450 is 0.0022.

## How it works

Dropout in Keras is *inverted* dropout. In training it sets each unit to zero with probability `rate`, and it multiplies the units that stay by `1 / (1 - rate)`. In inference it does nothing. The scale factor keeps the mean of the output the same in both modes. You do not rescale the weights when you switch to inference.

The next block feeds a tensor of ones through dropout with a rate of 0.5.

```python
import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)
dropout = layers.Dropout(0.5)
ones = np.ones((1, 10000), dtype="float32")

training = ops.convert_to_numpy(dropout(ones, training=True))
inference = ops.convert_to_numpy(dropout(ones, training=False))
print("values in training mode:", sorted(set(training.ravel().tolist())))
print("share of units set to zero:", round(float((training == 0).mean()), 3))
print("mean in training mode:", round(float(training.mean()), 3))
print("values in inference mode:", sorted(set(inference.ravel().tolist())))
```

```text
values in training mode: [0.0, 2.0]
share of units set to zero: 0.503
mean in training mode: 0.994
values in inference mode: [1.0]
```

In training the output has two values: 0 and 2. The factor is 1 / (1 - 0.5) = 2. The share of zeros is near 0.5 and the mean is near 1. In inference the output is the input.

The invariant is: *the expected value of the output does not change with the rate.* The variance does change. A unit that is kept is twice as large, and a unit that is dropped is zero, so the next layer sees a noisier input. This noise is the regularizer. It is also a reason that a model with dropout trains more slowly. The training accuracy in the history of `fit` is measured with the noise on, so it is lower than the accuracy in inference.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `keras.layers.Dropout` | Four regularizers, one table | zero a share of the inputs in training, and scale the rest |
| `keras.regularizers.L2` | Four regularizers, one table | `l2 * sum(w ** 2)`, added to the loss |
| `keras.layers.GaussianNoise` | Four regularizers, one table | normal noise in training, with a standard deviation from 0 to 1 |

`keras.layers.BatchNormalization` is in [Chapter 3](03-layers-and-weights.md).

## What you have now

You have a table of four regularizers with their gaps and their scores. You also have a learning curve and the rule that a gap is not a score. The result that matters most is a negative one. On this problem, no regularizer beat the baseline by more than 3 images.

[Chapter 9](09-custom-training.md) opens the training loop itself. Exercises for this chapter are in [exercises/08.md](../exercises/08.md), and the worked solutions are in [exercises/solutions/08.md](../exercises/solutions/08.md).
