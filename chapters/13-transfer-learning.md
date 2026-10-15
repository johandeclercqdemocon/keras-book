# Chapter 13 — Transfer learning

> **Build** — a base network that was trained on clean digits and reused on noisy digits with 2, 5 and 20 labeled images for each class, in three modes: from scratch, with a frozen base and with a fine-tuned base. Three training budgets show that the budget decides the sign of the comparison.
>
> **Theory kept** — the first layers of a trained network are a feature extractor. A new head can learn on top of them with few labels. Freezing keeps the features. Fine-tuning moves them.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch13_transfer_learning`, which is [`examples/ch13_transfer_learning.py`](../examples/ch13_transfer_learning.py).

Labels are expensive. A network that learned on a large labeled set can pass what it learned to a task with few labels. This chapter measures how much it passes, and how that depends on one choice that is easy to get wrong.

## The setup

The base task is the digit classification of the earlier chapters. The new task is the same ten classes, with noise added to every pixel. The noise has a standard deviation of 0.35 on pixels in the range 0 to 1. A model that was trained on clean images scores lower on noisy ones. The target set has 2, 5 or 20 labeled noisy images for each class. The test set is all 450 noisy test images.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load(images=True)
noise = np.random.default_rng(0)

def add_noise(x, deviation=0.35):
    return np.clip(x + noise.normal(0, deviation, x.shape), 0, 1).astype("float32")

x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)

keras.utils.set_random_seed(0)
base = keras.Sequential([
    keras.Input((8, 8, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.GlobalMaxPooling2D(),
], name="base")
pretrained = keras.Sequential([base, layers.Dense(10)])
pretrained.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
pretrained.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

print("base output shape:", base.output_shape, " base weights:", base.count_params())
print("trained on clean, test clean:", round(pretrained.evaluate(x_test, y_test, verbose=0)[1], 3))
print("trained on clean, test noisy:", round(pretrained.evaluate(x_test_noisy, y_test, verbose=0)[1], 3))
```

```text
base output shape: (None, 32)  base weights: 4800
trained on clean, test clean: 0.991
trained on clean, test noisy: 0.673
```

The base turns each image into 32 numbers. The model reaches 0.991 on clean test images and falls on noisy ones. The question is whether the 32-number features help a new model that has only a few noisy labels.

## Three modes

The code below defines one function that trains a new head in each of the three modes. It runs the function for several sizes of the target set.

- **scratch**: a new network with the same layers and random weights.
- **frozen**: a copy of the base with `trainable = False`, and a new `Dense(10)` head.
- **fine-tune**: a copy of the base with all weights trainable, and a new head.

The copy is made with `keras.models.clone_model` and `set_weights`, so that the original base is not changed. The model is compiled after `trainable` is set, as [Chapter 3](03-layers-and-weights.md) requires.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load(images=True)
noise = np.random.default_rng(0)

def add_noise(x, deviation=0.35):
    return np.clip(x + noise.normal(0, deviation, x.shape), 0, 1).astype("float32")

x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def convolutions():
    return [layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.GlobalMaxPooling2D()]

keras.utils.set_random_seed(0)
base = keras.Sequential([keras.Input((8, 8, 1))] + convolutions(), name="base")
pretrained = keras.Sequential([base, layers.Dense(10)])
pretrained.compile("adam", loss, metrics=["accuracy"])
pretrained.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

def new_model(mode):
    if mode == "scratch":
        body = keras.Sequential([keras.Input((8, 8, 1))] + convolutions())
    else:
        body = keras.models.clone_model(base)
        body.set_weights(base.get_weights())
        body.trainable = mode == "fine-tune"
    return keras.Sequential([body, layers.Dense(10)])

def accuracy(per_class, mode, learning_rate, epochs, seed):
    rows = np.concatenate([
        np.random.default_rng(seed).choice(np.where(y_train == c)[0], per_class, replace=False)
        for c in range(10)])
    keras.utils.set_random_seed(seed)
    model = new_model(mode)
    model.compile(keras.optimizers.Adam(learning_rate), loss, metrics=["accuracy"])
    model.fit(x_train_noisy[rows], y_train[rows], epochs=epochs, batch_size=16, verbose=0)
    return model.evaluate(x_test_noisy, y_test, verbose=0)[1]

def table(learning_rate, epochs):
    print(f"learning rate {learning_rate}, {epochs} epochs; mean of 2 seeds")
    print(f"{'images per class':>17s} {'scratch':>8s} {'frozen':>8s} {'fine-tune':>10s}")
    for per_class in (2, 5, 20):
        scores = [np.mean([accuracy(per_class, mode, learning_rate, epochs, seed) for seed in (1, 2)])
                  for mode in ("scratch", "frozen", "fine-tune")]
        print(f"{per_class:17d} {scores[0]:8.3f} {scores[1]:8.3f} {scores[2]:10.3f}")

table(0.001, 40)
```

```text
# output varies
learning rate 0.001, 40 epochs; mean of 2 seeds
 images per class  scratch   frozen  fine-tune
                2    0.131    0.070      0.102
                5    0.157    0.114      0.156
               20    0.592    0.291      0.528
```

The tables of this chapter are marked as output that varies. Between two runs of the same code, a cell moved by one image of 450. In a mean of 2 seeds that is 0.001 to 0.002. Read the third decimal with that in mind.

Read the table by rows. The result is not what transfer learning promises. At this budget, the frozen base does not beat the new network that starts from scratch. At 20 images for each class the frozen head is far below the other two. If you stopped here, you will conclude that transfer does not help. The next two sections show that this conclusion depends on the budget.

The copy used three callables. `get_weights` and `set_weights` have their blocks in [Chapter 3](03-layers-and-weights.md), with the attribute `trainable`. `keras.optimizers.Adam` is in [Chapter 6](06-losses-metrics-optimizers.md).

#### `keras.models.clone_model`

```text
keras.models.clone_model(
    model,
    input_tensors=None,
    clone_function=None,
    call_function=None,
    recursive=False,
    **kwargs,
)
```

The function returns a new model with the same architecture. The new layers get new weights from their initializers.

The table under `keras.models.clone_model` in the Reference section of this chapter gives each parameter and what goes wrong with it.

## A different budget reverses the picture

The first table has a hypothesis behind it that you can test. The frozen head is a single `Dense` layer on 32 inputs. With 200 images and a batch size of 16, one epoch is 13 steps. Then 40 epochs is 520 steps at a rate of 0.001. The hypothesis is that this is too little for the head. The next block gives the same three modes a learning rate of 0.01 and 100 epochs.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load(images=True)
noise = np.random.default_rng(0)

def add_noise(x, deviation=0.35):
    return np.clip(x + noise.normal(0, deviation, x.shape), 0, 1).astype("float32")

x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def convolutions():
    return [layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.GlobalMaxPooling2D()]

keras.utils.set_random_seed(0)
base = keras.Sequential([keras.Input((8, 8, 1))] + convolutions(), name="base")
pretrained = keras.Sequential([base, layers.Dense(10)])
pretrained.compile("adam", loss, metrics=["accuracy"])
pretrained.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

def new_model(mode):
    if mode == "scratch":
        body = keras.Sequential([keras.Input((8, 8, 1))] + convolutions())
    else:
        body = keras.models.clone_model(base)
        body.set_weights(base.get_weights())
        body.trainable = mode == "fine-tune"
    return keras.Sequential([body, layers.Dense(10)])

def accuracy(per_class, mode, learning_rate, epochs, seed):
    rows = np.concatenate([
        np.random.default_rng(seed).choice(np.where(y_train == c)[0], per_class, replace=False)
        for c in range(10)])
    keras.utils.set_random_seed(seed)
    model = new_model(mode)
    model.compile(keras.optimizers.Adam(learning_rate), loss, metrics=["accuracy"])
    model.fit(x_train_noisy[rows], y_train[rows], epochs=epochs, batch_size=16, verbose=0)
    return model.evaluate(x_test_noisy, y_test, verbose=0)[1]

print("learning rate 0.01, 100 epochs; mean of 2 seeds")
print(f"{'images per class':>17s} {'scratch':>8s} {'frozen':>8s} {'fine-tune':>10s}")
for per_class in (2, 5, 20):
    scores = [np.mean([accuracy(per_class, mode, 0.01, 100, seed) for seed in (1, 2)])
              for mode in ("scratch", "frozen", "fine-tune")]
    print(f"{per_class:17d} {scores[0]:8.3f} {scores[1]:8.3f} {scores[2]:10.3f}")
```

```text
# output varies
learning rate 0.01, 100 epochs; mean of 2 seeds
 images per class  scratch   frozen  fine-tune
                2    0.219    0.419      0.324
                5    0.378    0.568      0.469
               20    0.592    0.724      0.653
```

At this budget the picture is the opposite. The frozen base is ahead of the new network in every row: by about 0.20 at 2 images for each class, by about 0.19 at 5 and by about 0.13 at 20. Fine-tuning is in between.

The first table did not measure transfer. It measured a head that had not finished training. The mistake is easy to make, because nothing fails: the loss falls, the run ends, and the number is low. A frozen base needs a budget for the head. The first test of a negative result is to ask whether the new part converged.

## Every mode depends on its budget

The second table raises a fair question. If the head needed a larger budget, did the new network need one too? The next block runs a third budget, a learning rate of 0.003 and 60 epochs, for all three modes.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load(images=True)
noise = np.random.default_rng(0)

def add_noise(x, deviation=0.35):
    return np.clip(x + noise.normal(0, deviation, x.shape), 0, 1).astype("float32")

x_train_noisy, x_test_noisy = add_noise(x_train), add_noise(x_test)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def convolutions():
    return [layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.GlobalMaxPooling2D()]

keras.utils.set_random_seed(0)
base = keras.Sequential([keras.Input((8, 8, 1))] + convolutions(), name="base")
pretrained = keras.Sequential([base, layers.Dense(10)])
pretrained.compile("adam", loss, metrics=["accuracy"])
pretrained.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

def new_model(mode):
    if mode == "scratch":
        body = keras.Sequential([keras.Input((8, 8, 1))] + convolutions())
    else:
        body = keras.models.clone_model(base)
        body.set_weights(base.get_weights())
        body.trainable = mode == "fine-tune"
    return keras.Sequential([body, layers.Dense(10)])

def accuracy(per_class, mode, learning_rate, epochs, seed):
    rows = np.concatenate([
        np.random.default_rng(seed).choice(np.where(y_train == c)[0], per_class, replace=False)
        for c in range(10)])
    keras.utils.set_random_seed(seed)
    model = new_model(mode)
    model.compile(keras.optimizers.Adam(learning_rate), loss, metrics=["accuracy"])
    model.fit(x_train_noisy[rows], y_train[rows], epochs=epochs, batch_size=16, verbose=0)
    return model.evaluate(x_test_noisy, y_test, verbose=0)[1]

print("learning rate 0.003, 60 epochs; mean of 2 seeds")
print(f"{'images per class':>17s} {'scratch':>8s} {'frozen':>8s} {'fine-tune':>10s}")
for per_class in (2, 5, 20):
    scores = [np.mean([accuracy(per_class, mode, 0.003, 60, seed) for seed in (1, 2)])
              for mode in ("scratch", "frozen", "fine-tune")]
    print(f"{per_class:17d} {scores[0]:8.3f} {scores[1]:8.3f} {scores[2]:10.3f}")
```

```text
# output varies
learning rate 0.003, 60 epochs; mean of 2 seeds
 images per class  scratch   frozen  fine-tune
                2    0.244    0.166      0.229
                5    0.358    0.294      0.386
               20    0.652    0.653      0.653
```

The third budget gives a third picture. At 20 images for each class the three modes tie, at 0.652 to 0.653. At 2 images the new network is ahead, with 0.244 against 0.166. At 5 images fine-tuning is ahead, with 0.386. The sign of the comparison changes with the budget, inside one table and between tables.

A claim that transfer helps, or that it does not help, needs the budget stated, and it needs a search over budgets for every mode. Take the best entry of each mode at 20 images. Scratch reaches 0.652, the frozen base 0.724 and fine-tuning 0.653. The frozen base is highest. It is ahead of the next best entry, 0.653, by about 0.07, which is about 30 images of 450.

Two limits apply to that number. We chose the budgets by looking at the test accuracy, so each best entry is optimistic. Each row is the mean of 2 seeds. The chapter did not measure the spread over seeds, so it cannot say that 0.07 is outside it. Read it as a lead and not as a result.

## How it works

Freezing does two things. It moves the weights of the base to the non-trainable list, so that the optimizer skips them. It also stops the base from changing in any other way: there are no batch statistics in this base, so its output for an input is a fixed function.

The next block measures both. It counts the trainable weights in each mode. Then it checks the base weights after a short training. They must not change in frozen mode, and they must change in fine-tune mode.

```python
import keras
import numpy as np
from keras import layers

def convolutions():
    return [layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.GlobalMaxPooling2D()]

keras.utils.set_random_seed(0)
base = keras.Sequential([keras.Input((8, 8, 1))] + convolutions())
x = np.random.default_rng(0).random((32, 8, 8, 1)).astype("float32")
y = np.random.default_rng(1).integers(0, 10, size=(32,))

for mode in ("frozen", "fine-tune"):
    body = keras.models.clone_model(base)
    body.set_weights(base.get_weights())
    body.trainable = mode == "fine-tune"
    model = keras.Sequential([body, layers.Dense(10)])
    model.compile(keras.optimizers.Adam(0.01), keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    before = [w.copy() for w in body.get_weights()]
    trainable = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    model.fit(x, y, epochs=3, batch_size=16, verbose=0)
    moved = any(not np.array_equal(a, b) for a, b in zip(before, body.get_weights()))
    print(f"{mode:9s} trainable weights {trainable:5d} of {model.count_params()}, base weights changed: {moved}")
```

```text
frozen    trainable weights   330 of 5130, base weights changed: False
fine-tune trainable weights  5130 of 5130, base weights changed: True
```

In frozen mode, only the head trains: 330 weights of 5,130. The base is a fixed function. In fine-tune mode, all of them train.

The invariant is this: a frozen base gives the same 32 numbers for the same image in every epoch. The head then solves a small problem on 32 inputs. A small problem needs fewer examples.

This is the reason that transfer can work with few labels. It is also the reason that its result depends on how well the head is trained.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `keras.models.clone_model` | Three modes | the same architecture with new weights |

`trainable`, `get_weights` and `set_weights` are in [Chapter 3](03-layers-and-weights.md), and `keras.optimizers.Adam` is in [Chapter 6](06-losses-metrics-optimizers.md).

### Parameters

##### `keras.models.clone_model`

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `model` | required | the model to copy | the weights are not copied: the clone of the base had other weights. Copy them with `set_weights`, as this chapter does |
| `input_tensors` | `None` | new inputs for the clone | — |
| `clone_function` | `None` | a function that makes the new layer from each old layer | a function that returns its argument gives the same layer objects, and the clone then shares its weights with the original |
| `call_function` | `None` | a function that calls each new layer, for a functional model | — |
| `recursive` | `False` | apply `clone_function` inside nested models too | with `False`, the function saw the nested model `base` and `dense`. With `True`, it saw `conv2d`, `global_max_pooling2d` and `dense` |
| `**kwargs` | — | no other keyword is accepted | `foo=1` raises `ValueError: Unexpected keyword argument(s): ('foo',)` |


## What you have now

You have a measured case in which transfer learning wins, a measured case in which it seems to lose, and a test for the difference. Each mode needs a budget that suits it, and a result that is negative is a reason to check the budget before you conclude.

[Chapter 14](14-saving-loading-export.md) saves what you built. Exercises for this chapter are in [exercises/13.md](../exercises/13.md), and the worked solutions are in [exercises/solutions/13.md](../exercises/solutions/13.md).
