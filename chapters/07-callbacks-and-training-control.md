# Chapter 7 — Callbacks and training control

> **Build** — a training run that stops itself when it should, keeps the best weights, lowers its own learning rate and stops on `nan`, and two custom callbacks of ten lines each.
>
> **Theory kept** — a callback is an object that Keras calls at fixed points of the training loop. It can read the logs, and it can change the model or stop the run.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch07_callbacks_and_training_control`, which is [`examples/ch07_callbacks_and_training_control.py`](../examples/ch07_callbacks_and_training_control.py).

[Chapter 1](01-the-first-model.md) showed two runs that finished without an error and gave a bad result. A fixed number of epochs is a guess. A callback replaces the guess with a rule that reads the run.

All the runs in this chapter use the same overfitting setup. The training set is 200 images from the digits data, and the model has two hidden layers of 256 units. This model memorizes 200 images in a few epochs, so the validation loss falls and then rises. That shape is what the callbacks react to.

## Stop when the validation loss stops falling

`EarlyStopping` watches one number in the logs. It stops the run when that number has not improved for `patience` epochs. The argument `restore_best_weights` decides which weights you get back.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
rows = np.random.default_rng(0).choice(len(x_train), 200, replace=False)
x_small, y_small = x_train[rows], y_train[rows]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def overfitting_model(learning_rate=0.001):
    keras.utils.set_random_seed(0)
    model = keras.Sequential([keras.Input((64,)), layers.Dense(256, activation="relu"),
                              layers.Dense(256, activation="relu"), layers.Dense(10)])
    model.compile(keras.optimizers.Adam(learning_rate), loss, metrics=["accuracy"])
    return model

for restore in (False, True):
    model = overfitting_model()
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=restore)
    history = model.fit(x_small, y_small, epochs=100, batch_size=32,
                        validation_data=(x_test, y_test), callbacks=[stopper], verbose=0)
    val_loss = history.history["val_loss"]
    final = model.evaluate(x_test, y_test, verbose=0)[0]
    print(f"restore_best_weights={restore!s:5s} epochs run {len(val_loss)}  "
          f"best epoch {int(np.argmin(val_loss)) + 1}  best val loss {min(val_loss):.4f}  "
          f"loss of the returned model {final:.4f}")
print("default of restore_best_weights:", keras.callbacks.EarlyStopping().restore_best_weights)
```

```text
restore_best_weights=False epochs run 36  best epoch 31  best val loss 0.2176  loss of the returned model 0.2186
restore_best_weights=True  epochs run 36  best epoch 31  best val loss 0.2176  loss of the returned model 0.2176
default of restore_best_weights: False
```

The run stops at epoch 36. The best epoch is 31, five epochs earlier, which is the patience. With `restore_best_weights=False` the returned model is the one from epoch 36, and its loss is a little higher than the best. With `restore_best_weights=True` the returned model has the weights of epoch 31, and its loss equals the best value.

The last line of the block shows the default of `restore_best_weights`. It is `False`. If you do not set it, you get the last model and not the best one.

## Save the best model to disk

`ModelCheckpoint` writes the model to a file. With `save_best_only=True` it writes only when the monitored value improves. The file is a complete model: architecture, weights and optimizer state. [Chapter 14](14-saving-loading-export.md) opens the format.

```python
import os
import tempfile

import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
rows = np.random.default_rng(0).choice(len(x_train), 200, replace=False)
x_small, y_small = x_train[rows], y_train[rows]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(256, activation="relu"),
                          layers.Dense(256, activation="relu"), layers.Dense(10)])
model.compile("adam", loss, metrics=["accuracy"])

path = os.path.join(tempfile.mkdtemp(), "best.keras")
checkpoint = keras.callbacks.ModelCheckpoint(path, monitor="val_loss", save_best_only=True)
history = model.fit(x_small, y_small, epochs=60, batch_size=32,
                    validation_data=(x_test, y_test), callbacks=[checkpoint], verbose=0)

best = keras.saving.load_model(path)
val_loss = history.history["val_loss"]
print("file exists:", os.path.exists(path), " suffix:", os.path.splitext(path)[1])
print("best epoch:", int(np.argmin(val_loss)) + 1, " last epoch:", len(val_loss))
print("val loss, best epoch :", round(min(val_loss), 4))
print("val loss, last epoch :", round(val_loss[-1], 4))
print("val loss, file       :", round(best.evaluate(x_test, y_test, verbose=0)[0], 4))
```

```text
file exists: True  suffix: .keras
best epoch: 31  last epoch: 60
val loss, best epoch : 0.2176
val loss, last epoch : 0.2297
val loss, file       : 0.2176
```

The file holds the model of the best epoch. After 60 epochs the last model is worse than the file. This is the setup to use for long runs: the run can go on, and the best model is on disk when it ends.

## Lower the learning rate on a plateau

`ReduceLROnPlateau` multiplies the learning rate by `factor` when the monitored value has not improved for `patience` epochs. Keras writes the learning rate into the history under the key `learning_rate`, so no extra code is needed to see it.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile(keras.optimizers.Adam(0.01), loss, metrics=["accuracy"])
reducer = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=0.0001)
history = model.fit(x_train, y_train, epochs=15, batch_size=32,
                    validation_data=(x_test, y_test), callbacks=[reducer], verbose=0)

print("keys:", sorted(history.history))
print("val loss     :", [round(v, 3) for v in history.history["val_loss"]])
print("learning rate:", [round(v, 5) for v in history.history["learning_rate"]])
```

```text
keys: ['accuracy', 'learning_rate', 'loss', 'val_accuracy', 'val_loss']
val loss     : [0.339, 0.185, 0.145, 0.12, 0.104, 0.097, 0.09, 0.085, 0.08, 0.078, 0.081, 0.084, 0.076, 0.073, 0.073]
learning rate: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.0025, 0.0025, 0.0025]
```

The validation loss rises at epoch 11, from 0.078 to 0.081. With `patience=1`, the rate halves at once, from 0.01 to 0.005. The loss rises again at epoch 12, and the rate halves again to 0.0025. The `learning_rate` entry of an epoch is the rate that the epoch used. The reduction after epoch 11 shows in the entry for epoch 12.

## Stop on `nan`

[Chapter 1](01-the-first-model.md) trained a line with `SGD(1.5)` and got a loss of `nan` without an error. `TerminateOnNaN` turns that silent result into a stop.

```python
import keras
import numpy as np

keras.utils.set_random_seed(0)

x = np.linspace(-1, 1, 64, dtype="float32").reshape(-1, 1)
model = keras.Sequential([keras.Input((1,)), keras.layers.Dense(1)])
model.compile(keras.optimizers.SGD(1.5), "mse")

history = model.fit(x, 2 * x + 1, epochs=100, batch_size=16,
                    callbacks=[keras.callbacks.TerminateOnNaN()], verbose=0)
print("epochs run:", len(history.history["loss"]), "of 100")
print("last loss:", history.history["loss"][-1])
```

```text
Batch 3: Invalid loss, terminating training
epochs run: 16 of 100
last loss: inf
```

The run stops in epoch 16 and not in epoch 100. The callback prints the batch where the loss became invalid. This callback has no setting, and it costs nothing. Add it to every run that you do not watch.

## Write a callback

A callback is a class with methods named `on_epoch_end`, `on_train_batch_end` and others. Each method gets the logs of that moment. The model is in `self.model`. To stop the run, set `self.model.stop_training = True`.

The first callback stops the run when the validation accuracy reaches a target. The second one records the loss of every batch, which the history does not keep.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

class StopAtAccuracy(keras.callbacks.Callback):
    def __init__(self, target):
        super().__init__()
        self.target = target

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] >= self.target:
            self.model.stop_training = True

class BatchLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.values = []

    def on_train_batch_end(self, batch, logs=None):
        self.values.append(float(logs["loss"]))

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile("adam", loss, metrics=["accuracy"])
recorder = BatchLosses()
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test),
                    callbacks=[StopAtAccuracy(0.95), recorder], verbose=0)

epochs = len(history.history["loss"])
print("stopped after epoch:", epochs, " val accuracy:", round(history.history["val_accuracy"][-1], 4))
print("batches recorded:", len(recorder.values), "= epochs x 43:", epochs * 43)
print("loss of the first and last batch:", round(recorder.values[0], 3), round(recorder.values[-1], 3))

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile("adam", loss, metrics=["accuracy"])
try:
    model.fit(x_train, y_train, epochs=1, batch_size=32, callbacks=[StopAtAccuracy(0.95)], verbose=0)
except KeyError as error:
    print("without validation data: KeyError", error)
```

```text
stopped after epoch: 10  val accuracy: 0.9511
batches recorded: 430 = epochs x 43: 430
loss of the first and last batch: 2.297 0.258
without validation data: KeyError 'val_accuracy'
```

The run stops after epoch 10, when the validation accuracy first reaches 0.95. The recorder holds 430 values, one for each batch. The last value is not the loss of the last batch alone, which has only 3 images. [How it works](#how-it-works) shows what it is.

Two warnings apply to the custom stop. The target is read on the validation data, so it is a decision made with the validation set ([Chapter 8](08-overfitting-and-regularization.md) returns to this). Also, `logs` can lack a key. The last lines of the block pass no validation data, and `logs["val_accuracy"]` raises a `KeyError`.

## How it works

`fit` calls the methods of every callback at fixed points. The callbacks form a list, and each point calls them in list order. A callback that you write overrides only the methods it needs.

The next block records the points of a run with 2 batches in one epoch and a validation set. It prints them in order.

```python
import keras
import numpy as np
from keras import layers

class Trace(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.events = []

for name in ("on_train_begin", "on_epoch_begin", "on_train_batch_begin", "on_train_batch_end",
             "on_test_begin", "on_test_batch_begin", "on_test_batch_end", "on_test_end",
             "on_epoch_end", "on_train_end"):
    setattr(Trace, name, lambda self, *args, _name=name, **kwargs: self.events.append(_name))

x = np.zeros((64, 4), dtype="float32")
y = np.zeros((64, 1), dtype="float32")
keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((4,)), layers.Dense(1)])
model.compile("sgd", "mse")
trace = Trace()
model.fit(x, y, epochs=1, batch_size=32, validation_data=(x[:32], y[:32]), callbacks=[trace], verbose=0)
for event in trace.events:
    print(event)
```

```text
on_train_begin
on_epoch_begin
on_train_batch_begin
on_train_batch_end
on_train_batch_begin
on_train_batch_end
on_test_begin
on_test_batch_begin
on_test_batch_end
on_test_end
on_epoch_end
on_train_end
```

The run has one epoch of two training batches, then a validation pass of one batch. The order is the contract. `on_epoch_end` comes after the validation pass, so its `logs` contain the `val_` keys. This is the reason that `StopAtAccuracy` can read `logs["val_accuracy"]` in `on_epoch_end` and not in `on_train_batch_end`.

The `logs` argument of a batch method does not hold the loss of that batch. The next block records it for two epochs and compares the last batch of each epoch with the history.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()

class BatchLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.values = []

    def on_train_batch_end(self, batch, logs=None):
        self.values.append(float(logs["loss"]))

keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
recorder = BatchLosses()
history = model.fit(x_train, y_train, epochs=2, batch_size=32, shuffle=False,
                    callbacks=[recorder], verbose=0).history

values = np.array(recorder.values)
print("batches recorded:", len(values))
print("last batch of epoch 1:", round(values[42], 4), " history loss of epoch 1:", round(history["loss"][0], 4))
print("last batch of epoch 2:", round(values[85], 4), " history loss of epoch 2:", round(history["loss"][1], 4))
print("first four values:", [round(float(v), 3) for v in values[:4]])
```

```text
batches recorded: 86
last batch of epoch 1: 2.0879  history loss of epoch 1: 2.0879
last batch of epoch 2: 1.5542  history loss of epoch 2: 1.5542
first four values: [2.37, 2.318, 2.313, 2.32]
```

The value at the last batch of an epoch equals the loss of that epoch in the history. The callback gets a running average over the batches of the epoch so far, and not the loss of one batch. The invariant is this: the logs of a batch are the epoch average up to that batch. To get the loss of a single batch, compute it yourself in a `train_step` ([Chapter 9](09-custom-training.md)).

## Reference

```python
import keras
from examples.signatures import show

show(keras.callbacks.EarlyStopping, keras.callbacks.ModelCheckpoint,
     keras.callbacks.ReduceLROnPlateau, keras.callbacks.TerminateOnNaN, keras.callbacks.Callback)
```

```text
EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
    mode='auto', baseline=None, restore_best_weights=False,
    start_from_epoch=0)
ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
    save_weights_only=False, mode='auto', save_freq='epoch',
    initial_value_threshold=None)
ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0,
    mode='auto', min_delta=0.0001, cooldown=0, min_lr=0.0, **kwargs)
TerminateOnNaN(raise_error: bool = False)
Callback()
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `EarlyStopping(monitor)` | `'val_loss'` | the log key to watch | — |
| `EarlyStopping(patience)` | `0` | epochs without improvement before the stop | — |
| `EarlyStopping(restore_best_weights)` | `False` | return the weights of the best epoch | `False` returns the last model, not the best |
| `EarlyStopping(mode)` | `'auto'` | direction of "better" | `'min'` on accuracy stopped at epoch 6 with 0.6444 |
| `ModelCheckpoint(filepath)` | required | where to write the model | the chapter wrote a `.keras` file and loaded it back |
| `ModelCheckpoint(save_best_only)` | `False` | write only on improvement | — |
| `ReduceLROnPlateau(factor)` | `0.1` | multiplier of the rate | — |
| `ReduceLROnPlateau(patience)` | `10` | epochs without improvement before the cut | — |
| `TerminateOnNaN(raise_error)` | `False` | raise an error and not only stop | — |

## What you have now

You have the four callbacks that most runs need: `EarlyStopping` with `restore_best_weights=True`, `ModelCheckpoint` with `save_best_only=True`, `ReduceLROnPlateau` and `TerminateOnNaN`. You also have the two methods that cover most custom callbacks: `on_epoch_end` and `on_train_batch_end`.

[Chapter 8](08-overfitting-and-regularization.md) takes the overfitting setup of this chapter and measures what regularization does to it. Exercises for this chapter are in [exercises/07.md](../exercises/07.md), and the worked solutions are in [exercises/solutions/07.md](../exercises/solutions/07.md).
