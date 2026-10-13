# Chapter 14 — Saving, loading and export

> **Build** — a round trip that gives back the same predictions bit for bit and the same training, a list of four ways that a load fails or silently changes a model, and an exported artifact that a service can load without Keras.
>
> **Theory kept** — a saved model is a configuration plus a list of arrays. The configuration says how to rebuild the objects. The arrays fill them.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0. The code is New.
>
> **Run this first** — `uv run python -m examples.ch14_saving_loading_export`, which is [`examples/ch14_saving_loading_export.py`](../examples/ch14_saving_loading_export.py).

A trained model that you cannot load again is lost work. Saving looks like one call, and it is one call when the model uses only built-in parts. It becomes a source of silent faults when the model has custom parts. This chapter measures each case.

## What a `.keras` file holds

`model.save("name.keras")` writes a zip archive. The next block opens it and lists what is inside. It also prints how the weights are keyed.

```python
import io
import json
import os
import tempfile
import zipfile

import h5py
import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(0)
model = keras.Sequential([
    keras.Input((8,)),
    layers.Dense(4, activation="relu", name="hidden"),
    layers.Dense(2, name="out"),
])
model.compile("adam", "mse")
model.fit(np.zeros((8, 8), dtype="float32"), np.zeros((8, 2), dtype="float32"), verbose=0)

path = os.path.join(tempfile.mkdtemp(), "model.keras")
model.save(path)
archive = zipfile.ZipFile(path)
print("files:", archive.namelist())

config = json.loads(archive.read("config.json"))
print("config keys:", sorted(config))
print("layer names in the config:", [layer["config"]["name"] for layer in config["config"]["layers"]])

weights = h5py.File(io.BytesIO(archive.read("model.weights.h5")))
print("weight groups:", sorted(weights["layers"]))
print("optimizer arrays:", len(weights["optimizer"]["vars"]))
```

```text
files: ['metadata.json', 'config.json', 'model.weights.h5']
config keys: ['build_config', 'class_name', 'compile_config', 'config', 'module', 'registered_name']
layer names in the config: ['input_layer', 'hidden', 'out']
weight groups: ['dense', 'dense_1']
optimizer arrays: 10
```

The archive has three files: a small `metadata.json`, a `config.json` that rebuilds the objects, and a `model.weights.h5` with the arrays. The optimizer state is in the weights file too. This is why a loaded model can continue training from where it stopped.

One detail is easy to miss. In the config, the layers are named `hidden` and `out`. In the weights file, the groups are named `dense` and `dense_1`. The file keys the arrays by the class of the layer and its order, and not by the name that you gave.

## How it works

Loading is two steps. First Keras reads `config.json` and builds each object from its `class_name` and `config`. A class that Keras does not know must be found in a registry or in the `custom_objects` that you pass. Then Keras builds the weights and assigns the arrays, in order, to the objects that it made.

The invariant is this: the config must rebuild an object that has the same weights as the saved one, in the same order. Every fault in this chapter breaks one half of this sentence. A class that cannot be found breaks the first step. An argument that is not in the config gives an object with a different shape. Then the second step fails, or it fills the object with the wrong arrays.

## A round trip is exact

The next block saves a trained model, loads it, and compares. Then it trains both the original and the loaded model for one more epoch on the same batches.

```python
import os
import tempfile

import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,)), layers.Dense(32, activation="relu"), layers.Dense(10)])
model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)

path = os.path.join(tempfile.mkdtemp(), "model.keras")
model.save(path)
loaded = keras.saving.load_model(path)

print("predictions equal:", np.array_equal(model.predict(x_test, verbose=0), loaded.predict(x_test, verbose=0)))
print("loaded model is compiled:", loaded.compiled)
print("optimizer iterations:", int(model.optimizer.iterations.numpy()), int(loaded.optimizer.iterations.numpy()))

model.fit(x_train, y_train, epochs=1, batch_size=32, shuffle=False, verbose=0)
loaded.fit(x_train, y_train, epochs=1, batch_size=32, shuffle=False, verbose=0)
same = all(np.array_equal(a, b) for a, b in zip(model.get_weights(), loaded.get_weights()))
print("weights equal after one more epoch:", same)
```

```text
predictions equal: True
loaded model is compiled: True
optimizer iterations: 86 86
weights equal after one more epoch: True
```

The predictions are bit for bit equal. The loaded model is compiled, and its optimizer has the same step count. After one more epoch of the same batches, the weights are still equal. The file holds everything that training needs.

## A custom layer must be findable

A custom class is not known to Keras when it loads a file. The next block saves a model with a custom layer and loads it three ways.

```python
import os
import tempfile

import keras
from keras import layers

class Scale(keras.layers.Layer):
    def __init__(self, factor=2.0, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {**super().get_config(), "factor": self.factor}

@keras.saving.register_keras_serializable(package="book")
class RegisteredScale(Scale):
    pass

folder = tempfile.mkdtemp()
plain = keras.Sequential([keras.Input((4,)), Scale(3.0), layers.Dense(1)])
plain.save(os.path.join(folder, "plain.keras"))
registered = keras.Sequential([keras.Input((4,)), RegisteredScale(3.0), layers.Dense(1)])
registered.save(os.path.join(folder, "registered.keras"))

try:
    keras.saving.load_model(os.path.join(folder, "plain.keras"))
except TypeError as error:
    print("not registered:", str(error).split(" could not")[0])

with_objects = keras.saving.load_model(os.path.join(folder, "plain.keras"), custom_objects={"Scale": Scale})
print("with custom_objects, factor:", with_objects.layers[0].factor)

by_registry = keras.saving.load_model(os.path.join(folder, "registered.keras"))
print("registered class, factor:", by_registry.layers[0].factor)
```

```text
not registered: <class 'keras.src.models.sequential.Sequential'>
with custom_objects, factor: 3.0
registered class, factor: 3.0
```

Without registration the load fails with a `TypeError` that names the model class and not the missing layer. The message is not helpful, so know this cause. There are two fixes: pass the class in `custom_objects`, or decorate it with `register_keras_serializable`. The decorator is the better one, because the file then loads anywhere that the class is imported.

## A subclassed model needs `get_config` for its arguments

A model with a constructor argument has a second trap. The next block saves four subclassed models and tries to load each one. Every class is registered.

```python
import os
import tempfile

import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)
x = np.zeros((2, 3), dtype="float32")
folder = tempfile.mkdtemp()

@keras.saving.register_keras_serializable(package="book")
class WithKwargs(keras.Model):
    def __init__(self, width=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden = layers.Dense(width)
        self.out = layers.Dense(2)

    def call(self, x):
        return self.out(self.hidden(x))

@keras.saving.register_keras_serializable(package="book")
class WithConfig(WithKwargs):
    def __init__(self, width=4, **kwargs):
        super().__init__(width=width, **kwargs)
        self.width = width

    def get_config(self):
        return {**super().get_config(), "width": self.width}

@keras.saving.register_keras_serializable(package="book")
class NoKwargs(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = layers.Dense(4)
        self.out = layers.Dense(2)

    def call(self, x):
        return self.out(self.hidden(x))

def round_trip(label, model):
    model(x)
    path = os.path.join(folder, label + ".keras")
    model.save(path)
    try:
        loaded = keras.saving.load_model(path)
        equal = np.array_equal(ops.convert_to_numpy(model(x)), ops.convert_to_numpy(loaded(x)))
        print(f"{label:28s} loads, outputs equal: {equal}")
    except Exception as error:
        print(f"{label:28s} fails: {type(error).__name__}")

round_trip("default argument", WithKwargs())
round_trip("width=7, no get_config", WithKwargs(width=7))
round_trip("width=7, with get_config", WithConfig(width=7))
round_trip("__init__ without **kwargs", NoKwargs())
```

```text
default argument             loads, outputs equal: True
width=7, no get_config       fails: ValueError
width=7, with get_config     loads, outputs equal: True
__init__ without **kwargs    fails: TypeError
```

The first and the third cases load, and their outputs are equal. The second case fails: the file holds no `width`, so Keras rebuilds the model with the default of 4, and the weights of the file do not fit. The error message talks about missing variables, and it does not mention the argument. The fourth case fails because Keras passes `name`, `trainable` and `dtype` to `__init__`, and the class accepts none of them.

The rule has two lines. A subclassed model must pass `**kwargs` to its parent. It must return every non-default argument from `get_config`.

## A loss without `get_config` forgets its arguments

A compiled model saves its loss too. A loss class that has no `get_config` loads with its default arguments. Nothing fails.

```python
import os
import tempfile

import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)

@keras.saving.register_keras_serializable(package="book")
class ScaledError(keras.losses.Loss):
    def __init__(self, factor=0.1, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, y_true, y_pred):
        return ops.mean(ops.square(y_true - y_pred), axis=-1) * (1.0 + self.factor)

@keras.saving.register_keras_serializable(package="book")
class ScaledErrorWithConfig(ScaledError):
    def get_config(self):
        return {**super().get_config(), "factor": self.factor}

folder = tempfile.mkdtemp()
for loss_class in (ScaledError, ScaledErrorWithConfig):
    model = keras.Sequential([keras.Input((3,)), layers.Dense(1)])
    model.compile("sgd", loss_class(0.5))
    model.fit(np.zeros((4, 3), dtype="float32"), np.zeros((4, 1), dtype="float32"), verbose=0)
    path = os.path.join(folder, loss_class.__name__ + ".keras")
    model.save(path)
    print(f"{loss_class.__name__:22s} factor saved 0.5, factor after load:",
          keras.saving.load_model(path).loss.factor)
```

```text
ScaledError            factor saved 0.5, factor after load: 0.1
ScaledErrorWithConfig  factor saved 0.5, factor after load: 0.5
```

The class without `get_config` comes back with a factor of 0.1. The class with `get_config` comes back with 0.5. A model that you load to continue training now trains with a different loss, and it reports no error. Give every custom class a `get_config`, and test the round trip.

## Export for serving

A `.keras` file needs Keras and your classes. A service often has neither. `model.export` writes a TensorFlow SavedModel. It holds the computation and the weights and no Keras code.

```python
import os
import tempfile

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load()
keras.utils.set_random_seed(0)
model = keras.Sequential([keras.Input((64,), name="pixels"), layers.Dense(32, activation="relu"),
                          layers.Dense(10)])
model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)

folder = os.path.join(tempfile.mkdtemp(), "export")
model.export(folder, verbose=False)
print("files:", sorted(os.listdir(folder)))

served = tf.saved_model.load(folder)
print("endpoints:", sorted(served.signatures))
endpoint = served.signatures["serving_default"]
result = endpoint(pixels=tf.constant(x_test[:5]))
print("output names:", list(result))
expected = model.predict(x_test[:5], verbose=0)
print("equal to model.predict:", bool(np.allclose(result["output_0"].numpy(), expected, atol=1e-5)))
```

```text
files: ['assets', 'fingerprint.pb', 'saved_model.pb', 'variables']
endpoints: ['serve', 'serving_default']
output names: ['output_0']
equal to model.predict: True
```

The exported model has a `serving_default` endpoint, and it returns the same numbers as `predict`. The key of the input in the signature is the name of the input of the model, `pixels`. Name your inputs when you build the model. An unnamed input gets a name such as `keras_tensor_37`, and the number depends on how many tensors the process made before it. In the first version of this block, the call with `keras_tensor` worked when the block ran alone. It failed when it ran after the other blocks of the chapter.

## Reference

```python
import keras
from examples.signatures import show

show(("Model.save", keras.Model.save), ("saving.load_model", keras.saving.load_model),
     ("Model.save_weights", keras.Model.save_weights), ("Model.load_weights", keras.Model.load_weights),
     ("Model.export", keras.Model.export),
     ("saving.register_keras_serializable", keras.saving.register_keras_serializable))
```

```text
Model.save(filepath, overwrite=True, zipped=None, **kwargs)
saving.load_model(filepath, custom_objects=None, compile=True, safe_mode=True)
Model.save_weights(filepath, overwrite=True, max_shard_size=None)
Model.load_weights(filepath, skip_mismatch=False, **kwargs)
Model.export(filepath, format='tf_saved_model', verbose=None,
    input_signature=None, **kwargs)
saving.register_keras_serializable(package='Custom', name=None)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `save(filepath)` | required | path that ends in `.keras` | — |
| `save(overwrite)` | `True` | replace an existing file | — |
| `load_model(custom_objects)` | `None` | map from name to class | needed for an unregistered class |
| `load_model(compile)` | `True` | rebuild the optimizer and the loss from the file | the loss class must be registered, or pass `custom_objects` |
| `load_model(safe_mode)` | `True` | refuse to run unsafe code from a file | — |
| `save_weights(filepath)` | required | write the arrays only; the name ends in `.weights.h5` | the architecture must be rebuilt by you |
| `export(filepath)` | required | write a SavedModel for serving | an unnamed input became `keras_tensor_37` in one run, so a service cannot use it as a key |
| `export(format)` | `'tf_saved_model'` | kind of artifact | — |
| `register_keras_serializable(package)` | `'Custom'` | put the class in the registry under `package>Name` | an unregistered class failed to load (this chapter) |

## What you have now

You have a round trip that you measured, and four faults that you can now find: a class that is not registered, a subclassed model that does not keep its arguments, a loss that forgets its arguments, and a missing `**kwargs`. You also have an exported artifact that a service can load.

[Chapter 15](15-performance-and-reproducibility.md) is about speed and repeatability. Exercises for this chapter are in [exercises/14.md](../exercises/14.md), and the worked solutions are in [exercises/solutions/14.md](../exercises/solutions/14.md).
