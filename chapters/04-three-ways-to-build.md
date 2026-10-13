# Chapter 4 — Three ways to build a model

> **Build** — the same small network written as a `Sequential` model, as a functional model and as a subclassed model, with a measurement that shows the three are one function. Then a model with two inputs, a shared layer and two outputs.
>
> **Theory kept** — Keras has one model class with three construction styles. The styles differ in what Keras knows about the graph before data arrives.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0. The code is New.
>
> **Run this first** — `uv run python -m examples.ch04_three_ways_to_build`, which is [`examples/ch04_three_ways_to_build.py`](../examples/ch04_three_ways_to_build.py).

You can build a Keras model in three ways. Each way has a use. This chapter builds one network three times and shows that the three networks are the same function. Then it uses the one style that can express a graph with branches.

## The same network, three times

The network has an input of 8 features, a hidden layer of 16 units with a ReLU, and an output of 3 values.

```python
import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)

sequential = keras.Sequential([
    keras.Input((8,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(3),
])

inputs = keras.Input((8,))
hidden = layers.Dense(16, activation="relu")(inputs)
functional = keras.Model(inputs, layers.Dense(3)(hidden))

class Subclassed(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = layers.Dense(16, activation="relu")
        self.out = layers.Dense(3)

    def call(self, x):
        return self.out(self.hidden(x))

subclassed = Subclassed()
x = np.random.default_rng(0).normal(size=(5, 8)).astype("float32")
subclassed(x)

for model in (sequential, functional, subclassed):
    print(f"{type(model).__name__:11s} params {model.count_params()}  weight arrays {len(model.weights)}")

functional.set_weights(sequential.get_weights())
subclassed.set_weights(sequential.get_weights())
outputs = [ops.convert_to_numpy(m(x)) for m in (sequential, functional, subclassed)]
print("functional equals sequential:", np.array_equal(outputs[0], outputs[1]))
print("subclassed equals sequential:", np.array_equal(outputs[0], outputs[2]))
```

```text
Sequential  params 195  weight arrays 4
Functional  params 195  weight arrays 4
Subclassed  params 195  weight arrays 4
functional equals sequential: True
subclassed equals sequential: True
```

All three have 195 parameters: 8 × 16 + 16 for the first layer and 16 × 3 + 3 for the second. After the weights are copied, the three models give identical outputs. The styles are three ways to describe one function.

## What Keras knows before data arrives

The styles differ in when the weights exist. The next block builds each model and asks for the summary before the model sees data.

```python
import keras
from keras import layers

keras.utils.set_random_seed(0)

with_input = keras.Sequential([keras.Input((8,)), layers.Dense(4), layers.Dense(2)])
print("Sequential with Input: weights =", len(with_input.weights))

without_input = keras.Sequential([layers.Dense(4), layers.Dense(2)])
print("Sequential without Input: built =", without_input.built, "weights =", len(without_input.weights))

class Subclassed(keras.Model):
    def __init__(self):
        super().__init__()
        self.hidden = layers.Dense(4)
        self.out = layers.Dense(2)

    def call(self, x):
        return self.out(self.hidden(x))

subclassed = Subclassed()
print("Subclassed: built =", subclassed.built, "weights =", len(subclassed.weights))
subclassed.summary(line_length=60)
```

```text
Sequential with Input: weights = 4
Sequential without Input: built = False weights = 0
Subclassed: built = False weights = 0
Model: "subclassed"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Layer (type)             ┃ Output Shape      ┃   Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ dense_4 (Dense)          │ ?                 │         0 │
│                          │                   │ (unbuilt) │
├──────────────────────────┼───────────────────┼───────────┤
│ dense_5 (Dense)          │ ?                 │         0 │
│                          │                   │ (unbuilt) │
└──────────────────────────┴───────────────────┴───────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)
```

A `Sequential` model with an `Input` creates its weights at once. Without the `Input` it waits for data. A subclassed model always waits, and its summary shows `?` for every shape. This is not a defect. A subclassed `call` is ordinary Python code, and Keras cannot know the shapes until it runs the code.

The consequence is practical. For a functional or `Sequential` model with an `Input`, `summary()` is a complete check of the shapes before training. For a subclassed model, call it once on a small batch first.

## Branches, shared layers and two outputs

A `Sequential` model is a straight line. Many real models are not. The functional style handles any directed graph of layers. The next model compares two inputs of 4 features with one shared `Dense` layer, joins the results and returns two outputs: a score and a class.

```python
import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(0)

left = keras.Input((4,), name="left")
right = keras.Input((4,), name="right")
shared = layers.Dense(8, activation="relu", name="shared")
joined = layers.Concatenate()([shared(left), shared(right)])
score = layers.Dense(1, name="score")(joined)
label = layers.Dense(3, activation="softmax", name="label")(joined)
model = keras.Model([left, right], [score, label])

print("layers:", [layer.name for layer in model.layers])
print("outputs:", model.output_names)
print("parameters:", model.count_params())

print("a second Dense(8) on 4 inputs would add:", 4 * 8 + 8, "parameters")
```

```text
layers: ['left', 'right', 'shared', 'concatenate', 'score', 'label']
outputs: ['score', 'label']
parameters: 108
a second Dense(8) on 4 inputs would add: 40 parameters
```

The shared layer appears once in the layer list and its 40 parameters count once: 108 in total is 40 + 17 + 51. Both inputs go through the same weights. This is the idea of a Siamese network: one function, applied twice.

The model has two outputs. Keras needs a loss for each one. The `compile` call takes a dictionary keyed by the output names.

```python
import keras
import numpy as np
from keras import layers

keras.utils.set_random_seed(0)
left = keras.Input((4,), name="left")
right = keras.Input((4,), name="right")
shared = layers.Dense(8, activation="relu", name="shared")
joined = layers.Concatenate()([shared(left), shared(right)])
score = layers.Dense(1, name="score")(joined)
label = layers.Dense(3, activation="softmax", name="label")(joined)
model = keras.Model([left, right], [score, label])

model.compile(
    optimizer="adam",
    loss={"score": "mse", "label": "sparse_categorical_crossentropy"},
    loss_weights={"score": 1.0, "label": 0.5},
)

rng = np.random.default_rng(0)
data = {"left": rng.normal(size=(32, 4)).astype("float32"),
        "right": rng.normal(size=(32, 4)).astype("float32")}
targets = {"score": rng.normal(size=(32, 1)).astype("float32"),
           "label": rng.integers(0, 3, size=(32,))}
history = model.fit(data, targets, epochs=2, batch_size=16, verbose=0)
last = {name: values[-1] for name, values in history.history.items()}
print(sorted(last))
print("loss == score_loss + 0.5 * label_loss:",
      bool(np.isclose(last["loss"], last["score_loss"] + 0.5 * last["label_loss"], rtol=1e-4)))
```

```text
['label_loss', 'loss', 'score_loss']
loss == score_loss + 0.5 * label_loss: True
```

The history has a total `loss` and one loss for each output. The total is `1.0 × score_loss + 0.5 × label_loss`. Pass inputs and targets as dictionaries keyed by name. Then the order of the arguments cannot cause a silent mistake.

## Which style to use

Use `Sequential` when the model is a stack. Use the functional style when the graph has branches, shared layers or several inputs. Use subclassing when `call` must contain Python control flow that the graph cannot express, such as a loop of data-dependent length.

Prefer the functional style when you have a choice. It checks the shapes when you build the model. It makes the graph visible in `summary()`. It saves and loads with only built-in parts.

A subclassed model gives up the first two, and in return it gives you full control of `call`. You can save a subclassed model too. [Chapter 14](14-saving-loading-export.md) measures what it needs: a registered class, `**kwargs` in `__init__` and a `get_config` for every argument.

You can also mix the styles. A subclassed model can contain a functional model as a layer, and a functional model can contain a subclassed layer. [Chapter 12](12-attention-and-transformers.md) builds a transformer block as a layer and then stacks the blocks in a functional model.

## How it works

A functional model is a recorded graph. When you call a layer on a `keras.Input`, you do not give it data. You give it a symbolic tensor with a shape and a type. The layer returns another symbolic tensor, and Keras records a node: this layer, applied to these inputs. `keras.Model(inputs, outputs)` collects the nodes between the inputs and the outputs.

This record is the reason a functional model has a complete `summary()` before any data arrive. It is also the reason you can save its configuration and rebuild the model from it. The next block shows the record.

```python
import keras
import numpy as np
from keras import layers, ops

keras.utils.set_random_seed(0)

inputs = keras.Input((8,), name="features")
hidden = layers.Dense(16, activation="relu", name="hidden")(inputs)
outputs = layers.Dense(3, name="out")(hidden)
model = keras.Model(inputs, outputs)

print("symbolic tensor:", type(inputs).__name__, inputs.shape)
config = model.get_config()
print("recorded layers:", [(layer["class_name"], layer["name"]) for layer in config["layers"]])

clone = keras.Model.from_config(config)
clone.set_weights(model.get_weights())
x = np.random.default_rng(0).normal(size=(4, 8)).astype("float32")
same = np.array_equal(ops.convert_to_numpy(model(x)), ops.convert_to_numpy(clone(x)))
print("clone from the config gives the same output:", same)
```

```text
symbolic tensor: KerasTensor (None, 8)
recorded layers: [('InputLayer', 'features'), ('Dense', 'hidden'), ('Dense', 'out')]
clone from the config gives the same output: True
```

The configuration lists the three layers in order, and a model built from it is the same function once it has the weights. The invariant is: *a functional model is its graph plus its weights.* A subclassed model has no graph. It has only Python code in `call`, so Keras cannot show its shapes before data arrive. This is the mechanism behind the `?` in the subclassed summary.

## Reference

```python
import keras
from keras import layers
from examples.signatures import show

show(("Sequential", keras.Sequential.__init__), ("Model", keras.Model.__init__),
     ("Model.summary", keras.Model.summary), ("Input", keras.Input), layers.Concatenate)
```

```text
Sequential(layers=None, trainable=True, name=None)
Model(*args, **kwargs)
Model.summary(line_length=None, positions=None, print_fn=None,
    expand_nested=False, show_trainable=False, layer_range=None)
Input(shape=None, batch_size=None, dtype=None, sparse=None, ragged=None,
    batch_shape=None, name=None, tensor=None, optional=False)
Concatenate(axis=-1, **kwargs)
```

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `Sequential(layers)` | `None` | list of layers in order | without an `Input` the model is built at the first call (Chapter 4) |
| `Model(inputs, outputs)` | — | the signature shows `*args, **kwargs`: pass `inputs` and `outputs` for a graph, nothing for a subclass | a subclass must call `super().__init__()` |
| `Model(name)` | — | name of the model, passed as a keyword | — |
| `Input(name)` | `None` | key for a dictionary of inputs | name inputs when a model has more than one |
| `Input(shape)` | `None` | shape without the batch axis | — |
| `Concatenate(axis)` | `-1` | axis along which the inputs join | inputs must agree on all other axes |
| `summary(line_length)` | `None` | width of the printed table | an unbuilt subclassed model prints `?` and 0 parameters |
| `summary(expand_nested)` | `False` | show the layers of nested models | — |

## What you have now

You have three equal models and a measurement of when each style creates its weights. You also have a model with a shared layer and two outputs that trains with a dictionary of losses.

The next part of the book is about training. [Chapter 5](05-data-pipelines.md) starts with the data. Exercises for this chapter are in [exercises/04.md](../exercises/04.md), and the worked solutions are in [exercises/solutions/04.md](../exercises/solutions/04.md).
