# Chapter 10 — Convolutional networks

> **Build** — a convolutional network with 5,130 weights that classifies the digits with 99.1% accuracy, and a table that shows how five models react when the test images move by one pixel.
>
> **Theory kept** — a convolution applies the same small filter at every position. Its output moves when the input moves. A pooling step over all positions removes most of the dependence on position.
>
> **Source** — Keras 3.15.1 on TensorFlow 2.21.0 and the digits data of scikit-learn 1.9.1. The code is New.
>
> **Run this first** — `uv run python -m examples.ch10_convolutional_networks`, which is [`examples/ch10_convolutional_networks.py`](../examples/ch10_convolutional_networks.py).

A dense layer connects every input to every output. For an image, this is a poor fit: it has one weight for each pair of pixel and unit, and it does not know that two pixels are neighbors. A convolutional layer uses a small filter, 3 × 3 for example, and slides it over the image. The same weights are used at every position.

This chapter uses the 8 × 8 digits of [Chapter 5](05-data-pipelines.md), now as images with one channel. The images are small, but the ideas are the same as for large images. Keras uses the layout `(batch, height, width, channels)`.

## Shapes and parameters

Three arguments decide the output shape of a `Conv2D` layer: the kernel size, the padding and the stride. With `padding="valid"` the filter stays inside the image, so the output is smaller. With `padding="same"` the image is padded with zeros, so the output has the same size at stride 1. A stride of 2 halves the size.

The parameters do not depend on the size of the image. A layer with `filters` filters of size `k × k` on `c` input channels has `k × k × c × filters + filters` weights.

```python
import keras
from keras import layers

keras.utils.set_random_seed(0)

print(f"{'layer':44s} {'output shape':18s} parameters")
for label, layer in [
    ("Conv2D(16, 3, padding='valid')", layers.Conv2D(16, 3, padding="valid")),
    ("Conv2D(16, 3, padding='same')", layers.Conv2D(16, 3, padding="same")),
    ("Conv2D(16, 3, padding='same', strides=2)", layers.Conv2D(16, 3, padding="same", strides=2)),
    ("Conv2D(16, 5, padding='valid')", layers.Conv2D(16, 5, padding="valid")),
]:
    layer.build((None, 8, 8, 1))
    shape = tuple(layer.compute_output_shape((None, 8, 8, 1)))
    print(f"{label:44s} {str(shape):18s} {layer.count_params()}")
print("formula for the first row: 3 * 3 * 1 * 16 + 16 =", 3 * 3 * 1 * 16 + 16)
```

```text
layer                                        output shape       parameters
Conv2D(16, 3, padding='valid')               (None, 6, 6, 16)   160
Conv2D(16, 3, padding='same')                (None, 8, 8, 16)   160
Conv2D(16, 3, padding='same', strides=2)     (None, 4, 4, 16)   160
Conv2D(16, 5, padding='valid')               (None, 4, 4, 16)   416
formula for the first row: 3 * 3 * 1 * 16 + 16 = 160
```

The `valid` filter of size 3 turns 8 pixels into 6, and the `same` filter keeps 8. The weight count does not change with the padding. A filter of size 5 has 25 weights for each filter instead of 9. The same layer has the same 160 weights on a 1,000 × 1,000 image. A dense layer from 64 inputs to 16 units has 1,040 weights. On a 1,000 × 1,000 image the same dense layer has 16 million.

#### `keras.layers.Conv2D`

```text
keras.layers.Conv2D(
    filters,
    kernel_size,
    strides=(1, 1),
    padding='valid',
    data_format=None,
    dilation_rate=(1, 1),
    groups=1,
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    **kwargs,
)
```

The layer slides `filters` kernels of the size `kernel_size` over the image. The parameters from `activation` to `bias_constraint` have the same meaning as in `Dense` ([Chapter 1](01-the-first-model.md)).

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `filters` | required | the number of kernels, which is the number of output channels | — |
| `kernel_size` | required | the height and width of a kernel, as one number or a pair | a kernel of 9 on an 8 × 8 image with `padding="valid"` raises `ValueError: Computed output size would be zero or negative` |
| `strides` | `(1, 1)` | the step of the window | with 2 and `"valid"`, the 8 × 8 input gave 3 × 3 |
| `padding` | `'valid'` | `"valid"` or `"same"` | `"causal"` raises `ValueError`. It exists for `Conv1D` only ([Chapter 11](11-sequences.md)) |
| `data_format` | `None` | `"channels_last"` or `"channels_first"`. With `None`, the setting of `keras.config`, `channels_last` | with `"channels_first"`, the input `(8, 8, 1)` was read as 8 channels of 8 × 1, and the layer raised `ValueError` |
| `dilation_rate` | `(1, 1)` | the gap between the taps of the kernel | with 2, a kernel of 3 covers 5 × 5: the output was 4 × 4 with the same 160 weights. With `strides=2` too, it raises `ValueError: strides > 1 not supported in conjunction with dilation_rate > 1` |
| `groups` | `1` | split the input channels into groups with their own kernels | with 2 on 4 input channels, 304 weights and not 592. On 1 channel it raises `ValueError` |
| `activation` | `None` | the function on the output | — |
| `use_bias` | `True` | add one bias for each filter | — |
| `kernel_initializer` | `'glorot_uniform'` | the start values of the kernel | — |
| `bias_initializer` | `'zeros'` | the start values of the bias | — |
| `kernel_regularizer` | `None` | a penalty on the kernel | — |
| `bias_regularizer` | `None` | a penalty on the bias | — |
| `activity_regularizer` | `None` | a penalty on the output | — |
| `kernel_constraint` | `None` | a rule for the kernel after each update | — |
| `bias_constraint` | `None` | a rule for the bias | — |
| `**kwargs` | — | the arguments of `Layer`, such as `name` | — |

## Convolution moves with the input

The property that matters is *equivariance*: if you move the input, the output moves by the same amount. The next block checks it. It builds one convolutional layer with random weights, shifts the images by one pixel to the right, and compares the two outputs.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

keras.utils.set_random_seed(0)

_, x_test, _, _ = load(images=True)
empty_edge = (x_test[:, :, -1, :].max(axis=(1, 2)) == 0)
images = x_test[empty_edge]
shifted = np.zeros_like(images)
shifted[:, :, 1:, :] = images[:, :, :-1, :]

layer = layers.Conv2D(4, 3, padding="valid")
before = ops.convert_to_numpy(layer(images))
after = ops.convert_to_numpy(layer(shifted))

print("test images with an empty right edge:", int(empty_edge.sum()), "of", len(x_test))
print("output moved by one column:", bool(np.allclose(after[:, :, 1:, :], before[:, :, :-1, :], atol=1e-6)))
same_max = np.isclose(after.max(axis=(1, 2)), before.max(axis=(1, 2)), atol=1e-6)
print("share of (image, channel) pairs with the same maximum:", round(float(same_max.mean()), 3))
```

```text
test images with an empty right edge: 387 of 450
output moved by one column: True
share of (image, channel) pairs with the same maximum: 0.979
```

The output of the layer on the shifted image is the output on the original image, one column to the right. This holds for every column that both outputs contain. The shifted image has one new window at the left edge, and it loses one window at the right edge. So the output moves, but it is not a pure shift at the edges.

The last line measures the effect. The maximum of a channel over all positions is the same for most of the pairs, and not for all of them. A maximum that sat in the window that left, or in the new window, changes. A global pool makes the answer nearly independent of the position. It does not make it exactly independent.

## Five models and one pixel

The next block trains five models for 40 epochs on the centered training images. It then scores each model on three sets of test images: the original images, the images shifted one pixel to the right, and the images shifted one pixel to the left. The shifts fill the free column with zero.

- **dense**: flatten, a layer of 64 units, and the output.
- **cnn flatten**: a convolution, a max pool of size 2, flatten, and a dense layer.
- **cnn global**: two convolutions and a global max pool.
- **dense aug**: the dense model with random translations of the training images.
- **cnn aug**: the `cnn flatten` model with the same random translations.

```python
import keras
import numpy as np
from keras import layers
from examples.digits import load

x_train, x_test, y_train, y_test = load(images=True)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def shift(x, dx):
    out = np.zeros_like(x)
    if dx > 0:
        out[:, :, dx:, :] = x[:, :, :-dx, :]
    elif dx < 0:
        out[:, :, :dx, :] = x[:, :, -dx:, :]
    else:
        out = x.copy()
    return out

def build(kind):
    keras.utils.set_random_seed(0)
    stack = [keras.Input((8, 8, 1))]
    if kind in ("dense aug", "cnn aug"):
        stack.append(layers.RandomTranslation(0.125, 0.125, fill_mode="constant"))
    if kind in ("dense", "dense aug"):
        stack += [layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10)]
    elif kind in ("cnn flatten", "cnn aug"):
        stack += [layers.Conv2D(16, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
                  layers.Flatten(), layers.Dense(64, activation="relu"), layers.Dense(10)]
    else:
        stack += [layers.Conv2D(16, 3, padding="same", activation="relu"),
                  layers.Conv2D(32, 3, padding="same", activation="relu"),
                  layers.GlobalMaxPooling2D(), layers.Dense(10)]
    return keras.Sequential(stack)

print(f"{'model':12s} {'weights':>8s} {'centered':>9s} {'right 1px':>10s} {'left 1px':>9s}")
for kind in ("dense", "cnn flatten", "cnn global", "dense aug", "cnn aug"):
    model = build(kind)
    model.compile("adam", loss, metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=0)
    scores = [model.evaluate(shift(x_test, d), y_test, verbose=0)[1] for d in (0, 1, -1)]
    print(f"{kind:12s} {model.count_params():8d} {scores[0]:9.3f} {scores[1]:10.3f} {scores[2]:9.3f}")
```

```text
model         weights  centered  right 1px  left 1px
dense            4810     0.978      0.462     0.482
cnn flatten     17258     0.978      0.638     0.562
cnn global       5130     0.991      0.991     0.991
dense aug        4810     0.949      0.853     0.847
cnn aug         17258     0.991      0.951     0.936
```

Read the first column: the dense model and the first CNN both score 0.978 on centered images. They do not differ there.

Now read the last two columns. With a shift of one pixel the dense model falls to 0.462 and 0.482. The `cnn flatten` model falls to 0.638 and 0.562, which is better but not good. It has convolutions, but the flatten step gives each position its own weights again.

The `cnn global` model scores 0.991 in all three columns. It is the best model on the centered images and it does not change under the shift. It also has 5,130 weights, fewer than the 17,258 of `cnn flatten`. Its global max pool removes the position.

Augmentation is the other way to get robustness. It works for both models and it costs training time. The dense model scores 0.949 on centered images and 0.853 on the right shift. The CNN scores 0.991 and 0.951. The augmented CNN is still below the `cnn global` model on the shifted images. A structure that has the right property beats data that teaches it.

The models used three more layers.

#### `keras.layers.MaxPooling2D`

```text
keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=None,
    padding='valid',
    data_format=None,
    name=None,
    **kwargs,
)
```

The layer takes the maximum of each window. It has no weights.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `pool_size` | `(2, 2)` | the size of the window | on 7 × 7, a window of 2 gave 3 × 3: the last row and column were dropped |
| `strides` | `None` | the step. With `None`, `pool_size` | with `strides=1`, 7 × 7 gave 6 × 6, and the windows overlap |
| `padding` | `'valid'` | `"valid"` or `"same"` | with `"same"`, 7 × 7 gave 4 × 4, and no row was dropped |
| `data_format` | `None` | as for `Conv2D` | — |
| `name` | `None` | the name of the layer | — |
| `**kwargs` | — | the other arguments of `Layer` | — |

#### `keras.layers.GlobalMaxPooling2D`

```text
keras.layers.GlobalMaxPooling2D(
    data_format=None,
    keepdims=False,
    **kwargs,
)
```

The layer takes the maximum of each channel over all positions. The position is lost, which is the source of the invariance in the table.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `data_format` | `None` | as for `Conv2D` | with `"channels_first"`, the input `(7, 7, 16)` gave `(None, 7)` and not `(None, 16)`, with no error |
| `keepdims` | `False` | keep the two spatial axes with size 1 | with `True`, the output is `(None, 1, 1, 16)` and not `(None, 16)` |
| `**kwargs` | — | the arguments of `Layer` | — |

#### `keras.layers.RandomTranslation`

```text
keras.layers.RandomTranslation(
    height_factor,
    width_factor,
    fill_mode='reflect',
    interpolation='bilinear',
    seed=None,
    fill_value=0.0,
    data_format=None,
    **kwargs,
)
```

In training, the layer shifts each image by a random share of its height and width.

| parameter | default | what it does | what goes wrong |
|---|---|---|---|
| `height_factor` | required | the largest shift, as a share of the height, or a pair `(low, high)` | the shift is random and continuous: 0.125 on 8 pixels is any shift from -1 to 1 pixel, not exactly 1 pixel. A factor outside -1 to 1 raises `ValueError` |
| `width_factor` | required | the same for the width | — |
| `fill_mode` | `'reflect'` | how to fill the pixels that the shift uncovers: `"reflect"`, `"constant"`, `"nearest"` or `"wrap"` | on 200 test digits shifted by up to 2 pixels, `"reflect"` copied ink into the border (mean 0.3023 against 0.3019 before), and `"constant"` filled it with 0 (mean 0.2639). The chapter uses `"constant"` |
| `interpolation` | `'bilinear'` | how to compute a pixel at a fractional shift | with `"bilinear"`, 70% of the pixel values were no longer multiples of 1/16. With `"nearest"`, none changed in this way |
| `seed` | `None` | the seed of the layer | — |
| `fill_value` | `0.0` | the value for `fill_mode="constant"` | — |
| `data_format` | `None` | as for `Conv2D` | — |
| `**kwargs` | — | the arguments of `Layer` | — |

The layer is active in training only, and the definition of training is not the same as for `Dropout`. `model.predict` and a call with `training=False` returned the images unchanged. A direct call, and `model(x)` with no `training` argument, shifted them. `Dropout` does nothing in a direct call ([Chapter 8](08-overfitting-and-regularization.md)). Pass `training=False` when you call a model with augmentation layers yourself.

## What to take from the table

Three results are worth keeping. First, a convolution is not enough. The flatten step puts back what the filter sharing took out.

Second, a global pooling layer at the end gave the invariance at no cost in weights. Third, the choice of the layer is a statement about what does not change in the data. Use the statement that is true for your task. A task in which position carries the answer, such as reading a number from a fixed form, needs the flatten.

The invariance in the table is also limited. It holds for shifts that leave the digit inside the frame. This is a measured property of 8 × 8 digits with an empty border. It will not carry over to other data without a check.

## How it works

A `Conv2D` layer is a sliding dot product. For each output position, it takes a window of the input with the size of the kernel. It multiplies the window by the kernel, adds the products, and adds the bias. With several filters, it does this once for each filter. The kernel has the shape `(height, width, input channels, filters)`.

Keras does not flip the kernel. The operation is a cross-correlation, and the books of signal processing call a flipped version a convolution. For a learned kernel, the difference does not matter.

The next block does the sliding dot product with two loops and compares it with the layer.

```python
import keras
import numpy as np
from keras import layers, ops
from examples.digits import load

keras.utils.set_random_seed(0)

_, x_test, _, _ = load(images=True)
image = x_test[:1]
layer = layers.Conv2D(2, 3, padding="valid")
output = ops.convert_to_numpy(layer(image))
kernel, bias = (ops.convert_to_numpy(w) for w in layer.weights)
print("kernel shape:", kernel.shape, " output shape:", output.shape)

by_hand = np.zeros_like(output)
for i in range(output.shape[1]):
    for j in range(output.shape[2]):
        window = image[0, i:i + 3, j:j + 3, :]
        for f in range(output.shape[3]):
            by_hand[0, i, j, f] = (window * kernel[:, :, :, f]).sum() + bias[f]
print("two loops equal the layer:", bool(np.allclose(by_hand, output, atol=1e-5)))
print("multiplications for this image:", output.size * 3 * 3 * 1)
```

```text
kernel shape: (3, 3, 1, 2)  output shape: (1, 6, 6, 2)
two loops equal the layer: True
multiplications for this image: 648
```

The layer equals the two loops. The invariant is this: every output position uses the same kernel. The 18 numbers of the kernel are the only weights besides the 2 biases, and the layer makes 36 × 18 = 648 multiplications from them. This is the cause of the shift property in the chapter. A feature moves in the input, and the same kernel finds it at the new place.

## Reference

| Callable | Introduced in | Contract |
|---|---|---|
| `keras.layers.Conv2D` | Shapes and parameters | a sliding dot product. The weight count does not depend on the image size |
| `keras.layers.MaxPooling2D` | Five models and one pixel | the maximum of each window. A window that does not fit is dropped |
| `keras.layers.GlobalMaxPooling2D` | Five models and one pixel | the maximum of each channel over all positions |
| `keras.layers.RandomTranslation` | Five models and one pixel | a random continuous shift, active in training |

## What you have now

You have a convolutional network with 5,130 weights that scores 0.991 and that does not change when the input moves by one pixel. You also have the arithmetic of shapes and weights, and a test of equivariance.

[Chapter 11](11-sequences.md) moves from images to sequences. Exercises for this chapter are in [exercises/10.md](../exercises/10.md), and the worked solutions are in [exercises/solutions/10.md](../exercises/solutions/10.md).
