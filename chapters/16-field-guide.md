# Chapter 16 — A field guide

> **Build** — a table from symptom to cause to chapter, a check that every row of the table points at a real chapter, the list of errors that this book made and corrected while it was written, and the list of things it did not measure.
>
> **Theory kept** — most failures in a Keras project are not exceptions. They are results that look plausible. A measurement with a number is the only reliable way to tell.
>
> **Source** — the earlier chapters of this book. The code is New.
>
> **Run this first** — `uv run python -m examples.ch16_field_guide`, which is [`examples/ch16_field_guide.py`](../examples/ch16_field_guide.py).

## From symptom to cause

The table lists what you see, the most likely cause, the measurement of the book that shows it, and the chapter. Find your symptom in the first column.

| You see | Likely cause | Measured in this book | Chapter |
|---|---|---|---|
| the loss is `nan` after some epochs | the learning rate is too large for the scale of the input | `SGD(1.5)` on the line gave `nan`; raw pixels with SGD gave 0.296 | [1](01-the-first-model.md), [5](05-data-pipelines.md) |
| a run finishes with no error and a bad score | a wrong flag gives a valid loss for the wrong problem | logits with `from_logits=False`: 0.5711 against 0.9711 | [6](06-losses-metrics-optimizers.md) |
| training accuracy is high and validation accuracy is near zero | `validation_split` takes the last rows of sorted data | 0.015 validation accuracy | [5](05-data-pipelines.md) |
| a frozen layer keeps training, or an unfrozen layer stays frozen | `trainable` changed after `fit` without a new `compile` | all four cases printed | [3](03-layers-and-weights.md) |
| a value of 10,000,000 at test time | a feature that is constant in the training data | `Normalization` divided by 1e-7 | [5](05-data-pipelines.md) |
| the returned model is not the best one | `restore_best_weights` is `False` by default | loss 0.2186 against 0.2176 | [7](07-callbacks-and-training-control.md) |
| the loss in a callback is not the loss of the batch | the logs hold the running average of the epoch | last batch equals the epoch loss | [7](07-callbacks-and-training-control.md) |
| a regularizer lowers the gap and not the error | the gap is not the score | all four within 3 images of 450 | [8](08-overfitting-and-regularization.md) |
| a custom metric gives a plausible wrong value | `reset_state` does not clear the state | 0.6929 against 0.8934 | [9](09-custom-training.md) |
| a CNN is not robust to a one-pixel shift | flatten gives each position its own weights | 0.638 against 0.991 with a global max pool | [10](10-convolutional-networks.md) |
| a sequence model is at chance | the model has no sense of order | bag of tokens 0.502 | [11](11-sequences.md) |
| a transformer is at chance | no position signal | 0.502 without, 1.000 with | [12](12-attention-and-transformers.md) |
| a model with positions has the same weight count as one without | the embedding was called outside a layer and is not tracked | 8,897 against 9,409 | [12](12-attention-and-transformers.md) |
| a model fails to load with "expected 2 variables" | a constructor argument is not in `get_config` | `width=7` fails, with `get_config` loads | [14](14-saving-loading-export.md) |
| a loaded model trains with a different loss | the loss class has no `get_config` | factor 0.5 came back as 0.1 | [14](14-saving-loading-export.md) |
| the same seed gives different results | order of floating-point operations, or layer order | 2 weight hashes in 8 runs; layer kernels differ | [15](15-performance-and-reproducibility.md) |

The next block checks the table against the files of the book. It reads this chapter, takes every chapter number in the last column and tests that the chapter file exists.

```python
import pathlib
import re

root = pathlib.Path(".")
text = (root / "chapters" / "16-field-guide.md").read_text(encoding="utf-8")
links = re.findall(r"\[(\d+)\]\((\d\d-[a-z-]+\.md)\)", text)
missing = [name for _, name in links if not (root / "chapters" / name).exists()]
numbers_match = all(name.startswith(f"{int(number):02d}-") for number, name in links)
print("links in the table:", len(links))
print("missing chapter files:", missing)
print("the number in each link matches its file name:", numbers_match)
```

```text
links in the table: 17
missing chapter files: []
the number in each link matches its file name: True
```

## Errors that this book made and fixed

A book that shows its corrections is easier to trust. Each item below was found by a measurement and not by a reader. The text of the chapter now says the measured version.

| Chapter | The first draft said | The measurement said |
|---|---|---|
| 3 | exercise 1 expected 237 trainable parameters | 241; the by-hand count agrees |
| 6 | exercise 2 expected a top-3 accuracy above 0.95 | 0.8822 |
| 6 | `from_logits=False` divides by the row sum, then clips | it clips, then takes the log; the first formula gave 8.5090, Keras gave 13.2655 |
| 7 | the batch log is the loss of that batch | it is the running average of the epoch |
| 10 | a global max pool makes the output exactly shift-invariant | the maximum was the same for 97.9% of the image-channel pairs |
| 11 | the accuracy of the `conv` model falls steadily with the distance | it did not; token position at the edge leaks |
| 12 | a position embedding called outside a layer is a trained table | it is a fixed table with 0 trained weights; the count stayed at 8,897 |
| 4 | a subclassed model gives up saving without extra code | it saves and loads when the class is registered, `__init__` passes `**kwargs`, and `get_config` returns every non-default argument |
| 9 | a loss without `get_config` cannot be rebuilt | it loads, and its arguments return to the defaults |
| 6 | the accuracy of the broken `from_logits=False` setup is 0.7089 | 0.5711 with the determinism setting on; the setup is unstable, so no single number is firm |
| 13 | the frozen base loses to training from scratch | the head had not converged; at a larger budget the frozen base was ahead, and at a third budget the three modes tied at 20 images for each class |
| 14 | the export signature takes the key `keras_tensor` | the name depends on how many tensors the process made before; the key failed in the chapter module and works with a named input |
| 15 | a `Dense` layer without a seed fails under determinism | the shuffle in `fit` is the random operation; building the layer did not fail |
| all | outputs of CNN training are repeatable with a seed | 2 weight hashes in 8 runs; `TF_DETERMINISTIC_OPS=1` narrowed the loss spread from 1.16e-05 to 1.00e-08 and did not remove it |

## What this book did not measure

- **Speed.** All runs were on one CPU. `jit_compile` was refused for the model of Chapter 15, and no timing is reported.
- **Other back ends.** JAX and PyTorch were not installed. "Portable" means "uses only `keras.ops`", and not "was run on three back ends".
- **Accelerators.** No GPU or TPU was used. Mixed precision was run for its numbers and not for its speed.
- **Large data.** The largest training set has 1,347 images of 8 × 8 pixels. The traps are the same at scale. The sizes of the effects are not.
- **Other versions.** Every signature and every message comes from Keras 3.15.1 with TensorFlow 2.21.0. A later version can differ.

## What you have now

You have a table that takes you from a symptom to a measurement. You also have a list of the mistakes that this book made and a list of what it did not test. Use the table before you debug.

Exercises for this chapter are in [exercises/16.md](../exercises/16.md), and the worked solutions are in [exercises/solutions/16.md](../exercises/solutions/16.md).
