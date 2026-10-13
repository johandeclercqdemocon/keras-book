# Keras 3: From First Layer to Production

*A practical book on Keras 3 — every number in it comes from code that ships with it.*

This book is for a reader who knows a little Python and wants to build, train, debug and save neural
networks with Keras 3. It teaches the library through small models that train in seconds on a CPU: digits,
a line, a sequence-order task. Each of its 16 chapters ends with something that you built and tested.

**Nothing in it is asserted that was not run.** The numbers in the text come from the code blocks of the same
chapter. A checker runs every block and compares every printed output with the text.

Each chapter also carries
two sections that most tutorials leave out. **How it works** explains the mechanism under the API, with one
block that reproduces it. **Reference** prints the signatures of the functions that the chapter uses. The signatures come
from the installed library. A table explains the parameters that matter, with a measured consequence for each.

Where the book was wrong on its first draft, it says so: [Chapter 16](chapters/16-field-guide.md) lists the
corrections. Where it did not measure something, it says that too: [Appendix A](appendices/A-described-not-built.md).

## What it was measured on

Keras 3.15.1, TensorFlow 2.21.0 back end, Python 3.13, one CPU, no GPU. The code makes the data, or the data ship
with scikit-learn 1.9.1 (the 1,797 digit images). **The book needs no network.** JAX and PyTorch were not
installed. No result comes from them.

## Contents

| # | Chapter | You build | Run first |
|---|---|---|---|
| | **Part I — Foundations** | | |
| 1 | [The first model](chapters/01-the-first-model.md) | a line model; a table of optimizers and learning rates | `uv run python -m examples.ch01_the_first_model` |
| 2 | [Tensors and ops](chapters/02-tensors-and-ops.md) | layer norm with `keras.ops`; rules for dtypes and seeds | `uv run python -m examples.ch02_tensors_and_ops` |
| 3 | [Layers and weights](chapters/03-layers-and-weights.md) | a custom `Dense`; a counter layer; the freeze/compile rule | `uv run python -m examples.ch03_layers_and_weights` |
| 4 | [Three ways to build a model](chapters/04-three-ways-to-build.md) | one network three times; a shared-layer model | `uv run python -m examples.ch04_three_ways_to_build` |
| | **Part II — Training** | | |
| 5 | [Data pipelines](chapters/05-data-pipelines.md) | three input pipelines; four data traps | `uv run python -m examples.ch05_data_pipelines` |
| 6 | [Losses, metrics and optimizers](chapters/06-losses-metrics-optimizers.md) | a 97% classifier; the `from_logits` and clipping tables | `uv run python -m examples.ch06_losses_metrics_optimizers` |
| 7 | [Callbacks and training control](chapters/07-callbacks-and-training-control.md) | early stop, checkpoint, plateau, NaN stop; two custom callbacks | `uv run python -m examples.ch07_callbacks_and_training_control` |
| 8 | [Overfitting and regularization](chapters/08-overfitting-and-regularization.md) | four regularizers, a learning curve | `uv run python -m examples.ch08_overfitting_and_regularization` |
| 9 | [Custom losses, metrics and training steps](chapters/09-custom-training.md) | a loss, a metric and a `train_step`, each equal to its built-in | `uv run python -m examples.ch09_custom_training` |
| | **Part III — Architectures** | | |
| 10 | [Convolutional networks](chapters/10-convolutional-networks.md) | a CNN that does not change under a one-pixel shift | `uv run python -m examples.ch10_convolutional_networks` |
| 11 | [Sequences](chapters/11-sequences.md) | an order test; the reach of a filter; masks | `uv run python -m examples.ch11_sequences` |
| 12 | [Attention and transformers](chapters/12-attention-and-transformers.md) | attention by hand; a transformer block; positions | `uv run python -m examples.ch12_attention_and_transformers` |
| 13 | [Transfer learning](chapters/13-transfer-learning.md) | frozen, fine-tuned and scratch models under three budgets | `uv run python -m examples.ch13_transfer_learning` |
| | **Part IV — Production** | | |
| 14 | [Saving, loading and export](chapters/14-saving-loading-export.md) | an exact round trip; four load faults; an exported model | `uv run python -m examples.ch14_saving_loading_export` |
| 15 | [Performance and reproducibility](chapters/15-performance-and-reproducibility.md) | a determinism test; speed options measured for their effect | `uv run python -m examples.ch15_performance_and_reproducibility` |
| 16 | [A field guide](chapters/16-field-guide.md) | a symptom → cause → chapter table; the book's own corrections | `uv run python -m examples.ch16_field_guide` |

Exercises are in [`exercises/`](exercises/) (`NN.md`), worked solutions in [`exercises/solutions/`](exercises/solutions/).
Each chapter has three.

A **[measure]** exercise asks for a number, and it gives the expected answer. A **[break]** exercise changes one thing so that a result reverses, and asks why. A **[judge]** exercise has no single right answer. Its solution gives a position, the strongest objection and the evidence that can overturn it. The solutions execute.

## Run it

```bash
uv sync
uv run python -m examples.ch01_the_first_model
```

The script `scripts/extract_examples.py` generates the files `examples/chNN_*.py` from the python blocks of the chapters. They are the chapter, in order, as a program.

Importing the `examples` package
sets `KERAS_BACKEND`, `TF_DETERMINISTIC_OPS` and `TF_ENABLE_ONEDNN_OPTS` before TensorFlow loads. If you copy a block out
of a chapter and run it elsewhere, set them yourself. [Chapter 15](chapters/15-performance-and-reproducibility.md)
says what the second one does and what it does not do.

## Check it

```bash
uv run python scripts/check_snippets.py            # links, headers, every python block, every printed output
python3 scripts/audit_numbers.py --tables          # parameter tables against the printed signatures
python3 scripts/audit_numbers.py                   # numbers in prose that no output contains
```

The first command runs about a hundred TensorFlow processes and takes a while. Run it on an idle machine.
Printed values are rounded to 3 or 4 decimals. Two runs of the same training can differ in the eighth decimal of a
loss. The book compares at the precision that it prints.
