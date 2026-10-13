# Appendix A — Described, not built

This book measures what it states. A few statements are not measured, and each one is written as a statement
about the world and not as a result. They are here, so that you can see where the book's evidence stops.

| Statement | Where | Why it was not measured |
|---|---|---|
| Code that uses only `keras.ops` runs on JAX and PyTorch. | [2](../chapters/02-tensors-and-ops.md), [9](../chapters/09-custom-training.md), [15](../chapters/15-performance-and-reproducibility.md) | JAX and PyTorch were not installed. The book says what Keras prints when you ask for them. |
| A `train_step` with `tf.GradientTape` runs on TensorFlow only; `compute_loss` overrides run on every back end. | [9](../chapters/09-custom-training.md) | Same reason. The first half follows from the use of `tf`. The second half is the documented contract. |
| Mixed precision is faster on a GPU. | [15](../chapters/15-performance-and-reproducibility.md) | No GPU. The book measured the effect on the loss and the dtypes only. |
| `jit_compile=True` speeds up a model. | [15](../chapters/15-performance-and-reproducibility.md) | Keras refused it for the model of the chapter on this machine. No timing exists. |
| The cost of attention grows with the square of the length. | [12](../chapters/12-attention-and-transformers.md) | The book measured the shape of the score table, `T × T`. It did not time it. |
| The cause of the remaining run-to-run difference with `TF_DETERMINISTIC_OPS=1`. | [15](../chapters/15-performance-and-reproducibility.md) | The book measured the size of the difference and one setting that did not remove it. The cause is open. |
| Why a `conv1d` model scores 1.000 on pairs with a token at the edge. | [11](../chapters/11-sequences.md) | The edge test confirmed that edge pairs score 1.000 and interior pairs 0.504. The mechanism (fewer windows at the border) is the candidate and was not isolated further. |
| Behavior of Keras versions other than 3.15.1, and of TensorFlow other than 2.21.0. | all | Not installed. Every signature and message comes from the installed versions. |
| Behavior at scale: millions of images, long sequences, many GPUs. | all | The largest training set has 1,347 images of 8 × 8 pixels. |
