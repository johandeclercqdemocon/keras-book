# Appendix B — What went wrong in the parameter audit

[Chapter 16](../chapters/16-field-guide.md) lists the errors of the first draft. This appendix lists the errors that a second pass found. In that pass, every parameter of every printed signature got a row, and every claim in a row was run first. The errors are fixed in the text. They are listed here because the same habits come back.

## Wrong in this book

| Chapter | The wrong statement | What the run or the audit showed | How it was caught |
|---|---|---|---|
| all | each chapter explains the parameters of the functions that it uses | the Reference tables at the end of each chapter had rows for 3 to 11 parameters of each chapter, of 418 in all. For example, `ops.convert_to_tensor` showed `sparse` and `ragged` in its signature and had no row for them. No check tied a row to a signature | a reader found two missing rows on page 4 of a sister book. Each callable now has a `####` block at its first use, and the checker fails for a parameter without a row |
| [7](../chapters/07-callbacks-and-training-control.md) | `TerminateOnNaN` "has no setting" | it has `raise_error=False`. With `True`, the same run raised `RuntimeError: NaN or Inf loss encountered at batch 3` | the printed signature of the new block |
| [9](../chapters/09-custom-training.md) | gradient accumulation needs a custom `train_step` | every optimizer has `gradient_accumulation_steps`. With 4, one epoch of 43 batches made 10 updates | the printed signature of `SGD` in [Chapter 6](../chapters/06-losses-metrics-optimizers.md) |
| [10](../chapters/10-convolutional-networks.md) | `RandomTranslation(0.125, 0.125)` on 8 pixels "is 1 pixel" | the shift is random and continuous, from -1 to 1 pixel. With the default bilinear interpolation, 70% of the pixel values were no longer multiples of 1/16 | running the row |
| [6](../chapters/06-losses-metrics-optimizers.md) | in a draft row, two `SparseTopKCategoricalAccuracy` metrics with the default name collide | Keras renamed the second one `sparse_top_k_categorical_accuracy_1`. Nothing collides, and the key does not say which `k` it holds | running the row after it was written. The rule is to run it before |
| [1](../chapters/01-the-first-model.md) | in a draft, `predict(x, steps=1)` fails on a NumPy array | on a new model it works and returns all 64 rows. The failure came from an earlier `predict` on a rank-1 array of the same model, which breaks every later `predict` | a second run on a new model |
| [11](../chapters/11-sequences.md) | in draft rows, three statements with no run: a model with a fixed input length fails on some batches, a second recurrent layer needs `return_sequences=True` on the first, and the weights of the two forms of `reset_after` do not load into each other | the first was removed. The other two were run: `ValueError: ... expected ndim=3, found ndim=2`, and `ValueError` with the bias shapes `(2, 96)` and `(96,)` | a pass over each new row for a number or a message from a run |
| [7](../chapters/07-callbacks-and-training-control.md) | the expectation, before the run, that `EarlyStopping(patience=5, baseline=0.1)` stops after 5 epochs above the baseline | it ran 29 epochs, the same as `patience=0`. A new best value ends the check of the epoch before the stop test | the run, then the source of `EarlyStopping.on_epoch_end` |

## What the documentation or the usual advice says, and Keras 3.15.1 does not do

| Chapter | The statement | What the run showed |
|---|---|---|
| [1](../chapters/01-the-first-model.md) | the docstring of `Model.summary`: Keras calls `print_fn` on each line | it called it 1 time, with the whole table in one string |
| [4](../chapters/04-three-ways-to-build.md) | layers reject an unknown keyword | `keras.Model` accepted `foo=1` with no error. `inputs=` without `outputs=` raised a message about an unrecognized `inputs` |
| [7](../chapters/07-callbacks-and-training-control.md) | an argument with the wrong name raises an error | `ReduceLROnPlateau` accepted `foo=1` and the old name `epsilon=0.5` and ignored both. `min_delta` stayed at 0.0001 |
| [10](../chapters/10-convolutional-networks.md) | a random augmentation layer is active only in training | a direct call, and `model(x)` with no `training` argument, shifted the images. `Dropout` does nothing in a direct call |
| [8](../chapters/08-overfitting-and-regularization.md) | `GaussianNoise(stddev)` takes any standard deviation | values above 1 raise `ValueError`, so noise on raw pixels from 0 to 16 needs a scaled input |
| [12](../chapters/12-attention-and-transformers.md) | a layer normalization gives each row the standard deviation 1 | with the default `epsilon=0.001`, a row with the variance 0.002 came out with 0.816 |
| [14](../chapters/14-saving-loading-export.md) | `save(overwrite=False)` refuses to replace a file | it asks a question on the terminal. With no terminal, it raised `EOFError` |

## Checks that found nothing

Five checks found no error at the last run: the links, the contracts, the signature of every `####` block, the default in every row, and the `keras.*` names in the text.
