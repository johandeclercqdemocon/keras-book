"""Runnable code of "Keras 3: From First Layer to Production".

Importing this package sets three environment variables before TensorFlow loads, so that
`python -m examples.chNN_...` gives the numbers that the book prints:

  KERAS_BACKEND          the book runs on the TensorFlow back end
  TF_DETERMINISTIC_OPS   narrows run-to-run differences of CPU training; it does not remove them (Chapter 15)
  TF_ENABLE_ONEDNN_OPTS  oneDNN changes rounding in the last digits

If you copy a block out of a chapter and run it elsewhere, set these variables yourself.
"""
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
