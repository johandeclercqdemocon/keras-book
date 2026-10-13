"""Train the same small CNN once and print a fingerprint of the result.

The fingerprint is a hash of the bytes of all weights, then the last training loss. Equal fingerprints
mean equal weights bit for bit. Chapter 15 runs this module several times in separate processes and
counts the distinct fingerprints.
"""
import hashlib
import os

import keras
import numpy as np
from keras import layers

from examples.digits import load


def main() -> None:
    threads = int(os.environ.get("KERAS_BOOK_THREADS", "0"))
    if threads:
        import tensorflow as tf

        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.config.threading.set_inter_op_parallelism_threads(threads)
    x_train, _, y_train, _ = load(images=True)
    keras.utils.set_random_seed(0)
    model = keras.Sequential([
        keras.Input((8, 8, 1)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.GlobalMaxPooling2D(),
        layers.Dense(10),
    ])
    model.compile("adam", keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(x_train, y_train, epochs=40, batch_size=32, verbose=0).history
    digest = hashlib.sha256(b"".join(w.tobytes() for w in model.get_weights())).hexdigest()[:12]
    print(f"{digest} {history['loss'][-1]:.8f}")


if __name__ == "__main__":
    main()
