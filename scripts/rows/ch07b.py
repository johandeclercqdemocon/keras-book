# Runs every claim of the parameter tables of one chapter. Run from the book root:
#     PYTHONPATH=. python scripts/rows/ch07b.py
# The output is the evidence for the numbers and messages in the "what goes wrong" column.
import examples, keras, numpy as np
from keras import layers
from examples.digits import load
x_train, x_test, y_train, y_test = load()
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def run(**kw):
    keras.utils.set_random_seed(0)
    m = keras.Sequential([keras.Input((64,)), layers.Dense(64, activation="relu"), layers.Dense(10)])
    m.compile(keras.optimizers.Adam(0.01), loss, metrics=["accuracy"])
    h = m.fit(x_train, y_train, epochs=15, batch_size=32, validation_data=(x_test, y_test), verbose=0,
              callbacks=[keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=0.0001, **kw)])
    return [round(v, 5) for v in h.history["learning_rate"]]
print("chapter setup :", run())
print("cooldown=2    :", run(cooldown=2))
print("min_delta=0.01:", run(min_delta=0.01))
r = keras.callbacks.ReduceLROnPlateau(epsilon=0.5); print("epsilon=0.5 -> min_delta", r.min_delta)
