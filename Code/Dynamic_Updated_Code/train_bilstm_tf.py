import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------ Load splits ------------
X_train = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_train.npy").astype(np.float32)
y_train = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_train.npy").astype(np.int64)
X_val   = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_val.npy").astype(np.float32)
y_val   = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_val.npy").astype(np.int64)
X_test  = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/X_test.npy").astype(np.float32)
y_test  = np.load("/Users/nahidkhan/Local Drive/Research/Landmark_Data/y_test.npy").astype(np.int64)

num_classes = len(np.unique(y_train))
T, F = X_train.shape[1], X_train.shape[2]

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ------------ Model ------------
inputs = keras.Input(shape=(T, F))
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.4))(inputs)
x = layers.Bidirectional(layers.LSTM(128, dropout=0.4))(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------ Callbacks ------------
cb = [
    keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("/Users/nahidkhan/Local Drive/Research/Landmark_Data/best_bilstm_tf.keras",
                                    save_best_only=True, monitor="val_accuracy")
]

# ------------ Train ------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=cb,
    verbose=1
)

# ------------ Test ------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\nTEST ACC:", test_acc)
