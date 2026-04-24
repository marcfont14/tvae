import tensorflow as tf
from tensorflow import keras


def train(model, splits, run_id, results_dir, epochs=50, lr=1e-3,
          patience=10, clipnorm=1.0):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
        loss=keras.losses.Huber(delta=1.0),
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience // 2, min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        splits['train'],
        validation_data=splits['val'],
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history
