import os
import tensorflow as tf
from tensorflow import keras


def train(model, splits, run_id, results_dir, epochs=50, lr=1e-3,
          patience=10, clipnorm=1.0, loss=None):
    if loss is None:
        loss = keras.losses.Huber(delta=1.0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
        loss=loss,
    )

    weights_path = os.path.join(results_dir, f'weights_{run_id}.weights.h5')
    os.makedirs(results_dir, exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience // 2, min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=weights_path,
            monitor='val_loss', save_best_only=True,
            save_weights_only=True, verbose=0,
        ),
    ]

    history = model.fit(
        splits['train'],
        validation_data=splits['val'],
        validation_steps=splits.get('val_steps'),
        epochs=epochs,
        steps_per_epoch=splits.get('steps_per_epoch'),
        callbacks=callbacks,
        verbose=1,
    )

    return history
