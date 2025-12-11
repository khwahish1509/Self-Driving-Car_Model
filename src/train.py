import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.generator import DataGenerator
from src.model import nvidia_model
import tensorflow as tf
import matplotlib.pyplot as plt




def load_samples(csv_path):
    df = pd.read_csv(csv_path, header=None)
    # expect two columns: image_path, steering
    samples = df.values.tolist()
    return samples




def main(args):
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    samples = load_samples(args.csv)
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)
    batch_size = args.batch
    train_gen = DataGenerator(train_samples, batch_size=batch_size, is_training=True)
    val_gen = DataGenerator(val_samples, batch_size=batch_size, is_training=False)


    model = nvidia_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')


    steps_per_epoch = max(1, len(train_samples) // batch_size)
    validation_steps = max(1, len(val_samples) // batch_size)


    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.model_out, monitor='val_loss', save_best_only=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)


    history = model.fit(
    train_gen.generator(),
    steps_per_epoch=steps_per_epoch,
    epochs=args.epochs,
    validation_data=val_gen.generator(),
    validation_steps=validation_steps,
    callbacks=[checkpoint, reduce_lr, early]
    )


# plot
plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('logs/training_plots/loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--model_out', default='logs/models/model_best.h5')
    args = parser.parse_args()
    main(args)