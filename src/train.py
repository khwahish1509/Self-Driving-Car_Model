import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.generator import DataGenerator
from src.model import nvidia_model
import tensorflow as tf
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load CSV samples
# ---------------------------------------------------------
def load_samples(csv_path):
    df = pd.read_csv(csv_path, header=None)
    samples = df.values.tolist()
    return samples


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------
def main(args):
    # Ensure output directories exist
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    os.makedirs("logs/training_plots", exist_ok=True)

    samples = load_samples(args.csv)
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    batch_size = args.batch

    # Generators
    train_gen = DataGenerator(train_samples, batch_size=batch_size, is_training=True)
    val_gen = DataGenerator(val_samples, batch_size=batch_size, is_training=False)

    # Model
    model = nvidia_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')

    steps_per_epoch = max(1, len(train_samples) // batch_size)
    validation_steps = max(1, len(val_samples) // batch_size)

    print("\n=== TRAINING STARTED ===")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Batch size: {batch_size}\n")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        args.model_out, monitor="val_loss", save_best_only=True, verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, verbose=1
    )
    early = tf.keras.callbacks.EarlyStopping(
          monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
    )

    # ---------------------------------------------------------
    # TRAINING with full error logging
    # ---------------------------------------------------------
    try:
        history = model.fit(
            train_gen.generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            validation_data=val_gen.generator(),
            validation_steps=validation_steps,
            callbacks=[checkpoint, reduce_lr, early],
            verbose=1
        )

    except Exception as e:
        print("\nTRAINING FAILED!")
        print("Error message:")
        print("-------------------------------------")
        print(e)
        print("-------------------------------------")
        print("\nThe issue is likely:")
        print("• Missing image files")
        print("• Wrong image paths in CSV")
        print("• OpenCV cannot read an image")
        print("• Broken augmentation")
        print("\nPlease send me the full error and I will fix it.")
        return

    print("\n=== TRAINING COMPLETED SUCCESSFULLY ===")

    # ---------------------------------------------------------
    # PLOT TRAINING CURVES
    # ---------------------------------------------------------
    plt.figure()
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("logs/training_plots/loss.png")

    print("\nSaved training plot to logs/training_plots/loss.png")
    print(f"Best model saved to: {args.model_out}")


# ---------------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--model_out", default="logs/models/model_best.h5")
    args = parser.parse_args()

    main(args)
