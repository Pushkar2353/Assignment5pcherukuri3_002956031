import os

# Required by the Ray TensorFlow integration used in your
# current environment.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import ray
import tensorflow as tf

from ray import train
from ray.train import (
    ScalingConfig,
    RunConfig,
)

from ray.train.tensorflow import (
    TensorflowTrainer,
)


METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "problem2_tensorflow_ray_metrics.txt"
)


# ============================================================
# Load MNIST
# ============================================================

def load_mnist():

    (
        x_train,
        y_train,
    ), (
        x_test,
        y_test,
    ) = (
        tf.keras.datasets
        .mnist
        .load_data()
    )


    # Use 10% = 6,000 images.
    percent = 0.1


    subset_size = int(
        len(x_train)
        * percent
    )


    x_train = x_train[
        :subset_size
    ]

    y_train = y_train[
        :subset_size
    ]


    # Normalize.
    x_train = (
        x_train
        / np.float32(255)
    )

    x_test = (
        x_test
        / np.float32(255)
    )


    y_train = y_train.astype(
        np.int64
    )

    y_test = y_test.astype(
        np.int64
    )


    return (
        x_train,
        y_train,
        x_test,
        y_test,
    )


# ============================================================
# Required CNN
# ============================================================

def build_model():

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(28, 28)
            ),

            tf.keras.layers.Reshape(
                target_shape=(
                    28,
                    28,
                    1,
                )
            ),

            tf.keras.layers.Conv2D(
                16,
                (3, 3),
                strides=1,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Conv2D(
                32,
                (3, 3),
                strides=1,
                padding="same",
                activation="relu",
            ),

            tf.keras.layers.MaxPooling2D(
                (2, 2)
            ),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(
                128,
                activation="relu",
            ),

            tf.keras.layers.Dense(
                10,
                activation="softmax",
            ),
        ]
    )


    return model


# ============================================================
# Ray worker
# ============================================================

def train_func(config):

    context = train.get_context()

    world_size = (
        context.get_world_size()
    )

    world_rank = (
        context.get_world_rank()
    )


    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = load_mnist()


    print(
        "Ray Train world size:",
        world_size,
    )

    print(
        "Ray Train world rank:",
        world_rank,
    )

    print(
        "Training images:",
        len(x_train),
    )

    print(
        "Testing images:",
        len(x_test),
    )


    # --------------------------------------------------------
    # TensorFlow multi-worker strategy
    # --------------------------------------------------------

    strategy = (
        tf.distribute
        .MultiWorkerMirroredStrategy()
    )


    with strategy.scope():

        model = build_model()


        model.compile(
            optimizer="adam",

            loss=(
                "sparse_categorical_"
                "crossentropy"
            ),

            metrics=[
                "accuracy"
            ],
        )


    per_worker_batch_size = (
        config.get(
            "batch_size",
            64,
        )
    )


    global_batch_size = (
        per_worker_batch_size
        * world_size
    )


    train_dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                x_train,
                y_train,
            )
        )
        .shuffle(
            len(x_train)
        )
        .batch(
            global_batch_size
        )
    )


    epochs = config.get(
        "epochs",
        10,
    )


    model.fit(
        train_dataset,
        epochs=epochs,
        verbose=2,
    )


    test_loss, test_accuracy = (
        model.evaluate(
            x_test,
            y_test,
            verbose=0,
        )
    )


    print(
        "\nTensorFlow + Ray MNIST "
        "Test Accuracy:",
        f"{test_accuracy:.4f}",
    )


    # --------------------------------------------------------
    # Rank 0 writes final metrics
    # --------------------------------------------------------

    if world_rank == 0:

        with open(
            METRICS_FILE,
            "w",
        ) as file:

            file.write(
                "TensorFlow + Ray "
                "MNIST CNN\n"
            )

            file.write(
                f"Training images: "
                f"{len(x_train)}\n"
            )

            file.write(
                f"Testing images: "
                f"{len(x_test)}\n"
            )

            file.write(
                f"Ray Train workers: "
                f"{world_size}\n"
            )

            file.write(
                "GPU used: False\n"
            )

            file.write(
                f"Epochs: "
                f"{epochs}\n"
            )

            file.write(
                f"Per-worker batch size: "
                f"{per_worker_batch_size}\n"
            )

            file.write(
                f"Global batch size: "
                f"{global_batch_size}\n"
            )

            file.write(
                "Conv1: 16 filters, "
                "3x3, stride 1, padding 1\n"
            )

            file.write(
                "Conv2: 32 filters, "
                "3x3, stride 1, padding 1\n"
            )

            file.write(
                f"Test loss: "
                f"{test_loss:.6f}\n"
            )

            file.write(
                f"Test accuracy: "
                f"{test_accuracy:.6f}\n"
            )


        print(
            "\nCreated:",
            METRICS_FILE,
        )


    # All workers report once.
    train.report(
        {
            "test_accuracy":
                float(
                    test_accuracy
                ),

            "test_loss":
                float(
                    test_loss
                ),
        }
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    ray.init(
        address="auto"
    )


    print(
        "Connected to Ray cluster."
    )


    trainer = TensorflowTrainer(

        train_loop_per_worker=(
            train_func
        ),

        train_loop_config={
            "epochs": 10,
            "batch_size": 64,
        },

        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
        ),

        run_config=RunConfig(
            storage_path="/mnt/shared"
        ),
    )


    # Important:
    # Do not use result.metrics afterward.
    result = trainer.fit()


    print(
        "\n========================================"
    )

    print(
        "TensorFlow + Ray MNIST "
        "training completed successfully."
    )

    print(
        "========================================"
    )

    print(
        "Metrics file:",
        METRICS_FILE,
    )


    ray.shutdown()
