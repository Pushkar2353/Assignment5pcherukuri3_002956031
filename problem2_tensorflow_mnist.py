import tensorflow as tf

from tensorflow.keras import (
    layers,
    models,
)

from tensorflow.keras.datasets import (
    mnist,
)


METRICS_FILE = (
    "/mnt/shared/Assignment5/"
    "problem2_tensorflow_metrics.txt"
)


# ============================================================
# Load MNIST dataset
# ============================================================

(
    x_train,
    y_train,
), (
    x_test,
    y_test,
) = mnist.load_data()


print(
    "Original training images:",
    len(x_train),
)

print(
    "Testing images:",
    len(x_test),
)


# ============================================================
# Use 10% of training data
#
# 60,000 × 0.10 = 6,000 images
# ============================================================

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


print(
    "Training images used:",
    len(x_train),
)


# ============================================================
# Normalize image values from [0,255] to [0,1]
# ============================================================

x_train = (
    x_train
    / 255.0
)

x_test = (
    x_test
    / 255.0
)


# ============================================================
# Required CNN architecture
# ============================================================

model = models.Sequential(
    [
        layers.Input(
            shape=(28, 28)
        ),

        layers.Reshape(
            target_shape=(
                28,
                28,
                1,
            )
        ),

        # ----------------------------------------------------
        # Convolution Layer 1
        #
        # 16 filters
        # 3x3
        # stride 1
        # padding 1 = "same"
        # ReLU
        # ----------------------------------------------------

        layers.Conv2D(
            16,
            (3, 3),
            strides=1,
            padding="same",
            activation="relu",
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------------------
        # Convolution Layer 2
        #
        # 32 filters
        # 3x3
        # stride 1
        # padding 1 = "same"
        # ReLU
        # ----------------------------------------------------

        layers.Conv2D(
            32,
            (3, 3),
            strides=1,
            padding="same",
            activation="relu",
        ),

        layers.MaxPooling2D(
            (2, 2)
        ),

        layers.Flatten(),

        layers.Dense(
            128,
            activation="relu",
        ),

        layers.Dense(
            10,
            activation="softmax",
        ),
    ]
)


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ],
)


print(
    "\nTensorFlow MNIST CNN"
)

model.summary()


# ============================================================
# Train
# ============================================================

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_data=(
        x_test,
        y_test,
    ),
    shuffle=True,
    verbose=2,
)


# ============================================================
# Evaluate
# ============================================================

test_loss, test_accuracy = (
    model.evaluate(
        x_test,
        y_test,
        verbose=0,
    )
)


print(
    "\n================================"
)

print(
    "TensorFlow MNIST Results"
)

print(
    "================================"
)

print(
    "Training images:",
    len(x_train),
)

print(
    "Testing images:",
    len(x_test),
)

print(
    "Test Loss:",
    f"{test_loss:.6f}",
)

print(
    "Test Accuracy:",
    f"{test_accuracy:.4f}",
)


# ============================================================
# Save results
# ============================================================

with open(
    METRICS_FILE,
    "w",
) as file:

    file.write(
        "TensorFlow MNIST CNN\n"
    )

    file.write(
        "Training images: "
        f"{len(x_train)}\n"
    )

    file.write(
        "Testing images: "
        f"{len(x_test)}\n"
    )

    file.write(
        "Epochs: 10\n"
    )

    file.write(
        "Batch size: 64\n"
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
