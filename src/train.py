from pathlib import Path
from typing import Tuple
from PIL import Image
from loguru import logger
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from dvclive import Live
from dvclive.keras import DVCLiveCallback

from unet import unet_model


# generator for data
def image_data_generator(
    image_dir: Path,
    ground_truth_dir: Path,
    image_indexes: np.ndarray,
    batch_size: int,
    model_image_size: Tuple[int, int],
    norm_upper_bound: float,
    norm_lower_bound: float,
):
    """Generate batches of images and ground truth masks."""

    while True:
        # Select files for the batch
        batch_indexes = np.random.choice(a=image_indexes, size=batch_size, replace=False)
        batch_input = []
        batch_output = []

        # Load images and ground truth
        for index in batch_indexes:
            # Load the image and ground truth
            image = np.load(image_dir / f"image_{index}.npy")
            ground_truth = np.load(ground_truth_dir / f"mask_{index}.npy").astype(bool)

            # TODO: Augment the images: Scale and translate

            # Resize without interpolation
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(model_image_size, resample=Image.NEAREST)
            image = np.array(pil_image)

            pil_ground_truth = Image.fromarray(ground_truth)
            pil_ground_truth = pil_ground_truth.resize(model_image_size, resample=Image.NEAREST)
            ground_truth = np.array(pil_ground_truth)

            # Normalise the image
            image = np.clip(image, norm_lower_bound, norm_upper_bound)
            image = image - norm_lower_bound
            image = image / (norm_upper_bound - norm_lower_bound)

            # TODO: Augment images: Flipping and rotate

            # Add the image and ground truth to the batch
            batch_input.append(image)
            batch_output.append(ground_truth)

        # Convert the lists to numpy arrays
        batch_x = np.array(batch_input).astype(np.float32)
        batch_y = np.array(batch_output).astype(np.float32)

        yield (batch_x, batch_y)


def train_model(
    image_dir: Path,
    ground_truth_dir: Path,
    model_save_dir: Path,
    model_image_size: Tuple[int, int],
    batch_size: int,
    epochs: int,
    norm_upper_bound: float,
    norm_lower_bound: int,
    test_size: float,
):
    """Train a model to segment images."""

    # Set the random seed
    seed = 0
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Find the maximum index of images and ground truth
    num_images = len(list(image_dir.glob("image_*.npy")))
    num_masks = len(list(ground_truth_dir.glob("mask_*.npy")))
    if num_images != num_masks:
        raise ValueError("Different number of images and masks.")

    # Train test split
    image_indexes = range(0, num_images)

    # Create an image data generator
    train_indexes, validation_indexes = train_test_split(image_indexes, test_size=test_size, random_state=SEED)
    logger.info(f"Training on {len(train_indexes)} images | validating on {len(validation_indexes)} images.")

    train_generator = image_data_generator(
        image_dir=image_dir,
        ground_truth_dir=ground_truth_dir,
        image_indexes=train_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    validation_generator = image_data_generator(
        image_dir=image_dir,
        ground_truth_dir=ground_truth_dir,
        image_indexes=validation_indexes,
        batch_size=batch_size,
        model_image_size=model_image_size,
        norm_upper_bound=norm_upper_bound,
        norm_lower_bound=norm_lower_bound,
    )

    # Load the model
    model = unet_model(IMG_HEIGHT=model_image_size[0], IMG_WIDTH=model_image_size[1], IMG_CHANNELS=1)

    steps_per_epoch = len(train_indexes) // batch_size
    logger.info(f"Steps per epoch: {steps_per_epoch}")

    # At the end of each epoch, DVCLive will log the metrics
    with Live() as live:

        history = model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=steps_per_epoch,
            verbose=1,
            callbacks=[DVCLiveCallback(live=live)],
        )

        model.save("mymodel")
        live.log_artifact("mymodel", type="model")

        # loss = history.history["loss"]
        # val_loss = history.history["val_loss"]
        # epoch_indexes = range(1, len(loss) + 1)
        # plt.plot(epoch_indexes, loss, "bo", label="Training loss")
        # plt.plot(epoch_indexes, val_loss, "b", label="Validation loss")
        # plt.title("Training and validation loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.legend()
        # plt.show()

        # date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Save the model
        # model.save(model_save_dir / f"model_{date}.h5")

        # Report the accuracy to dvclive
