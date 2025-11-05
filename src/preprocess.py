"""Scripts for preprocessing images before passing to the model."""

from PIL import Image
import numpy as np
import numpy.typing as npt
# pylint: disable=no-name-in-module
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def resize_image(
    image: npt.NDArray[np.float64],
    size: tuple[int, int],
) -> npt.NDArray[np.float64]:
    """Resize the image to the desired size."""
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize(size, resample=Image.NEAREST)
    return np.array(pil_image)

def resize_mask(
    mask: npt.NDArray[np.bool_],
    size: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Resize the mask to the desired size."""
    pil_mask = Image.fromarray(mask)
    pil_mask = pil_mask.resize(size, resample=Image.NEAREST)
    return np.array(pil_mask).astype(bool)

def normalise_image(
    image: npt.NDArray[np.float64],
    norm_upper_bound: float,
    norm_lower_bound: float
) -> npt.NDArray[np.float64]:
    """Normalise the image to the range [0, 1] based on the provided bounds."""
    # Normalise the image
    image = np.clip(image, norm_lower_bound, norm_upper_bound)
    image = image - norm_lower_bound
    image = image / (norm_upper_bound - norm_lower_bound)
    return image

def apply_hessian_filter(
    image: npt.NDArray[np.float64],
    hessian_component: str,
    sigma: int = 1
) -> npt.NDArray[np.float64]:
    """Apply a Hessian filter to the image"""
    hessian_matrix_image = hessian_matrix(image, sigma=sigma, order="rc", use_gaussian_derivatives=False)
    hessian_maximas, hessian_minimas = hessian_matrix_eigvals(hessian_matrix_image)
    if hessian_component == "minima":
        return hessian_minimas
    elif hessian_component == "maxima":
        return hessian_maximas
    else:
        raise ValueError(f"Invalid hessian_component value: {hessian_component}. Must be 'minima' or 'maxima'.")

def preprocess_image(
    image: npt.NDArray[np.float64],
    model_image_size: tuple[int, int],
    norm_upper_bound: float,
    norm_lower_bound: float,
    apply_hessian: bool,
    hessian_component: str,
    hessian_sigma: int,
) -> npt.NDArray[np.float64]:
    """Preprocess the image"""
    # Normalise the image
    image = normalise_image(image, norm_upper_bound, norm_lower_bound)

    # Optionally apply hessian filter
    if apply_hessian:
        image = apply_hessian_filter(image, hessian_component=hessian_component, sigma=hessian_sigma)

    # Resize the image
    image = resize_image(image, size=model_image_size)

    return image

def preprocess_mask(
    mask: npt.NDArray[np.bool_],
    model_image_size: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """Preprocess a mask"""
    mask = resize_mask(mask, model_image_size)
    return mask
