import random as rnd

import cv2 as cv
import numpy as np
from tqdm import tqdm
from collections import deque

import disk

EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "gif"]


def load(path: str, grayscale: bool = False) -> np.ndarray:
    """Load image from file.

    :param path: input image file
    :param grayscale: load as grayscale

    :return: loaded image
    """
    img = cv.imread(path, cv.IMREAD_COLOR if not grayscale else cv.IMREAD_GRAYSCALE)
    if img is None:
        raise OSError(f"Unable to load image from {path}")
    return img


def save(img: np.ndarray, path: str, fmt: str = "png") -> bool:
    """Save image to file.

    :param img: input image
    :param path: output image file
    :param fmt: image format ('jpg' or 'png')

    :return: True if successful, False otherwise
    """
    if fmt == "jpg":
        path = disk.set_ext(path, "jpg")
    elif fmt == "png":
        path = disk.set_ext(path, "png")
    else:
        raise f"Image format must be 'jpg' or 'png'"
    return cv.imwrite(path, img)


def normalize(img: np.ndarray, mode: str) -> np.ndarray:
    """Normalize image to range [0, 1].

    :param img: input image
    :param mode: normalization mode ('unit', 'sym', 'uint8')

    :return: normalized image
    """
    if mode == "unit":
        img = img / 255.0
    elif mode == "sym":
        img = img / 127.5 - 1
    elif mode == "uint8":
        return cv.normalize(
            img, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
    return img.astype(np.float32)


def denormalize(img: np.ndarray, mode: str) -> np.ndarray:
    """Denormalize image to range [0, 255].

    :param img: input image
    :param mode: original normalization mode ('unit' or 'sym')

    :return: denormalized image
    """
    if mode == "unit":
        img = img * 255.0
    elif mode == "sym":
        img = (img + 1) * 127.5
    else:
        raise ValueError(f"Unknown denormalization mode: {mode}")
    return img.astype(np.uint8)


def crop(img: np.ndarray, roi: list[int]) -> np.ndarray:
    """Crop image to region of interest.

    :param img: input image
    :param roi: region of interest [top, bottom, left, right]

    :return: cropped image
    """
    top, bottom, left, right = roi
    return (
        img[top:bottom, left:right, :]
        if len(img.shape) == 3
        else img[top:bottom, left:right]
    )


def perspective(img: np.ndarray, poly: list[int], size: int) -> np.ndarray:
    """Crop image to region of interest.

    :param img: input image
    :param poly: region of interest [x1, y1, x2, y2, x3, y3, x4, y4]
    :param size: output size (square image)

    :return: perspective transformed image
    """
    pts1 = np.array(poly, np.float32).reshape(4, 2)
    square = [[0, 0], [size, 0], [size, size], [0, size]]
    pts2 = np.array(square, np.float32)
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    warped = cv.warpPerspective(img, matrix, dsize=(size, size))
    return warped


def resize_sqr(img: np.ndarray, size: int, method: str = "linear") -> np.ndarray:
    """Resize image to specified (size x size) square.

    :param img: input image
    :param size: output square size
    :param method: interpolation method

    :return: resized image
    """
    return resize(img, size=(size, size), method=method)


def resize(
    img: np.ndarray, size: tuple[int, int], method: str = "linear"
) -> np.ndarray:
    """Resize image to specified size.

    :param img: input image
    :param size: output size (height, width)
    :param method: interpolation method

    :return: resized image
    """
    if img.shape == size:
        return img
    try:
        interpolation = {
            "nearest": cv.INTER_NEAREST,
            "linear": cv.INTER_LINEAR,
            "cubic": cv.INTER_CUBIC,
            "lanczos": cv.INTER_LANCZOS4,
            "area": cv.INTER_AREA,
        }[method]
    except KeyError:
        raise ValueError(f"Unknown interpolation method: {method}")
    return cv.resize(img, dsize=size, dst=None, interpolation=interpolation)


def subsample(
    img: np.ndarray, original: tuple[int, int], resolution: float
) -> np.ndarray:
    """Subsample image to specified grid size using provided resolution.

    :param img: input image
    :param original: original image size in meters (width, height)
    :param resolution: required resolution in meters per pixel

    :return: subsampled image
    """
    grid_size = tuple(int(dim / resolution) for dim in original)
    return resize(img, size=grid_size, method="nearest")


def random_mask(size: int, blobs: int, hard: bool) -> np.ndarray:
    """Create random mask image.

    :param size: output square size
    :param blobs: maximum number of random blobs
    :param hard: use hard threshold

    :return: random mask image
    """
    nsize = rnd.randint(a=2, b=max(2, blobs))
    noise = np.random.randint(low=0, high=2, size=(nsize, nsize)).astype(np.float32)
    noise = resize_sqr(noise, size, method="cubic")
    mask = cv.normalize(noise, dst=None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    if hard:
        mask = cv.threshold(noise, thresh=0.5, maxval=1, type=cv.THRESH_BINARY)[1]
    return mask


def composite(img0: np.ndarray, img1: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Composite two images using binary mask.

    :param img0: first image
    :param img1: second image
    :param mask: binary mask

    :return: composite image
    """
    mask3 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    comp1 = np.multiply(img0, 1 - mask3)
    comp2 = np.multiply(img1, mask3)
    comp = (comp1 + comp2).astype(np.uint8)
    return comp


def post_process(img: np.ndarray, radius: int) -> np.ndarray:
    """Apply postprocessing to image.

    :param img: input image
    :param radius: postprocessing radius

    :return: postprocessed image
    """
    if img.dtype != np.uint8:
        source = (img * 255).astype(np.uint8)
    else:
        source = img
    ksize = 2 * radius + 1
    processed = cv.medianBlur(source, ksize=ksize)
    if img.dtype != np.uint8:
        processed = processed.astype(np.float32) / 255
    return processed


def size_from_disk(filename: str) -> tuple[int, int]:
    """Return the size in pixels of an image file.

    :param filename: input image file

    :return: image size (height, width)
    """
    img = load(filename)
    if img is None:
        raise OSError(f"Unable to get size of image file: {filename}")
    return img.shape[0], img.shape[1]


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute the mean squared error between two images.

    :param img1: first image
    :param img2: second image

    :return: mean squared error
    """
    if img1.shape != img2.shape:
        raise ValueError("Image sizes do not match")
    return float(np.mean(np.square(img1 - img2)))


def intensity(img: np.ndarray, unit: bool = False) -> float:
    """Compute the average intensity of an image.

    :param img: input image
    :param unit: normalize to [0, 1]

    :return: average intensity
    """
    factor = 1 / 255 if unit else 1
    mean = np.mean(img * factor)
    return float(mean)


def contrast(img: np.ndarray, unit: bool = False) -> float:
    """Compute the contrast of an image (standard deviation).

    :param img: input image
    :param unit: normalize to [0, 1]

    :return: global contrast
    """
    factor = 1 / 255 if unit else 1
    std = np.std(img * factor)
    return float(std)


def overlay(
    img: np.ndarray, prob: np.ndarray, alpha: float, red: int = 1
) -> np.ndarray:
    """Overlay a mask on top of an image.

    :param img: input image
    :param prob: input mask
    :param alpha: overlay opacity
    :param red: red class (0 or 1)

    :return: overlay image
    """
    img_float = img.copy().astype(np.float32)
    prob3 = np.zeros_like(img_float)
    if red == 1:
        c1 = 1
        c2 = 2
    elif red == 0:
        c1 = 2
        c2 = 1
    else:
        raise ValueError("Red class must be 0 or 1")

    prob3[:, :, c1] = (1 - prob) * 255
    prob3[:, :, c2] = prob * 255
    over = cv.addWeighted(
        src1=img_float, alpha=1 - alpha, src2=prob3, beta=alpha, gamma=0
    ).astype(np.uint8)

    return over


def draw_poly(
    img: np.ndarray, poly: list[int], color: tuple[int, int, int]
) -> np.ndarray:
    """Draw polygon on image.

    :param img: input image
    :param poly: polygon coordinates
    :param color: polygon color

    :return: image with polygon
    """
    pts = np.array(poly, np.int32).reshape(4, 2)
    img = cv.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    return img


def draw_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    size: float,
    color: tuple[int, int, int],
    bold: bool = False,
) -> None:
    """Draw text on image.

    :param img: input image
    :param text: text string
    :param pos: text position
    :param size: text size
    :param color: text color
    :param bold: bold text

    :return: image with text
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, text, pos, font, size, color, 2 if bold else 1, cv.LINE_AA)


def draw_lines(
    img: np.ndarray, lines: np.ndarray, color: tuple[int, int, int], thickness: int = 1
) -> None:
    """Draw line on image.

    :param img: input image
    :param lines: Nx4 array with line coordinates
    :param color: line color
    :param thickness: line thickness

    :return: image with line
    """
    for line in lines:
        start = (line[0], line[1])
        end = (line[2], line[3])
        img = cv.line(img, start, end, color, thickness, cv.LINE_AA)


def point_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute Euclidean distance between two points.

    :param p1: first point as a numpy array
    :param p2: second point as a numpy array

    :return: distance between points
    """
    return float(np.linalg.norm(p1 - p2))


def temporal_filter(data: np.ndarray, length: int, mode: str) -> np.ndarray:
    """Apply temporal filter to image sequence.

    :param data: input image sequence
    :param length: filter length
    :param mode: filter mode (median or mean)

    :return: filtered image
    """
    filtered = data.copy()
    for i in tqdm(range(length, data.shape[0]), desc="Temporal filtering"):
        if mode == "median":
            filtered[i] = np.median(data[i - length : i], axis=0)
        elif mode == "mean":
            filtered[i] = np.mean(data[i - length : i], axis=0)
        else:
            raise ValueError(f"Unknown temporal filter mode: {mode}")
    return filtered


class RollingAvgStd:
    """
    Compute rolling or infinite mean and standard deviation of images.
    - If window_size > 0: keeps statistics for last N frames.
    - If window_size == 0: keeps cumulative statistics for all frames (infinite accumulation).
    """

    def __init__(self, window_size: int = 0):
        """
        :param window_size:
            0 → accumulate over all frames (infinite window)
            N → rolling window size (only last N frames)
        """
        if window_size < 0:
            raise ValueError("window_size must be >= 0")

        self.window_size = window_size
        self.buffer = deque(maxlen=window_size) if window_size > 0 else None

        self.mean = None
        self.variance = None
        self.shape = None
        self.count = 0

    def _initialize(self, shape: tuple[int, int]):
        """Allocate arrays once the first image shape is known."""
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.variance = np.zeros_like(self.mean)

    def _update_welford(self, image: np.ndarray):
        """Perform one Welford update step."""
        self.count += 1
        delta = image - self.mean
        self.mean += delta / self.count
        delta2 = image - self.mean
        self.variance += delta * delta2

    def _recompute_from_buffer(self):
        """Recalculate mean and variance from the current buffer (for finite window)."""
        if self.count == 0:
            self.mean.fill(0)
            self.variance.fill(0)
            return

        stack = np.stack(self.buffer, axis=0).astype(np.float32)
        self.mean = np.mean(stack, axis=0)
        self.variance = (
            np.var(stack, axis=0, ddof=1)
            if self.count > 1
            else np.zeros_like(self.mean)
        )

    def update(self, image: np.ndarray) -> None:
        """Update rolling or cumulative mean and variance with a new image."""
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError("Image must be a valid numpy array")

        # Convert to grayscale if needed
        if image.ndim == 3 and image.shape[2] == 3:
            gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray_image = image
        else:
            raise ValueError(
                "Unsupported image shape: expected grayscale or BGR color image"
            )

        gray_image = gray_image.astype(np.float32)

        # Lazy initialization
        if self.shape is None:
            self._initialize(gray_image.shape)

        # Shape check
        if gray_image.shape != self.shape:
            raise ValueError(
                f"Image shape {gray_image.shape} does not match initialized shape {self.shape}"
            )

        # === Infinite accumulation mode ===
        if self.window_size == 0:
            self._update_welford(gray_image)
            return

        # === Rolling window mode ===
        self.buffer.append(gray_image)
        self.count = len(self.buffer)

        if len(self.buffer) == self.window_size:
            # Recompute when sliding window full
            self._recompute_from_buffer()
        else:
            # Incremental update while filling
            self._update_welford(gray_image)

    def reset(self) -> None:
        """Reset all accumulated statistics and empty the buffer."""
        if self.buffer is not None:
            self.buffer.clear()
        self.mean = None
        self.variance = None
        self.shape = None
        self.count = 0

    def deviation(self) -> np.ndarray:
        """Return the rolling or cumulative standard deviation image."""
        if self.count < 2 or self.mean is None:
            return np.zeros(self.shape or (1, 1), dtype=np.float32)
        return np.sqrt(self.variance / (self.count - 1))

    def average(self) -> np.ndarray:
        """Return the current rolling or cumulative mean image."""
        if self.count == 0 or self.mean is None:
            return np.zeros(self.shape or (1, 1), dtype=np.float32)
        return self.mean.copy()
