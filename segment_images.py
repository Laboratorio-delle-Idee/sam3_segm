import argparse as ap
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

# -------------------------------------------------------------------------
# MINIMAL EMBEDDED UTILITIES (replacing disk.py and image.py)
# -------------------------------------------------------------------------

EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "tif", "tiff", "gif"]


def get_relpath(path: str) -> str:
    return os.path.relpath(path)


def get_basename(filename: str) -> str:
    return os.path.basename(filename)


def join_paths(a: str, b: str) -> str:
    return os.path.join(a, b)


def init_folder(folder: str):
    """Create folder if missing, otherwise clean it."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # Clean existing folder
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if os.path.isfile(p):
                os.remove(p)


def get_files(folder: str, extensions=None):
    """Return sorted list of image files."""
    if extensions is None:
        extensions = EXTENSIONS

    exts = set([e.lower() for e in extensions])
    files = []

    with os.scandir(folder) as it:
        for entry in it:
            if entry.is_file():
                ext = entry.name.lower().split(".")[-1]
                if ext in exts:
                    files.append(os.path.abspath(entry.path))

    return sorted(files)


def pil_save(img_array: np.ndarray, path: str):
    """
    Save a NumPy array as image using PIL.
    Handles both grayscale (H,W) and RGB (H,W,3).
    """
    img = Image.fromarray(img_array.astype(np.uint8))
    img.save(path)


def pil_draw_text(img_array: np.ndarray, text: str, pos, size=0.6, color=(255, 255, 0)):
    """
    Draw text using PIL (replacement for cv2.putText).
    img_array is (H,W,3) RGB NumPy.
    """
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)

    # PIL text size is not directly comparable to cv2 fonts — scale manually
    font_size = int(20 * size)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw.text(pos, text, fill=color, font=font)
    return np.array(img)


# -------------------------------------------------------------------------
# ARGUMENT PARSER
# -------------------------------------------------------------------------


def parse_args() -> ap.Namespace:
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Segment input images using SAM3 model.",
    )
    parser.add_argument("input", help="input image folder")
    parser.add_argument("output", help="output mask folder")
    parser.add_argument("--prompt", default="grass", help="segmentation text prompt")
    parser.add_argument(
        "--chkpt",
        default="assets/sam3.pt",
        help="model checkpoint path",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[640, 480],
        help="inference image resolution (width height)",
    )
    parser.add_argument(
        "--binthresh", default=0.5, type=float, help="segmentation threshold"
    )
    parser.add_argument("--debug", default=None, help="output debug folder")
    parser.add_argument("--alpha", default=0.2, type=float, help="mask overlay alpha")
    return parser.parse_args()


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()

    print("Initializing SAM3 model...")
    try:
        model = build_sam3_image_model(load_from_HF=False, checkpoint_path=args.chkpt)
    except FileNotFoundError:
        sys.exit(f"Model checkpoint not found: '{args.chkpt}'")
    processor = Sam3Processor(model)

    print(f"Input folder: '{get_relpath(args.input)}'...", end=" ")
    files = get_files(args.input)
    total = len(files)
    if total == 0:
        sys.exit("No images found. Exiting.")
    print(f"{total} images found")

    print(f"Inference size: {args.resize[0]}x{args.resize[1]}px")

    init_folder(args.output)
    print(f"Output folder: '{get_relpath(args.output)}'")

    if args.debug is not None:
        init_folder(args.debug)
        print(f"Debug folder: '{get_relpath(args.debug)}'")

    for f in tqdm(files, desc="Segmenting images"):
        # Load image from disk using PIL
        img = Image.open(f).convert("RGB")

        # Resize for inference (width, height)
        img_resized = img.resize((args.resize[0], args.resize[1]))

        # Prepare the image for inference
        inference = processor.set_image(img_resized)

        # Prompt the model with text
        output = processor.set_text_prompt(state=inference, prompt=args.prompt)

        # Extract masks & scores
        masks = output["masks"].cpu().detach().numpy()
        scores = output["scores"].cpu().detach().numpy()

        # Keep only masks above threshold
        masks = masks[scores >= args.binthresh]

        # Combine masks → binary {0,255}
        segmentation = (np.sum(masks, axis=0) > 0).astype(np.uint8) * 255
        segmentation = np.squeeze(segmentation)  # ensure (H, W) for downstream ops

        # Save segmentation mask (grayscale)
        out_path = join_paths(args.output, get_basename(f))
        pil_save(segmentation, out_path)

        # Debug overlay
        if args.debug is not None:
            img_np = np.array(img_resized, dtype=np.uint8)

            mask = (segmentation > 0).astype(np.uint8)

            # Green mask image
            green = np.zeros_like(img_np)
            green[:, :, 1] = 255  # pure green

            # Alpha blend: img * (1-alpha) + green * alpha * mask
            overlay = (
                img_np * (1 - args.alpha) + green * args.alpha * mask[:, :, None]
            ).astype(np.uint8)

            # Add confidence text
            max_score = (
                float(np.max(scores[scores >= args.binthresh]))
                if np.any(scores >= args.binthresh)
                else 0.0
            )

            if max_score > 0:
                text = f"Conf: {max_score:.2f}"
                text_x = overlay.shape[1] - 120
                text_y = overlay.shape[0] - 25

                overlay = pil_draw_text(
                    overlay, text, (text_x, text_y), size=0.8, color=(255, 255, 0)
                )

            debug_path = join_paths(args.debug, get_basename(f))
            pil_save(overlay, debug_path)

    print("Done.")
