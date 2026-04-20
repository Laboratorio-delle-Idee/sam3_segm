import torch
import json
import os
from PIL import Image
import time
import argparse

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ================= CONFIG =================
SCORE_THRESHOLD = 0.5

# ================= ARGPARSE =================
parser = argparse.ArgumentParser(description="Object detection with SAM3")
parser.add_argument(
    "--input_dir",
    type=str,
    help="Path to folder containing images - the output JSON files will be saved in the same folder",
    required=True   
)

parser.add_argument(
    "--classes_file",
    type=str,
    default="classes.txt",
    help="Path to txt file containing classes (one per line)"
)

args = parser.parse_args()

input_dir = args.input_dir


if args.classes_file:
    with open(args.classes_file) as f:
        list_classes = [line.strip() for line in f if line.strip()]

# ================= MODEL =================
print('Loading model...')
model = build_sam3_image_model()
processor = Sam3Processor(model)
print('Model ready.')

# ================= INFERENCE =================
print('Start batch inference...')

with torch.no_grad():
    for filename in os.listdir(input_dir):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(input_dir, filename)
        print(f"\nProcessing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        all_detections = []

        t0 = time.perf_counter()
        base_state = processor.set_image(image)

        for obj in list_classes:
            print(f"  Detecting: {obj}")

            # copia stato della backbone 
            state = {
                "backbone_out": dict(base_state["backbone_out"]),
                "original_height": base_state["original_height"],
                "original_width": base_state["original_width"],
            }

            out = processor.set_text_prompt(state=state, prompt=obj)

            boxes = out["boxes"]
            scores = out["scores"]

            for i in range(len(scores)):
                score = scores[i].item()

                if score < SCORE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = boxes[i].cpu().numpy()

                all_detections.append({
                    "label": obj,
                    "points": [
                        [float(x1), float(y1)],
                        [float(x2), float(y2)]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "mask": None
                })
                
        # End time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Image time: {(t1 - t0)*1000:.2f} ms")

        # ================= LABELME JSON =================
        labelme_json = {
            "version": "5.11.4",
            "flags": {},
            "shapes": all_detections,
            "imagePath": filename,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }

        json_path = os.path.join(input_dir, filename.rsplit(".", 1)[0] + ".json")

        with open(json_path, "w") as f:
            json.dump(labelme_json, f, indent=2)

        print(f"Saved: {json_path} ({len(all_detections)} classes)")

print("\nDone.")