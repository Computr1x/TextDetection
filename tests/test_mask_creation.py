import json
import os
from distinctipy import distinctipy
from PIL import Image, ImageDraw


def generate_mask(image_size, segments):
    mask = Image.new('RGBA', image_size, color=(0,0,0,255))
    segments_len = len(segments)
    colors = distinctipy.get_colors(segments_len)
    for segmentIndex in range(0, segments_len):
        polygon_flat = segments[segmentIndex]
        polygon = []
        for i in range(0, len(polygon_flat), 2):
            polygon.append((polygon_flat[i], polygon_flat[i+1]))

        color = colors[segmentIndex]
        rgb_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255)
        ImageDraw.Draw(mask).polygon(polygon, outline=rgb_color, fill=rgb_color)
    return mask

imageIndex = 0
root = './data/TextDataset'
img = Image.open(os.path.join(root, f"PNGImages/{imageIndex}.png")).convert("RGB")

with open(os.path.join(root, "annotation.json")) as f:
    annotation = json.load(f)

mask = generate_mask((512, 256), annotation["annotations"][imageIndex]["segmentation"])
mask.show()

