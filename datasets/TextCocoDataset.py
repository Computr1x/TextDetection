import json
import os
import numpy as np
import numpy.lib.recfunctions as nlr
import torch
from distinctipy import distinctipy
from torch.utils.data import  Dataset
from PIL import Image, ImageDraw


def generate_mask(image_size, segments):
    mask = Image.new('RGB', image_size, color=(0,0,0))
    segments_len = len(segments)
    colors = distinctipy.get_colors(segments_len)
    for segmentIndex in range(0, segments_len):
        polygon_flat = segments[segmentIndex]
        polygon = []
        for i in range(0, len(polygon_flat), 2):
            polygon.append((polygon_flat[i], polygon_flat[i + 1]))

        color = colors[segmentIndex]
        rgb_color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
        ImageDraw.Draw(mask).polygon(polygon, outline=rgb_color, fill=rgb_color)
    return mask


class TextCocoDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        with open(os.path.join(self.root, "annotation.json")) as f:
            self.annotation = json.load(f)

    def __getitem__(self, idx):
        image_info = self.annotation["images"][idx]
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", image_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = generate_mask((image_info["width"], image_info["height"]),
                             self.annotation["annotations"][idx]["segmentation"])
        # convert the PIL Image into a numpy array
        mask = nlr.unstructured_to_structured(np.array(mask).reshape(image_info["height"], -1, 3)).astype('O')
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                  "iscrowd": iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotation["images"])

