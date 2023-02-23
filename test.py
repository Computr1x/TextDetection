import numpy as np

from vision.utils import *
from train import get_instance_segmentation_model
from torchvision import transforms
from PIL import Image


def main():
    root = "./data/TextDataset/Other"
    file = "0.png"
    input_image = Image.open(os.path.join(root, file)).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 512)),
        transforms.ToTensor(),
    ])
    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    model = get_instance_segmentation_model(2)

    # move the input and model to GPU for speed if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        input_batch = input_batch.to(device)
        model.to(device)

    model.load_state_dict(torch.load("./data/Models/model.pt"))
    model.eval()

    with torch.no_grad():
        prediction = model(input_batch)

    mask_and_scores = list(filter(lambda tup: tup[1] >= 0.9, zip(prediction[0]['masks'], prediction[0]['scores'])))
    result_mask = mask_and_scores[0][0]
    for (mask, score) in mask_and_scores[1:]:
        result_mask += mask
    Image.fromarray(np.clip(result_mask, 0, 1).mul(255).permute(1, 2, 0).byte().numpy().squeeze()).show()

if __name__ == '__main__':
    main()