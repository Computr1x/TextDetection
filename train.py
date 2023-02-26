from vision.utils import *
from vision.engine import train_one_epoch, evaluate
import vision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datasets.TextCocoDataset import TextCocoDataset


def get_transform(train):
    transforms = [T.PILToTensor(), T.ConvertImageDtype(torch.float)]
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def train(path):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and text
    num_classes = 2
    # use our dataset and defined transformations
    dataset = TextCocoDataset('./data/TextDataset', get_transform(train=True))
    dataset_test = TextCocoDataset('./data/TextDataset', get_transform(train=False))

    # split the dataset in train and test set
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    dataset = torch.utils.data.Subset(dataset, range(train_size))
    dataset_test = torch.utils.data.Subset(dataset_test, range(train_size, train_size + test_size))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=collate_fn)

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    models_path = "./data/Models/"
    path = models_path + path

    # if models exist - continue training
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    # move model to the right device
    model.to(device)
    # ensure module in train mode
    model.train()

    # let's train
    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        # save model
        torch.save({
            'epoch': epoch,
            'model_state_dict':  model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("Train complete!")


if __name__ == '__main__':
    train("model.pt")

