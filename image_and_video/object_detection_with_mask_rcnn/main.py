import torch

from dataset import PennFudanDataset
from utils import get_transform
from helper.engine import train_one_epoch, evaluate
from model import get_model_instance_segmentation
import helper.utils



def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 2

    dataset = PennFudanDataset('PennFudanPed/PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed/PennFudanPed', get_transform(train=False))

    # random permutation
    indices = torch.randperm(len(dataset)).tolist()

    # split dataset
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=helper.utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=helper.utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)


    model.to(device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 10

    for epoch in range(num_epochs):

        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        lr_scheduler.step()

        evaluate(model, data_loader_test, device=device)

    print("That's it!")

main()