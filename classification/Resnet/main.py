import os
import yaml

import torch
import torch.nn as nn

from utils import Config, eval_accuracy
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from model import *



def train(train_dataloader, test_dataloader, device, config):
    if config.model == 'resnet18':
        model = resnet18(num_classes=config.num_classes)
    elif config.model == 'resnet34':
        model = resnet34(num_classes=config.num_classes)
    elif config.model == 'resnet50':
        model = resnet50(num_classes=config.num_classes)
    elif config.model == 'resnet101':
        model = resnet101(num_classes=config.num_classes)
    elif config.model == 'resnet152':
        model = resnet152(num_classes=config.num_classes)
    else:
        print("Not implemented error")
        return 

    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # loss for the network
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) # optimizer to perform gradient descent


    for epoch in range(config.num_epochs):
        loss_sum = 0.
        model.train()

        for j, (x, y) in enumerate(train_dataloader):

            x = x.to(device)
            y = y.to(device)

            output = model(x)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval() # model validation
            loss_sum += loss.item()

        test_acc = eval_accuracy(model, test_dataloader, device)

        print("Epoch [{:2d}/{}], Loss : {:.3f},Test Accuracy : {:.3f}".format(epoch+1,
                config.num_epochs, loss_sum/len(train_dataloader), test_acc))


    model_name = '{}.pth'.format(config.model)
    torch.save(model.state_dict(),join(config.CHK_DIR,model_name))

if __name__ == '__main__':
    config = Config("config.yaml") # load config

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU) # choose which gpu to use

    device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")

    # Image normalization
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataloader = DataLoader(datasets.CIFAR10(root='./data', train=True,
                                                   download=True, transform=transform), batch_size=config.batch_size,
                                                   shuffle=True, num_workers=config.num_workers)

    test_dataloader = DataLoader(datasets.CIFAR10(root='./data', train=False,
                                                  download=True, transform=transform), batch_size=config.batch_size,
                                                  shuffle=False, num_workers=config.num_workers)

    train(train_dataloader, test_dataloader, device, config)

