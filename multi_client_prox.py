import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, GTSRB

# TODO : Define Configuration
logging.basicConfig(filename='fedprox.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device)


def average_state_dicts(state_dicts):
    num_models = len(state_dicts)
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        if all(key in state_dict for state_dict in state_dicts):
            tensors = [state_dict[key] for state_dict in state_dicts]
            averaged_tensor = torch.stack(tensors)

            averaged_state_dict[key] = averaged_tensor.float().mean(0)

    return averaged_state_dict


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# TODO : Load Processed Data


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = CIFAR10('../.data', train=True, download=True, transform=transform_train)
test_set = CIFAR10('../.data', train=False, download=True, transform=transform_test)


# train_set = torchvision.datasets.ImageFolder(root="data/skin-disease/Lumpy Skin Images Dataset", transform=transform_train)
# test_set = torchvision.datasets.ImageFolder(root="data/skin-disease/Lumpy Skin Images Dataset", transform=transform_train)


# TODO: Building Model for Federated Learning
class FedProx(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, mu=0.1):
        defaults = dict(lr=lr, mu=mu)
        super(FedProx, self).__init__(params, defaults)
        self.start = None

    def step(self, global_model=None):
        for group in self.param_groups:
            for p, q in zip(group['params'], global_model):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['prev_params'] = p.data.clone()

                prev_params = state['prev_params']

                p.data.add_(-group['lr'], grad + group['mu'] * (p.data - q))
                state['prev_params'] = p.data.clone()


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    @staticmethod
    def validation_epoch_end(outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    @staticmethod
    def epoch_end(epoch, result):
        logging.info("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, predict = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predict == labels).item() / len(predict))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet50(ImageClassificationBase):
    def __init__(self, block=ResidualBlock, layers=None, num_classes=10):
        layers = [3, 4, 6, 3]
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class FedProx(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, mu=0.1):
        defaults = dict(lr=lr, mu=mu)
        super(FedProx, self).__init__(params, defaults)
        self.start = None

    def step(self, global_model=None):
        for group in self.param_groups:
            for p, q in zip(group['params'], global_model):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['prev_params'] = p.data.clone()

                prev_params = state['prev_params']

                p.data.add_(-group['lr'], grad + group['mu'] * (p.data - q))
                state['prev_params'] = p.data.clone()



# TODO : Build Client and Server Classes
class Client:
    def __init__(self, device, train_loader, test_loader, epochs, client_id, lr=0.001, opt_func=None):
        self.device = device
        self.model = ResNet50(num_classes=10)
        self.model.to(device)
        self.train_loader = DeviceDataLoader(train_loader, device)
        self.test_loader = DeviceDataLoader(test_loader, device)
        self.epochs = epochs
        self.id = client_id
        self.lr = lr
        self.opt_fn = opt_func

    def train_local_epochs(self):
        history = []
        optimizer = self.opt_fn(self.model.parameters(), self.lr)
        glob = optimizer.param_groups[0]['params']
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            for batch in self.train_loader:
                loss = self.model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step(glob)
                optimizer.zero_grad()
            # Validation phase
            result = evaluate(self.model, self.test_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            self.model.epoch_end(epoch, result)
            history.append(result)
        return history

    def load_global(self, weights):
        self.model.load_state_dict(weights)


class GlobalModel:
    def __init__(self, n_clients, communication_rounds, num_epochs_per_round):
        self.n_clients = n_clients
        self.testloader = []
        self.dataloaders = []
        self.communication_rounds = communication_rounds
        self.num_epochs_per_round = num_epochs_per_round
        self.prepare_data(train_set, test_set)
        self.model = ResNet50(num_classes=10)
        self.clients = [Client(
            epochs=num_epochs_per_round,
            client_id=i,
            device='cuda:2',
            train_loader=self.dataloaders,
            test_loader=self.testloader,
            opt_func=FedProx
        ) for i in range(n_clients)]

    def prepare_data(self, train_set, test_set):
        self.dataloaders = DataLoader(train_set, batch_size=64, shuffle=True)
        self.testloader = DataLoader(test_set, batch_size=64, shuffle=True)

    def run(self):
        for i in range(self.communication_rounds):
            logging.info(f"Communication Round {i + 1}/{self.communication_rounds} : ")
            weights = []
            for client in self.clients:
                logging.info(f'Client {client.id} : ')
                client.load_global(self.model.state_dict())
                client.train_local_epochs()
                copy_mod = copy.deepcopy(client.model)
                copy_mod.to('cpu')
                weights.append(copy_mod.state_dict())

            weight_avg = average_state_dicts(weights)
            self.model.load_state_dict(weight_avg)
            model_copy = copy.deepcopy(self.model)
            model_copy.to('cuda:2')
            test_load = DeviceDataLoader(self.testloader, device='cuda:2')
            logging.info(evaluate(model_copy, test_load))


print(sum(p.numel() for p in ResNet50().parameters()))

for i in range(1, 11):
    logging.info(f"With Clients Count : {i}")
    fn = GlobalModel(
        n_clients=i,
        communication_rounds=20,
        num_epochs_per_round=2,
    )

    fn.run()
