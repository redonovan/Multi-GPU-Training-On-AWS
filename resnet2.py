# ResNet - multi-GPU implementation.
# Sep-Oct 2023 (v2).

# Launch an AWS EC2 Ubuntu Deep Learning AMI GPU PyTorch g4dn.12xlarge instance (this has 4 GPUs).
# In a local machine window ssh into the EC2 instance with:
# ssh -i "YourPrivateKeys.pem" -L 7000:localhost:6006 ubuntu@EC2Host
# Then on the EC2 instance run:
# source activate pytorch
# pip install tensorboard
# tensorboard --logdir=runs
# On your local machine:
# Point browser at http://localhost:7000/ to see TensorBoard output
# In another local machine window run:
# scp -i "YourPrivateKeys.pem" resnet2.py ubuntu@EC2Host:/home/ubuntu/
# ssh -i "YourPrivateKeys.pem" ubuntu@EC2Host
# Then on the EC2 instance run:
# source activate pytorch
# python resnet2.py
# This will download the data and then train all the networks.
#
# To colour corresponding networks the same colour in TensorBoard, click on the colour palette
# icon, top left, under Time Series, near Run, and enter (Error_20|Error_32|Error_44|Error_56)
# Then in the Settings column click 'Link by Step' and click at the end of your data for a legend.


import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class Same(nn.Module):
    '''Two layer conv net with optional shortcut connection'''
    def __init__(self, resnet, channels):
        super().__init__()
        self.resnet = resnet
        self.c1     = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding='same')
        self.bn1    = nn.BatchNorm2d(channels)
        self.c2     = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding='same')
        self.bn2    = nn.BatchNorm2d(channels)
        self.relu   = nn.ReLU()
    def forward(self, i):
        x = self.c1(i)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        if self.resnet:
            x = x + i
        x = self.relu(x)
        return x


class Down(nn.Module):
    '''Similar to Same, but doubles the channels and halves the feature map size'''
    def __init__(self, resnet, in_channels):
        super().__init__()
        self.resnet      = resnet
        self.in_channels = in_channels
        self.c1          = nn.Conv2d(in_channels,  in_channels*2,kernel_size=3,stride=2,padding=1)
        self.bn1         = nn.BatchNorm2d(in_channels*2)
        self.c2          = nn.Conv2d(in_channels*2,in_channels*2,kernel_size=3,stride=1,padding='same')
        self.bn2         = nn.BatchNorm2d(in_channels*2)
        self.relu        = nn.ReLU()
    def forward(self, i):
        # i shape                     (batch, in_channels,   feat_map_size,   feat_map_size)
        x = self.c1(i)
        # x shape                     (batch, in_channels*2, feat_map_size/2, feat_map_size/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        if self.resnet:
            x[:, :self.in_channels, :, :] += i[:, :, ::2, ::2]
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    '''CIFAR-10 plain or residual network following Section 4.2 of the ResNet paper'''
    def __init__(self, resnet: bool, n: int):
        super().__init__()
        self.resnet = resnet          # resnet or plain net
        self.n      = n               # number of blocks per out_channel size
        self.md     = nn.ModuleDict() # ordered dict of convolutional layers / blocks
        #
        # the first layer converts 3-channel data into 16-channel data
        self.md.update({'conv1_0' : nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding='same')})
        self.md.update({'relu1_0' : nn.ReLU()})
        #
        # there are then 3n convolution blocks, n with each of {16,32,64} filters:
        #
        # the conv2_i blocks use 16-channels throughout
        for i in range(n):
            self.md.update({f'conv2_{i}' : Same(resnet, channels=16)})
        #
        # the conv3_0 block increases channels from 16 to 32 and downsamples the feature map by 2
        self.md.update({'conv3_0' : Down(resnet, in_channels=16)})
        # subsequent conv3_i block use 32-channels throughout
        for i in range(1,n):
            self.md.update({f'conv3_{i}' : Same(resnet, channels=32)})
        #
        # the conv4_0 block increases channels from 32 to 64 and downsamples the feature map by 2
        self.md.update({'conv4_0' : Down(resnet, in_channels=32)})
        # subsequent conv4_i blocks use 64-channels throughout
        for i in range(1,n):
            self.md.update({f'conv4_{i}' : Same(resnet, channels=64)})
        #
        # the global average pooling layer requires the dynamnic feature map size
        # the network ends with a fully connected layer converting pooled 64-channel data into 10-logits
        self.fc = nn.Linear(64, 10)
        #
    def forward(self, x):
        # the input data x must be 3-channel data because that is hardcoded in the module dict above
        # any feature map size can be used; the following comments assume 32x32 inputs
        # shapes are                             (batch, ch, hi, wi)
        # input x is assumed to be               (batch,  3, 32, 32)
        # 
        # apply the convolutional layers / blocks
        for l in self.md.values():
            x = l(x)
        # output x is expected to be             (batch, 64,  8,  8)
        #
        # the global average pooling layer needs to know the dynamic shape of x
        x = nn.AvgPool2d(x.shape[-2],x.shape[-1])(x)
        # post pooling shape                     (batch, 64,  1,  1)
        # squeeze last two dims
        x = x[:,:,0,0]
        # post squeeze shape                     (batch, 64)
        # fully connected layer
        x = self.fc(x)
        # post fc shape                          (batch, 10)
        # return logits
        return x


def gather(value):
    '''
    sum value over all workers
    '''
    value_tensor = torch.tensor(value).cuda()
    dist.all_reduce(value_tensor)
    value = value_tensor.item()
    del value_tensor
    return value
    

def gather_results(correct, total, loss, len_loader):
    '''
    gather and compute results from all workers
    '''
    # sum values from all workers
    correct    = gather(correct)
    total      = gather(total)
    loss       = gather(loss)
    len_loader = gather(len_loader)
    # compute results
    loss /= len_loader
    acc = 100 * correct / total
    err = 100 - acc
    return err, loss

    
def train_model(model_type, n, writer, num_epochs, ilr, train_loader, test_loader, rank):
    '''
    train a model of the specified type and depth, writing results to TensorBoard
    '''
    # obtain repeatable results even when training each model individually
    torch.manual_seed(6)
    # instantiate a model and move it to the GPU
    model = ResNet(model_type == 'ResNet', n)
    model.cuda()
    #
    # Wrap the model in DistributedDataParallel to support multi-GPU training.
    # This broadcasts the state_dict from the process with rank 0 to all other processes in the group.
    model = DDP(model, device_ids=[rank], output_device=rank)
    #
    if rank == 0:
        layers = len([n for n, p in model.named_parameters() if "bn" not in n])//2
        print('Model type       = ', model_type)
        print('n                = ', n)
        print('Model Layers     = ', layers)
        print('Model Parameters = ', sum([p.numel() for p in model.parameters()]))
    #
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=ilr, momentum=0.9, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [num_epochs//2,3*num_epochs//4], gamma=0.1, verbose=False)
    #
    # train loop
    steps = 0
    for epoch in range(1,num_epochs+1):
        # train
        _ = model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        # ensure sampler shuffle works correctly
        train_loader.sampler.set_epoch(epoch)
        for b in train_loader:
            images, labels = b
            images = images.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            logits = model(images)
            loss   = loss_fn(logits, labels)
            train_loss += loss.item()
            # calling backward causes DDP to average gradients over all workers
            loss.backward()
            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total   += labels.shape[0]
            optimizer.step()
            steps += 1
        # gather results from all workers
        train_err, train_loss = gather_results(correct, total, train_loss, len(train_loader))
        # test
        _ = model.eval()
        with torch.no_grad():
            test_loss = 0.0
            correct = 0
            total = 0
            for b in test_loader:
                images, labels = b
                images = images.cuda()
                labels = labels.cuda()
                logits = model(images)
                loss   = loss_fn(logits, labels)
                test_loss += loss.item()
                _, preds = torch.max(logits, 1)
                correct += (preds == labels).sum().item()
                total   += labels.shape[0]
            # gather results from all workers
            test_err, test_loss = gather_results(correct, total, test_loss, len(test_loader))
        if rank == 0:
            writer.add_scalars('Train Error', {f'{layers}': train_err}, steps)
            writer.add_scalars('Test Error', {f'{layers}': test_err}, steps)
            print(f'epoch {epoch:2d} steps {steps:4d} '
                  f'Loss : train {train_loss:.3f} test {test_loss:.3f} '
                  f'Error(%) : train {train_err:.3f} test {test_err:.3f}')
        scheduler.step()
        

def train_type(model_type, nseq, num_epochs, ilr, train_loader, test_loader, rank):
    '''
    train a sequence of models of the specified type, writing graphs to TensorBoard
    '''
    # create separate TensorBoard logs for each model type
    if rank == 0:
        writer = SummaryWriter(f'runs/{model_type}')
    else:
        writer = None
    #
    for n in nseq:
        train_model(model_type, n, writer, num_epochs, ilr, train_loader, test_loader, rank)


def worker(rank, world_size, train_data, test_data):
    '''
    A worker runs on each GPU
    '''
    # sanity check
    assert(torch.cuda.is_available())
    #
    # ensure this process works on a single GPU
    torch.cuda.set_device(rank)
    # initialize torch.distributed using MASTER_PORT and MASTER_ADDR environment variables
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    #
    # obtain repeatable results
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #
    # hyperparameters
    num_epochs        = 164       # number of training epochs per model (164 = 64k steps)
    ilr               = 0.1       # initial learning rate
    nseq              = [3,5,7,9] # n determines network depth
    global_batch_size = 128       # batch size applied during each update, from all GPUs pooled
    #
    # DistributedSampler pulls rank and world_size from the current distributed group
    train_sampler = DistributedSampler(train_data, shuffle=True)
    test_sampler  = DistributedSampler(test_data,  shuffle=False)
    #
    # split the global batch size between all the GPUs
    batch_size = global_batch_size // world_size
    #
    # create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader  = DataLoader(test_data,  batch_size=batch_size, sampler=test_sampler)
    #
    # train a sequence of Plain networks and their corresponding ResNets
    train_type('Plain',  nseq, num_epochs, ilr, train_loader, test_loader, rank)
    train_type('ResNet', nseq, num_epochs, ilr, train_loader, test_loader, rank)


def main():
    '''
    main() gets executed only once, so it does things that only need to be done once.
    Namely downloading the data to disk and spawning the worker processes.
    '''
    #
    # Data
    # The training data undergoes some basic augmentation, but the test data does not.
    training_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(4,fill=128),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
    ])
    #
    train_data = datasets.CIFAR10(
        root="/home/ubuntu/pytorch_data",            # this is where it will download it to
        train=True,                                  # create dataset from training set
        download=True,                               # download, but only once
        transform=training_transforms,               # takes in a PIL image [0-255] and returns a transformed version [0-1]
    )
    #
    test_data = datasets.CIFAR10(
        root="/home/ubuntu/pytorch_data",            # this is where it will download it to
        train=False,                                 # create dataset from test set
        download=True,                               # download, but only once
        transform=transforms.ToTensor(),             # takes in a PIL image and returns a Tensor
    )
    #
    # Workers
    #
    # environment variables for dist.init_process_group()'s "env" initialization mode
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    #
    # run on all GPUs on this machine
    world_size = torch.cuda.device_count()
    #
    mp.spawn(worker,
        args=(world_size, train_data, test_data),
        nprocs=world_size,
        join=True)


if __name__ == "__main__":
    main()
