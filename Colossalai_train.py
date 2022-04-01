import os
from pathlib import Path
import math
import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CosineAnnealingLR
from colossalai.nn.metric import Accuracy
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import argparse
import numpy as np


###################### Real training configs ###################################
config = {'BATCH_SIZE':128, 'NUM_EPOCHS':30}
################################################################################

parser = argparse.ArgumentParser(description='train with certain optimizer and initial learning rate')
parser.add_argument('--opt', type=str, help='ConstantLR|MultiStepLR')
parser.add_argument('--lr', type=float, help='Initial LR')
args = parser.parse_args()
assert args.opt in ['ConstantLR', 'MultiStepLR']
assert args.lr > 0
config.update({'OPT': args.opt, 'LR': args.lr})
print('TRAINING CONFIG:', config)


class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits



colossalai.launch(config=config,rank=0,world_size=1,host='127.0.0.1',port=1234)

logger = get_dist_logger()

# build 

model = LeNet5(n_classes=10)

# build dataloaders
train_dataset = MNIST(
    root=Path('./tmp/'),
    download=True,
    transform = transforms.Compose([transforms.Resize((32, 32)),
                              transforms.ToTensor()])
)

test_dataset = MNIST(
    root=Path('./tmp/'),
    train=False,
    transform = transforms.Compose([transforms.Resize((32, 32)),
                              transforms.ToTensor()])
)

train_dataloader = get_dataloader(dataset=train_dataset,
                                  shuffle=True,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

test_dataloader = get_dataloader(dataset=test_dataset,
                                  add_sampler=False,
                                  batch_size=gpc.config.BATCH_SIZE,
                                  num_workers=1,
                                  pin_memory=True,
                                  )

# build criterion
criterion = torch.nn.CrossEntropyLoss()


###################### Choose one optimizer ####################################
optimizer = torch.optim.SGD(model.parameters(), lr=gpc.config.LR, momentum=0.9, weight_decay=5e-4)
################################################################################


###################### Choose two schedulers ####################################
if gpc.config.OPT == 'ConstantLR':
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
elif gpc.config.OPT == 'MultiStepLR':
    milestones = [10, 20]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones)
################################################################################


engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model,
                                                                      optimizer,
                                                                      criterion,
                                                                      train_dataloader,
                                                                      test_dataloader,
                                                                      )
# build a timer to measure time
timer = MultiTimer()

# create a trainer object
trainer = Trainer(
    engine=engine,
    timer=timer,
    logger=logger
)

# define the hooks to attach to the trainer
opt_name = gpc.config.OPT
lr_name = np.format_float_scientific(gpc.config.LR, precision=3, exp_digits=3)
hook_list = [
    hooks.LossHook(),
    hooks.LRSchedulerHook(lr_scheduler=lr_scheduler, by_epoch=False),
    # hooks.AccuracyHook(accuracy_func=Accuracy()),
    hooks.LogMetricByEpochHook(logger),
    hooks.LogMemoryByEpochHook(logger),
    hooks.LogTimingByEpochHook(timer, logger),

    # you can uncomment these lines if you wish to use them
    hooks.TensorboardHook(log_dir=f'./tb_logs_{opt_name}_{lr_name}', ranks=[0]),
    # hooks.SaveCheckpointHook(checkpoint_dir='./ckpt')
]

# start training
trainer.fit(
    train_dataloader=train_dataloader,
    epochs=gpc.config.NUM_EPOCHS,
    test_dataloader=test_dataloader,
    test_interval=1,
    hooks=hook_list,
    display_progress=False
)
