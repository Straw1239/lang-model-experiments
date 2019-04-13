import ConvLM
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import codecs
from pathlib import PureWindowsPath

def train(model, dataset, iterations, block_size):

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0

    for i in range(iterations):

        r = rand.randint(0,len(dataset)-block_size-1)
        inputs = dataset[r : r+block_size].unsqueeze(0).cuda()
        labels = dataset[r+1 : r+block_size+1].unsqueeze(0).cuda()

        opt.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print(running_loss)
            running_loss = 0.0

f = open(PureWindowsPath('c://Users/Edwin/Documents/enwik8/enwik8.txt'),mode="r",encoding="utf-8")
if f.mode == 'r':

    inputlist = []
    while True:
        c = f.read(1)
        if not c:
            break
        inputlist.append(ord(c))
    inputtensor=torch.tensor(inputlist)

    train(ConvLM.ConvLM(), inputtensor, 10000, 2000)

