import torch

class RingBuffer():
    def __init__(self, shape, maxEntries):
        self.data = torch.zeros([maxEntries] + shape)
        self.maxEntries = maxEntries
        self.index = 0
        self.count = 0

    def get(self, n):
        return self.data[self.index - n]

    def append(self, x):
        self.data[self.index] = x
        self.index += 1
        self.index %= self.maxEntries
        self.count += 1

    def all(self):
        return self.data[:len(self)]

    def __len__(self):
        return min(self.count, self.maxEntries)
