import numpy as np
import torch

class DataLoader:

    def __init__(self, NBatch, ContextSize, InputFileDir, Split):

        self.NBatch = NBatch
        self.ContextSize = ContextSize

        FileName = 'train.bin' if Split=='train' else 'val.bin'
        self.FilePath = f'{InputFileDir}/{FileName}'

        self.data = np.memmap(self.FilePath, dtype=np.uint16, mode='r')

        self.NFullLoop = 0
        self.current_pos = 0

    def next_batch(self):

        ChunkSize = self.NBatch*self.ContextSize
        
        this_data = self.data[self.current_pos:self.current_pos+(ChunkSize)+1] # +1 for target
        this_data = torch.from_numpy(this_data.astype(np.int64))

        x = this_data[:-1].view(self.NBatch, self.ContextSize)
        y = this_data[1:].view(self.NBatch, self.ContextSize)
        
        self.current_pos += ChunkSize

        if self.current_pos + (ChunkSize+1) > len(self.data):
            self.current_pos = 0
            self.NFullLoop += 1

        return x,y
    

DoTest = False

if DoTest:

    dl = DataLoader(4, 10, '/Users/jskim/Documents/MLStudy/mygpt/data/shakespeare', 'train')

    x, y = dl.next_batch()
    print(x)
    print(y)