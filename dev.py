from config import *
from data import MoonDataset
from torch.utils.data import DataLoader


dev_dataset = MoonDataset('train')

train_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


for i, data in enumerate(train_loader):
    print(data[0])
