from config import *
from data import MoonDataset
from torch.utils.data import DataLoader


dev_dataset = MoonDataset('train')
print(len(dev_dataset.image_files))
print(len(dev_dataset.label_files))
train_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

for i, data in enumerate(train_loader):
    print('i = %d' % i)
    labels = data[1]
    for j, v in enumerate(labels[0]):
        if v < 0 or v > 1:
            print('i = %d, j = %d, label =' % (i, j), labels[0])

