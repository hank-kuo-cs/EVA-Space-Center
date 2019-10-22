import os
import torch
import logging
from torch.utils.data import DataLoader
from data import MoonDataset
from net import VGG19
from config import *
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%m-%d %H:%M:%S')

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


logging.info('Load data')
train_set = MoonDataset('train')
train_loader = DataLoader(train_set, BATCH_SiZE, True, num_workers=2)

logging.info('Set VGG model')
net = VGG19().to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

logging.info('Start training')
for epoch in range(EPOCH_NUM):
    running_loss = 0.0

    for i, data in enumerate(tqdm(train_loader)):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs.double(), labels)
        if i % 5 == 0:
            print(loss)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % 1000 == 999:
            logging.info('[%d epoch, %5d step] loss: %.6f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    if epoch % 20 == 19 and epoch > 0:
            model_path = 'checkpoint/model_epoch%d.pth' % (epoch + 1)
            torch.save(net.state_dict(), model_path)

logging.info('Finished training')
