import re
from data import MoonDataset


dev_dataset = MoonDataset('train')

print('train dataset file loading unit test')

test = [False for i in range(0, 80000)]

for img_file in dev_dataset.image_files:
    s = re.search(r'random_(.+?).png', img_file)
    img_num = int(img_file[s.regs[0][0]: s.regs[0][1]])

    test[img_num] = True


for i, t in enumerate(test):
    if not t:
        print('image %d wrong' % i)
