# import numpy as np
# import cv2
# from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt
#
#
# def get_tsne(feature_vecs):
#     tsne = TSNE()
#
#     return tsne.fit_transform(feature_vecs)
#
#
# def draw_graph(data, label):
#
#     plt.plot('name hank')
#     plt.scatter(data[0], data[1], c=label, cmap='hsv', s=0.3)
#     plt.colorbar()
#     plt.show()
#
#
# img1 = cv2.imread('./train64000.png', cv2.IMREAD_GRAYSCALE)
# img_vec1 = img1.reshape(480000)
#
# img2 = cv2.imread('./train0.png', cv2.IMREAD_GRAYSCALE)
# img_vec2 = img2.reshape(480000)
#
# tsnes = get_tsne([img_vec1, img_vec2]).reshape(2, -1)
# draw_graph(tsnes, [1, 8])

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from data import MoonDataset

train_set = MoonDataset('train')

img_vecs = []
labels = []

# for i in range(2):
#     img, label = train_set[i]
#     img_vec = img.numpy().reshape(480000)
#     img_vecs.append(img_vec)
#     labels.append(label.numpy()[0])
#
# tsne = TSNE()
# results = tsne.fit_transform(img_vecs).reshape[2, -1]
results = np.array((2, 4))
np.save('r.npy', results)

plt.plot('Train Data Graph')
plt.scatter(results[0], results[1], c=labels, cmap='hsv', s=0.3)
plt.colorbar()
plt.savefig('train_tsne.png')
