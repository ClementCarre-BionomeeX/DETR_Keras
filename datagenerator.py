# %%
import copy
import json
from fileinput import filename

import numpy as np
import tensorflow as tf


class Datagenerator(tf.keras.utils.Sequence):
    def __init__(self, tr_images_path, tr_annots_path, batchsize, nqueries, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.images_path = tr_images_path
        self.annots_path = tr_annots_path

        self.batchsize = batchsize

        with open(self.annots_path, "r") as file:
            self.data = json.load(file)

        self.id_to_idx_dict = {
            cat["id"]: i + 1 for i, cat in enumerate(self.data["categories"])
        }
        self.indexes = np.arange(len(self.data["images"]))

        self.nqueries = nqueries

        self.size_list = [
            128, 256, 334, 480, 512, 640
        ]

        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batchsize : (index + 1) * self.batchsize]

        h, w = np.random.choice(self.size_list, 2, replace=True)

        X = np.zeros((self.batchsize, h, w, 3), dtype=np.uint8)
        Y = np.zeros((self.batchsize, self.nqueries, 4 + 1 + len(self.id_to_idx_dict)), dtype=np.float32)

        for i in range(self.batchsize):
            X[i, ], Y[i, ] = self._get_datum(indexes[i], (h, w))

        return X, Y

    def _get_datum(self, idx, target_size=None):
        # read image
        H, W = target_size

        img_data = self.data["images"][self.indexes[idx]]
        id = img_data["id"]
        filename = self.images_path + f"{id:0>12}.jpg"
        img = tf.keras.utils.load_img(filename, target_size=target_size)

        oH = img_data["height"]
        oW = img_data["width"]

        # read annotations
        annots = [
            [
                ann["bbox"][0] / oW,
                ann["bbox"][1] / oH,
                ann["bbox"][2] / oW,
                ann["bbox"][3] / oH,
                *tf.keras.utils.to_categorical(
                    self.id_to_idx_dict[ann["category_id"]],
                    num_classes=len(self.id_to_idx_dict) + 1,
                ),
            ]
            for ann in self.data["annotations"]
            if ann["image_id"] == id
        ]

        # add empty bbox to fill the number of queries
        for i in range(self.nqueries - len(annots)):
            annots.append(
                [
                    0, 0, 0, 0,
                    *tf.keras.utils.to_categorical(
                        0,
                        num_classes=len(self.id_to_idx_dict) + 1,
                    ),
                ]
            )

        return img, np.asarray(annots)

    def __len__(self):
        return int(len(self.indexes) // self.batchsize)


# %%

basepath = r"/home/ccarre/Documents/COCO"
annotspath = r"/annotations_trainval2017/annotations/"
trannotspath = r"instances_train2017.json"
vaannotspath = r"instances_val2017.json"

trimagespath = r"/train2017/"
vaimagespath = r"/val2017/"


# %%
dg = Datagenerator(basepath + trimagespath, basepath + annotspath + trannotspath, 9, 128)

print(len(dg))

import matplotlib.pyplot as plt

# %%

dg.on_epoch_end()

img, ann = dg[0]

fig = plt.figure(figsize=(15, 15))

for i in range(9):
    fig.add_subplot(3, 3, i + 1)
    plt.imshow(img[i])
    plt.axis('off')

# %%

# for k in dg.annotations_train.keys():
#     print(k)

# # %%

# print([img for img in dg.annotations_train["images"] if img["id"] == dg.annotations_train["annotations"][0]["image_id"]])

# bbox = dg.annotations_train["annotations"][0]["bbox"]
# # %%

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# img = plt.imread(basepath + "/train2017/" + f'{dg.annotations_train["annotations"][0]["image_id"]:0>12}.jpg')

# # Create figure and axes
# fig, ax = plt.subplots()

# # Display the image
# ax.imshow(img)

# # Create a Rectangle patch
# rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')

# # Add the patch to the Axes
# ax.add_patch(rect)

# plt.show()

# # %%

# %%

print(dg.data["annotations"][0])

# %%
