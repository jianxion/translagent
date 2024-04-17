import json
import pickle as pkl
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
# from torch.utils.serialization import load_lua

from util import *

random = np.random

def next_batch_dist_batch(data, batch_size, num_dist, num_cat):
    lsn_imgs, spk_imgs, labels, whichs = [], [], [], []
    keys = list(data.keys())
    assert len(keys) >= num_dist

    for _ in range(batch_size):
        rand_labels = random.choice(keys, num_dist, replace=False)  # (num_dist)
        num_imgs = [len(data[label]) for label in rand_labels]  # (num_dist)
        img_indices = [random.randint(0, num_img) for num_img in num_imgs]
        all_images = [data[label][img_idx] for label, img_idx in zip(rand_labels, img_indices)]  # (num_dist, 2048)

        which = random.randint(0, num_dist)  # (1)
        label_ = rand_labels[which]  # (1)

        lsn_imgs.append(all_images)  # (batch_size, num_dist, 2048)
        spk_imgs.append(all_images[which])  # (batch_size, 2048)
        labels.append(label_)  # (batch_size)
        whichs.append(which)  # (batch_size)

    # Convert lists of tensors to stacked tensors # removed.cuda()
    spk_imgs = torch.stack(spk_imgs).view(batch_size, -1)
    lsn_imgs = torch.stack([torch.stack(x) for x in lsn_imgs]).view(batch_size, num_dist, -1)
    labels_ = torch.LongTensor(labels).view(batch_size)
    whichs = torch.LongTensor(whichs).view(batch_size)
    label_onehot = idx_to_onehot(torch.LongTensor(labels), num_cat).view(batch_size, num_cat)

    return spk_imgs, lsn_imgs, label_onehot, labels_, whichs
