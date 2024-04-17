import argparse
import sys
import pickle as pkl
import os
import json
import datetime
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.autograd as autograd

from util import *
from models import *
from split import *
from load_data import *

def forward_pass_gumbel(data, model, loss_dict, args, loss_fn):
    data1 = next_batch_dist_batch(data[0], args.batch_size, args.num_dist, args.num_cat) # [spk_img, lst_imgs, label_onehot, label, which]
    data2 = next_batch_dist_batch(data[1], args.batch_size, args.num_dist, args.num_cat)

    # spk_imgs : (batch_size, 2048)
    # lsn_imgs : (batch_size, num_dist, 2048)
    # label : (batch_size)
    # which : (batch_size)

    output1, output2, comm_actions = model(data1[:2], data2[:2])
    # output1 : (a_spk_logits, b_lsn_dot)
    # output2 : (b_spk_logits, a_lsn_dot)
    # a_spk_logits : (batch_size, num_cat)
    # a_lsn_dot : (batch_size, num_dist)
    # comm_actions : (batch_size)

    final_loss = 0
    for output, label, direction, comm in zip((output1, output2), (data1[3:], data2[3:]), "agent1 agent2".split(), comm_actions):
        for each_output, each_label, person in zip(output, label, "spk lsn".split()):
            # each_output : [a_spk_logits, b_lsn_dot]
            # each_label : [labels, whichs]

            loss = loss_fn(each_output, each_label)
            acc = logit_to_acc(each_output, each_label)
            if person == "lsn":
                final_loss += loss * args.alpha
            elif person == "spk" and not args.lsn_loss_only:
                final_loss += loss

            loss_dict[direction][person]["loss"].update(loss.item())
            loss_dict[direction][person]["acc"].update(acc * 100)

    return final_loss

