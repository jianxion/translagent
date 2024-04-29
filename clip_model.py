import math
import sys
import pickle as pkl
import numpy as np

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from word.util import *

# initial commit

def sample_gumbel(shape, tt=torch, eps=1e-20):
    U = Variable(tt.FloatTensor(shape).uniform_(0, 1))
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temp, tt=torch):
    y = ( logits + sample_gumbel(logits.size(), tt) ) / temp
    return F.softmax(y)

def gumbel_softmax(logits, temp, hard, tt=torch):
    y = gumbel_softmax_sample(logits, temp, tt) # (batch_size, num_cat = 468)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = tt.FloatTensor(y.size()).zero_().scatter_(1, y_max_idx.data, 1)
        y = Variable( y_hard - y.data, requires_grad=False ) + y

    return y, y_max_idx

class TwoAgents(torch.nn.Module):
    def __init__(self, args):
        super(TwoAgents, self).__init__()
        self.agent1 = Agent(args.l1, args.l2, args)
        self.agent2 = Agent(args.l2, args.l1, args)

        self.agents = [self.agent1, self.agent2]
        self.num_cat = args.num_cat
        self.no_share_bhd = args.no_share_bhd
        self.train_how = args.train_how
        self.D_img = args.D_img
        self.D_hid = args.D_hid # 400
        self.l1 = args.l1
        self.l2 = args.l2

    def forward(self, data1, data2):
        a_spk_img, b_lsn_imgs = data1 # spk_imgs : (batch_size, 2048)
        b_spk_img, a_lsn_imgs = data2 # lsn_imgs : (batch_size, num_dist = 2, 2048)
        spk_inputs = [a_spk_img, b_spk_img] # [a, b]
        spk_outputs = [] # [a, b] logits
        lsn_inputs = [a_lsn_imgs, b_lsn_imgs] # [a, b]
        lsn_outputs = [] # [a, b]
        comm_onehots = [] # [a, b]
        comm_actions = []
        num_dist = b_lsn_imgs.size()[1]

        ##### Speaker #####
        for agent, spk_img in zip(self.agents, spk_inputs): # [a, b]
            spk_h_img = spk_img
            spk_h_img = agent.beholder1(spk_h_img) if self.no_share_bhd else agent.beholder(spk_h_img)

            spk_logits, comm_onehot, comm_action = agent.speaker(spk_h_img)
            spk_outputs.append(spk_logits)
            # spk_logits : (batch_size, num_cat)
            # comm_onehot : (batch_size, num_cat)
            # comm_action : (batch_size)

            # comm_onehots.append(comm_onehot.cuda())
            # comm_actions.append(comm_action.cuda())
            comm_onehots.append(comm_onehot)
            comm_actions.append(comm_action)

        comm_onehots = comm_onehots[::-1] # [b, a]

        ##### Listener #####
        for agent, comm_onehot, lsn_imgs in zip(self.agents, comm_onehots, lsn_inputs):
            # lsn_imgs : (batch_size, num_dist, 2048)
            lsn_imgs = lsn_imgs.view(-1, self.D_img) # (batch_size * num_dist, D_img = 2048)
            lsn_h_imgs = agent.beholder2(lsn_imgs) if self.no_share_bhd else agent.beholder(lsn_imgs)
            lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid) # (batch_size, num_dist, D_hid)

            lsn_dot = agent.listener(lsn_h_imgs, comm_onehot) # (batch_size, num_dist)
            lsn_outputs.append(lsn_dot)

        return (spk_outputs[0], lsn_outputs[1]), (spk_outputs[1], lsn_outputs[0]), comm_actions

    def translate_from_en(self, sample=False, print_neighbours=False):
        l1_dic = get_idx_to_cat(self.l1) 
        l2_dic = get_idx_to_cat(self.l2)

        # Debug: Print dictionary stats and types
        print("Some keys in l1_dic:", list(l1_dic.keys())[:5])
        print("Key types in l1_dic:", type(next(iter(l1_dic.keys()))))
        print("Some keys in l2_dic:", list(l2_dic.keys())[:5])
        print("Key types in l2_dic:", type(next(iter(l2_dic.keys()))))

        result = {1: {1: [], 0: []}, 0: {1: [], 0: []}}
        batch_size = 468
        keys = np.arange(1, self.num_cat + 1)
        labels = torch.LongTensor(keys).view(batch_size, 1)

        onehot1 = torch.FloatTensor(batch_size, self.num_cat).zero_()
        onehot1.scatter_(1, labels - 1, 1)
        onehot1 = Variable(onehot1, requires_grad=False)#.cuda()

        logits1 = self.agent2.translate(onehot1)
        onehot2, idx2 = sample_logit_to_onehot(logits1) if sample else max_logit_to_onehot(logits1)
        logits2 = self.agent1.translate(onehot2)
        onehot3, idx3 = sample_logit_to_onehot(logits2) if sample else max_logit_to_onehot(logits2)

        _, indices1 = torch.sort(logits1, 1, descending=True)
        _, indices2 = torch.sort(logits2, 1, descending=True)
        indices1 = indices1.cpu().data.numpy()
        indices2 = indices2.cpu().data.numpy()

        for idx in range(labels.nelement()):
            label_idx = labels[idx][0].item()
            if label_idx not in l1_dic:
                print(f"Label index {label_idx} not in l1_dic")
                continue

            if print_neighbours:
                for k in range(5):
                    k_index = indices1[idx][k].item()
                    if k_index not in l2_dic:
                        print(f"Index {k_index} not in l2_dic")
                        continue
                    print("{:>25} -> {:>25}".format(l1_dic[label_idx], l2_dic[k_index]))

            right1 = right2 = 0
            idx2_item = idx2[idx][0].item()
            idx3_item = idx3[idx][0].item()
            if label_idx == idx2_item and idx2_item in l2_dic:
                right1 = 1
            if label_idx == idx3_item and idx3_item in l1_dic:
                right2 = 1

            result[right1][right2].append((l1_dic[label_idx], l2_dic.get(idx2_item, 'Unknown'), l1_dic.get(idx3_item, 'Unknown')))

            if right2 == 0 or right1 == 0:
                print(right1, right2)
                for k in range(5):
                    k_index = indices1[idx][k].item()
                    if k_index not in l2_dic:
                        print(f"Index {k_index} not in l2_dic")
                        continue
                    print("{:>25} -> {:>25}".format(l1_dic[label_idx], l2_dic[k_index]))

        return result



    def en2de(self, onehot):
        logits = self.agent2.translate(onehot)
        return logits

    def de2en(self, onehot):
        logits = self.agent1.translate(onehot)
        return logits

    def en2de2en(self, onehot):
        logits1 = self.agent2.translate(onehot)
        onehot2, _ = max_logit_to_onehot(logits1)
        logits2 = self.agent1.translate(onehot2)
        return logits2

    def precision(self, keys, bs):
        ks = [1, 5, 20]
        result = []
        rounds = ["{}->{} (agent2/{}) ".format(self.l1, self.l2, self.l2),
                  "{}->{} (agent1/{}) ".format(self.l2, self.l1, self.l1),
                  "{}->{}->{} (agent2/{}->agent1/{}) ".format(self.l1, self.l2, self.l1, self.l2, self.l1)]

        for which_round, round_ in enumerate(rounds):
            acc = [[0,0] for x in range(len(ks))]
            cnt = 0
            for batch_idx in range(int(math.ceil( float(len(keys)) / bs ) ) ):
                labels_ = np.arange(batch_idx * bs , min(len(keys), (batch_idx+1) * bs ) )
                labels_ = keys[labels_]
                batch_size = len(labels_)
                cnt += batch_size

                labels = torch.LongTensor(labels_).view(-1)
                labels = torch.unsqueeze(labels, 1)
                #labels = Variable(labels, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False)

                onehot = torch.FloatTensor(batch_size, self.num_cat)
                onehot.zero_()
                onehot.scatter_(1, labels.data.cpu(), 1)
                #onehot = Variable(onehot, requires_grad=False).cuda()
                onehot = Variable(onehot, requires_grad=False)
                

                if which_round == 0:
                    logits = self.en2de(onehot)
                elif which_round == 1:
                    logits = self.de2en(onehot)
                elif which_round == 2:
                    logits = self.en2de2en(onehot)

                for prec_idx, k in enumerate(ks):
                    right, total = logit_to_top_k(logits, labels, k)
                    acc[prec_idx][0] += right
                    acc[prec_idx][1] += total
            assert( cnt == len(keys) )
            assert( acc[0][1] == len(keys) )

            pm = round_
            for prec_idx, k in enumerate(ks):
                curr_acc = float(acc[prec_idx][0]) / acc[prec_idx][1] * 100
                result.append( curr_acc )
                pm += "| P@{} {:.2f}% ".format(k, curr_acc )
            print (pm)

        return result

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

class Beholder(nn.Module):
    def __init__(self, dropout):
        super(Beholder, self).__init__()
        self.clip_model = clip_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, image, text):
        # Assume image is already preprocessed
        text = clip.tokenize([text]).to(device)  # Tokenize text input for CLIP
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

        combined_features = torch.cat((image_features, text_features.squeeze(0)), dim=1)
        combined_features = self.dropout(combined_features)
        return combined_features

class Agent(nn.Module):
    def __init__(self, native, foreign, args):
        super(Agent, self).__init__()
        self.beholder = Beholder(args.dropout)  
        self.speaker = Speaker(native, foreign, args.D_hid, args.num_cat, args.dropout, args.temp, args.hard, args.tt)
        self.listener = Listener(native, foreign, args.D_hid, args.num_cat, args.dropout)

    def forward():
        return
    
    def translate(self, image, text):
        combined_features = self.beholder(image, text)
        spk_logits = self.speaker(combined_features)
        return spk_logits

 

# Define the Scaled Dot-Product Attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale

    def forward(self, query, key, value):
        # Calculate the dot product (query * key^T)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Multiply by values
        output = torch.matmul(attention_probs, value)
        return output, attention_probs

class Speaker(torch.nn.Module):
    def __init__(self, native, foreign, D_hid, num_cat, dropout, temp, hard, tt):
        super(Speaker, self).__init__()
        self.hid_to_cat = torch.nn.Linear(D_hid, num_cat, bias=False) # Speaker
        self.drop = torch.nn.Dropout(p=dropout)
        self.num_cat = num_cat
        self.temp = temp
        self.hard = hard
        self.tt = tt
        self.native, self.foreign = native, foreign

         # Attention mechanism
        self.attention = ScaledDotProductAttention(scale=D_hid ** 0.5)
        self.query_projection = nn.Linear(D_hid, D_hid)
        self.key_projection = nn.Linear(D_hid, D_hid)
        self.value_projection = nn.Linear(D_hid, D_hid)

    def forward(self, h_img):
        #print(h_img.shape) # 128, 400
        query = self.query_projection(h_img)
        key = self.key_projection(h_img)
        value = self.value_projection(h_img)

        attended_values, _ = self.attention(query, key, value)
        attended_values = attended_values + h_img  # Residual connection

        spk_logits = self.hid_to_cat(attended_values)
        comm_onehot, comm_label = gumbel_softmax(spk_logits, temp=self.temp, hard=self.hard, tt=self.tt)
        return spk_logits, comm_onehot, comm_label

    def nn_words(self, batch_size = 5):
        word_idx = np.random.randint(0, self.num_cat, size=batch_size)
        for idx in word_idx:
            self.compute_dot_for_all(idx)

    def compute_dot_for_all(self, idx):
        l1_dic = get_idx_to_cat(self.native)
        assert len(l1_dic) == self.num_cat

        emb = torch.FloatTensor(self.hid_to_cat.weight.data.cpu()) # [num_cat, D_hid]
        vec = emb[idx] # [1, D_hid]

        vec_exp = torch.unsqueeze(vec,0).expand(emb.size()) # (num_cat, D_hid)
        prod = torch.mul(vec_exp, emb) # [num_cat, D_hid]
        prod = torch.sum(prod, 1) # [num_cat]
        norm1 = torch.norm(vec_exp, 2, 1) # [num_cat]
        norm2 = torch.norm(emb, 2, 1) # (num_cat)
        norm = torch.mul(norm1, norm2) # [num_cat]

        ans = prod / norm
        ans = ans.view(self.num_cat)

        logits_sorted, indices = torch.sort(ans, dim=0, descending=True)
        indices = indices[:5].cpu().numpy()

        print ("{} -> {}".format(l1_dic[idx], ", ".join(["{} ({:.2f})".format(l1_dic[idx1], ans[idx1]) for idx1 in indices])))

class Listener(nn.Module):
    def __init__(self, native, foreign, D_hid, num_cat, dropout):
        super(Listener, self).__init__()
        self.emb = nn.Linear(num_cat, D_hid, bias=False)
        self.D_hid = D_hid
        self.num_cat = num_cat
        self.native, self.foreign = native, foreign

        # Attention mechanism
        self.attention = ScaledDotProductAttention(scale=D_hid ** 0.5)

    def forward(self, lsn_h_imgs, comm_onehot):
        lsn_hid_msg = self.emb(comm_onehot)  # Embedding the communication onehot to hidden dimension

        # Prepare for attention by expanding dimensions
        lsn_hid_msg = lsn_hid_msg.unsqueeze(1).repeat(1, lsn_h_imgs.size(1), 1)
        
        # Apply attention
        attended_output, _ = self.attention(lsn_hid_msg, lsn_h_imgs, lsn_h_imgs)
        attended_output = attended_output + lsn_hid_msg  # Residual connection

        # Compute the difference
        diff = torch.pow(attended_output - lsn_h_imgs, 2)
        diff = torch.mean(diff, 2)  # (batch_size, num_dist)
        diff = 1 / (diff + 1e-10)
        diff = diff.squeeze()

        return diff

    def nn_words(self, batch_size = 5):
        word_idx = np.random.randint(0, self.num_cat, size=batch_size)
        for idx in word_idx:
            self.compute_dot_for_all(idx)

    def compute_dot_for_all(self, idx):
        l1_dic = bergsma_words(self.foreign)
        #assert len(l1_dic) == self.num_cat

        emb = torch.FloatTensor(torch.t(self.emb.weight.data.cpu())) # [num_cat, D_hid]
        vec = emb[idx] # [1, D_hid]

        vec_exp = torch.unsqueeze(vec,0).expand(emb.size()) # (num_cat, D_hid)
        prod = torch.mul(vec_exp, emb) # [num_cat, D_hid]
        prod = torch.sum(prod, 1) # [num_cat]
        norm1 = torch.norm(vec_exp, 2, 1) # [num_cat]
        norm2 = torch.norm(emb, 2, 1) # (num_cat)
        norm = torch.mul(norm1, norm2) # [num_cat]

        ans = prod / norm
        ans = ans.view(-1)

        logits_sorted, indices = torch.sort(ans, dim=0, descending=True)
        indices = indices[:5].cpu().numpy()

        print ("{} -> {}".format(l1_dic[idx], ", ".join(["{} ({:.2f})".format(l1_dic[idx1], ans[idx1]) for idx1 in indices])))

