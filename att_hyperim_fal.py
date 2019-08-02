#!/usr/bin/env python
# coding: utf-8

# learnable `alpha` in FLLoss

# In[1]:


import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from gensim.models import KeyedVectors
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from torch.autograd import Variable
import geoopt as gt

# from tqdm import tqdm_notebook as tqdm


# In[2]:

default_dtype = torch.float64
cuda_device = torch.device('cuda:3')
torch.cuda.set_device(device=cuda_device)

data_dir = Path('/home/blchen/nb/data/CCKS2019/CCKS_2019_-D/dataset')
train_path = data_dir / "sentTrain_resample.txt"
val_path = data_dir / "sentDev.txt"
test_path = data_dir / "sent_test.txt"

# ## load word2vec model

# In[3]:


w2v_model = KeyedVectors.load_word2vec_format('w2v.wv')
w2v_model.add("<UNK>", np.zeros(w2v_model.vector_size))
w2v_model.add("<PAD>", np.zeros(w2v_model.vector_size))
w2v_weights = torch.FloatTensor(w2v_model.vectors)
word_dict = {w: i for i, w in enumerate(w2v_model.index2entity)}  # mapping for word to index of embedding weight matrix

# ## input data processing
# The max length of sentences is 41

# In[4]:


max_sen_length = 40


def get_en_pos(words, en1, en2):
    # en_pos = (words.index(en1), list(reversed(words)).index(en2))
    en1_pos, en2_pos = None, None
    for i in range(len(words)):
        if en1 is not None and words[i] == en1:
            en1_pos = i
        if words[i] == en2:
            en2_pos = i
    if en1_pos is None:
        en1_pos = 0
    if en2_pos is None:
        en2_pos = len(words) - 1
    return en1_pos, en2_pos


def make_input_data(fname: str, test=False):
    sen_id_list, en_pos_list, word_indices_list, label_list = [], [], [], []
    with open(fname, "r", encoding="utf8") as f:
        if not test:
            for line in f:
                content = line.split()
                sen_id = content[-2]
                words = content[3:-1]
                words = words[:max_sen_length]
                words.extend(["<PAD>"] * (max_sen_length - len(words)))
                word_indices = [word_dict[w] if w in word_dict else word_dict["<UNK>"] for w in words]
                en1 = content[1]
                en2 = content[2]
                en_pos = get_en_pos(words, en1, en2)
                lbl = int(content[-1])

                sen_id_list.append(sen_id)
                en_pos_list.append(en_pos)
                word_indices_list.append(word_indices)
                label_list.append(lbl)

            return sen_id_list, torch.LongTensor(en_pos_list), \
                   torch.LongTensor(word_indices_list), \
                   torch.LongTensor(label_list)
        for line in f:
            content = line.split()
            sen_id = content[0]
            words = content[3:]
            words = words[:max_sen_length]
            words.extend(["<PAD>"] * (max_sen_length - len(words)))
            word_indices = [word_dict[w] if w in word_dict else word_dict["<UNK>"] for w in words]
            en1 = content[1]
            en2 = content[2]
            en_pos = get_en_pos(words, en1, en2)

            sen_id_list.append(sen_id)
            en_pos_list.append(en_pos)
            word_indices_list.append(word_indices)

        return sen_id_list, \
               torch.LongTensor(en_pos_list), torch.LongTensor(word_indices_list)


_, train_en_pos, X_train, y_train = make_input_data(train_path)
_, val_en_pos, X_val, y_val = make_input_data(val_path)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train, train_en_pos)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=500, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(X_val, y_val, val_en_pos)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000)

y_non_zero_indices = (y_val != 0)
val_nonzero_dataset = torch.utils.data.TensorDataset(X_val[y_non_zero_indices],
                                                     y_val[y_non_zero_indices],
                                                     val_en_pos[y_non_zero_indices])
val_nonzero_loader = torch.utils.data.DataLoader(val_nonzero_dataset, batch_size=1000)

# ## model

# In[5]:

# ## model

# In[5]:

embd_dim = w2v_model.vector_size
# unique_classes, _, class_counts = torch.unique(y_train, sorted=True, return_counts=True)
n_classes = torch.unique(y_train).size()[0]

class_freq_inverse = [torch.log(len(y_train) / (y_train == i).sum()).item() for i in range(n_classes)]


class PCNNNet(nn.Module):
    def __init__(self, n_classes=35, embd_dim=300, sequence_length=40, out_channels=210, kernel_height=3,
                 dense_hidden=35, piecewise_pooling=False):
        super(PCNNNet, self).__init__()
        self.embed = nn.Embedding.from_pretrained(w2v_weights)
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=out_channels,
                              kernel_size=(kernel_height, embd_dim),
                              padding=(1, 0))
        self.pmp = nn.MaxPool2d((sequence_length - kernel_height + 1, 1))
        if piecewise_pooling:
            self.dense = nn.Linear(sequence_length * 3, dense_hidden)
        else:
            self.dense = nn.Linear(sequence_length, dense_hidden)
        self.out = nn.Linear(dense_hidden, n_classes)
        self.sequence_length = sequence_length
        self.piecewise_pooling = piecewise_pooling

    def forward(self, x, pos_):
        ep1 = pos_[:, 0]
        ep2 = pos_[:, 1]
        embed = self.embed(x).unsqueeze(1)  # [batch, 1, sequence_length=40, embd_dim]

        conv = F.relu(self.conv(embed)).squeeze(3)  # [batch, out_channels, sequence_length=40]
        if self.piecewise_pooling:
            be1_mask = torch.zeros_like(conv)
            aes_mask = torch.zeros_like(conv)
            be2_mask = torch.zeros_like(conv)

            # be1_pad = torch.ones_like(conv) * -100
            # aes_pad = torch.ones_like(conv) * -100
            # be2_pad = torch.ones_like(conv) * -100

            for i in range(x.size(0)):
                if ep1[i] > ep2[i]:
                    ep1[i], ep2[i] = ep2[i], ep1[i]
                if ep1[i] == 0:
                    ep1[i] += 1
                    ep2[i] += 1
                be1_mask[i, :, :ep1[i]] = 1
                aes_mask[i, :, ep1[i]:ep2[i]] = 1
                be2_mask[i, :, ep2[i]:] = 1
            # be1 = conv * be1_mask + be1_pad
            # aes = conv * aes_mask + aes_pad
            # be2 = conv * be2_mask + be2_pad

            be1 = conv * be1_mask
            aes = conv * aes_mask
            be2 = conv * be2_mask

            p1 = self.pmp(be1)
            p2 = self.pmp(aes)
            p3 = self.pmp(be2)
            pooled = torch.cat((p1, p2, p3), dim=2).view(-1, self.sequence_length * 3)
        else:
            pooled = self.pmp(conv).view(conv.size(0), -1)
        return self.out(self.dense(pooled))


class LGMLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, alpha=0.1, lambda_=0.01):
        super(LGMLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.lambda_ = lambda_
        self.means = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.means, gain=2 ** 0.5)

    def forward(self, feat, labels=None):
        batch_size = feat.size()[0]

        XY = torch.matmul(feat, torch.transpose(self.means, 0, 1))
        XX = torch.sum(feat ** 2, dim=1, keepdim=True)
        YY = torch.sum(torch.transpose(self.means, 0, 1) ** 2, dim=0, keepdim=True)
        neg_sqr_dist = -0.5 * (XX - 2.0 * XY + YY)

        if labels is None:
            psudo_labels = torch.argmax(neg_sqr_dist, dim=1)
            means_batch = torch.index_select(self.means, dim=0, index=psudo_labels)
            likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch) ** 2) / 2) * (1. / batch_size)
            return neg_sqr_dist, likelihood_reg_loss, self.means

        labels_reshped = labels.view(labels.size()[0], -1)

        device = self.means.device
        # if torch.cuda.is_available():
        #     ALPHA = torch.zeros(batch_size, self.num_classes).cuda().scatter_(1, labels_reshped, self.alpha)
        #     K = ALPHA + torch.ones([batch_size, self.num_classes]).cuda()
        # else:
        ALPHA = torch.zeros(batch_size, self.num_classes, device=device).scatter_(1, labels_reshped, self.alpha)
        K = ALPHA + torch.ones([batch_size, self.num_classes], device=device)

        logits_with_margin = torch.mul(neg_sqr_dist, K)
        means_batch = torch.index_select(self.means, dim=0, index=labels)
        likelihood_reg_loss = self.lambda_ * (torch.sum((feat - means_batch) ** 2) / 2) * (1. / batch_size)
        return logits_with_margin, likelihood_reg_loss, self.means


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = nn.Parameter(torch.tensor((alpha, 1 - alpha)))
        if isinstance(alpha, list):
            self.alpha = nn.Parameter(torch.tensor(alpha))
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


PROJ_EPS = 1e-5
EPS = 1e-15


def project_hyp_vec(x):
    # To make sure hyperbolic embeddings are inside the unit ball.
    norm = torch.sum(x ** 2, dim=-1, keepdim=True)

    return x * (1. - PROJ_EPS) / torch.clamp(norm, 1. - PROJ_EPS)


def asinh(x):
    return torch.log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    return torch.log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def poinc_dist(u, v):
    m = mob_add(-u, v) + EPS
    atanh_x = torch.norm(m, dim=-1, keepdim=True)
    dist_poincare = 2.0 * atanh(atanh_x)
    return dist_poincare


def euclid_dist(u, v):
    return torch.norm(u - v, dim=-1, keepdim=True)


def mob_add(u, v):
    v = v + EPS

    norm_uv = 2 * torch.sum(u * v, dim=-1, keepdim=True)
    norm_u = torch.sum(u ** 2, dim=-1, keepdim=True)
    norm_v = torch.sum(v ** 2, dim=-1, keepdim=True)

    denominator = 1 + norm_uv + norm_v * norm_u
    result = (1 + norm_uv + norm_v) / denominator * u + (1 - norm_u) / denominator * v

    return project_hyp_vec(result)


def mob_scalar_mul(r, v):
    v = v + EPS
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    nomin = torch.tanh(r * atanh(norm_v))
    result = nomin / norm_v * v

    return project_hyp_vec(result)


def mob_mat_mul(M, x):
    x = project_hyp_vec(x)
    Mx = x.matmul(M)
    Mx_norm = torch.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = torch.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(torch.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def mob_mat_mul_d(M, x, d_ball):
    x = project_hyp_vec(x)
    Mx = x.view(x.shape[0], -1).matmul(M.view(M.shape[0] * d_ball, M.shape[0] * d_ball)).view(x.shape)
    Mx_norm = torch.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = torch.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(torch.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def lambda_x(x):
    return 2. / (1 - torch.sum(x ** 2, dim=-1, keepdim=True))


def exp_map_x(x, v):
    v = v + EPS
    second_term = torch.tanh(lambda_x(x) * torch.norm(v) / 2) / torch.norm(v) * v
    return mob_add(x, second_term)


def log_map_x(x, y):
    diff = mob_add(-x, y) + EPS
    return 2. / lambda_x(x) * atanh(torch.norm(diff, dim=-1, keepdim=True)) / \
           torch.norm(diff, dim=-1, keepdim=True) * diff


def exp_map_zero(v):
    v = v + EPS
    norm_v = torch.norm(v, dim=-1, keepdim=True)
    result = torch.tanh(norm_v) / norm_v * v

    return project_hyp_vec(result)


def log_map_zero(y):
    diff = project_hyp_vec(y + EPS)
    norm_diff = torch.norm(diff, dim=-1, keepdim=True)
    return atanh(norm_diff) / norm_diff * diff


def mob_pointwise_prod(x, u):
    # x is hyperbolic, u is Euclidean
    x = project_hyp_vec(x + EPS)
    Mx = x * u
    Mx_norm = torch.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = torch.norm(x, dim=-1, keepdim=True)

    result = torch.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx
    return project_hyp_vec(result)


class hyperRNN(nn.Module):

    def __init__(self, input_size, hidden_size, d_ball):
        super(hyperRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_ball = d_ball

        k = (1 / hidden_size) ** 0.5
        self.w = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.u = gt.ManifoldParameter(gt.ManifoldTensor(input_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.b = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, manifold=gt.PoincareBall()).zero_())

    def transition(self, x, h):
        W_otimes_h = mob_mat_mul_d(self.w, h, self.d_ball)
        U_otimes_x = mob_mat_mul_d(self.u, x, self.d_ball)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)

        return mob_add(Wh_plus_Ux, self.b)

    def init_rnn_state(self, batch_size, hidden_size, device=cuda_device):
        return torch.zeros((batch_size, hidden_size, self.d_ball), dtype=default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.transition(x, hidden)
            outputs += [hidden]
        return torch.stack(outputs).transpose(0, 1)


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size, d_ball):
        super(GRUCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_ball = d_ball

        k = (1 / hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.w_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.w_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.u_z = gt.ManifoldParameter(gt.ManifoldTensor(input_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.u_r = gt.ManifoldParameter(gt.ManifoldTensor(input_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.u_h = gt.ManifoldParameter(gt.ManifoldTensor(input_size, d_ball, hidden_size, d_ball).uniform_(-k, k))
        self.b_z = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, manifold=gt.PoincareBall()).zero_())
        self.b_r = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, manifold=gt.PoincareBall()).zero_())
        self.b_h = gt.ManifoldParameter(gt.ManifoldTensor(hidden_size, d_ball, manifold=gt.PoincareBall()).zero_())

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = mob_mat_mul_d(W, h, self.d_ball)
        U_otimes_x = mob_mat_mul_d(U, x, self.d_ball)
        Wh_plus_Ux = mob_add(W_otimes_h, U_otimes_x)

        return mob_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = torch.sigmoid(log_map_zero(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = torch.sigmoid(log_map_zero(r))

        r_point_h = mob_pointwise_prod(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)
        # h_tilde = torch.tanh(log_map_zero(h_tilde)) # non-linearity

        minus_h_oplus_htilde = mob_add(-hidden, h_tilde)
        new_h = mob_add(hidden, mob_pointwise_prod(minus_h_oplus_htilde, z))

        return new_h


class hyperGRU(nn.Module):

    def __init__(self, input_size, hidden_size, d_ball):
        super(hyperGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.d_ball = d_ball

        self.gru_cell = GRUCell(input_size, hidden_size, d_ball)

    def init_gru_state(self, batch_size, hidden_size, device=cuda_device):
        return torch.zeros((batch_size, hidden_size, self.d_ball), dtype=default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return torch.stack(outputs).transpose(0, 1)


class HyperIM(nn.Module):

    def __init__(self, feature_num, word_embed, label_embed, d_ball, hidden_size=5, if_gru=False, **kwargs):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.d_ball = d_ball

        self.word_embed = gt.ManifoldParameter(word_embed, manifold=gt.PoincareBall())
        self.label_embed = gt.ManifoldParameter(label_embed, manifold=gt.PoincareBall())

        if (if_gru):
            self.rnn = hyperGRU(input_size=word_embed.shape[1], hidden_size=self.hidden_size, d_ball=self.d_ball)
        else:
            self.rnn = hyperRNN(input_size=word_embed.shape[1], hidden_size=self.hidden_size, d_ball=self.d_ball)

        self.dense_1 = nn.Linear(feature_num, int(feature_num * 2))
        self.dense_2 = nn.Linear(int(feature_num * 2), 1)

    def forward(self, x):
        word_embed = self.word_embed[x]
        encode = self.rnn(word_embed)

        encode = encode.unsqueeze(dim=2)
        encode = encode.expand(-1, -1, self.label_embed.shape[0], -1, -1)

        interaction = poinc_dist(encode, self.label_embed.expand_as(encode))
        interaction = interaction.squeeze(dim=-1).sum(dim=-1).transpose(1, 2)

        out = F.relu(self.dense_1(interaction))
        out = self.dense_2(out).squeeze(dim=-1)

        return out


class Net(nn.Module):
    def __init__(self, n_classes=35, embd_dim=300, sequence_length=40, out_channels=35, kernel_height=3,
                 dense_hidden=30, feat_dim=35, alpha1=0.1, lambda1=0.01, gamma=0, alpha2=None, size_average=True,
                 lambda2=1.0):
        super(Net, self).__init__()
        self.models = nn.ModuleList([
            PCNNNet(n_classes, embd_dim, sequence_length, out_channels, kernel_height, dense_hidden),
            LGMLoss(n_classes, feat_dim, alpha1, lambda1),
            FocalLoss(gamma, alpha2, size_average)
        ])
        self.lambda2 = lambda2

    def forward(self, x, y, pos_):
        out = self.models[0](x, pos_)
        out, lgm_loss_, _ = self.models[1](out, y)
        if y is None:
            return out

        fl_loss_ = self.models[2](out, y)
        return out, fl_loss_ + self.lambda2 * lgm_loss_

    # In[6]:


model = HyperIM(50, embd_dim, n_classes, d_ball=8, hidden_size=128, if_gru=True)
model_best = HyperIM(50, embd_dim, n_classes, d_ball=8, hidden_size=128, if_gru=True)

# model = Net(n_classes, embd_dim, max_sen_length,
#             alpha1=0.01, lambda1=0.1, gamma=2, alpha2=class_freq_inverse).cuda()
# model_best = Net(n_classes, embd_dim, max_sen_length,
#                  alpha1=0.01, lambda1=0.1, gamma=2, alpha2=class_freq_inverse)  # deep copy the model with best acc

# criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# In[7]:


def evaluate(acc=True, report=False, eval_nonzero=False):
    predict_list = []
    label_list = []
    loader = val_loader if not eval_nonzero else val_nonzero_loader
    with torch.no_grad():
        for batch_idx, (X_, y, pos_) in enumerate(loader):
            X_ = X_.cuda()
            label_list.extend(y)
            y = y.cuda()
            pos_ = pos_.cuda()
            out, _ = model(X_, y, pos_)

            pred = torch.argmax(out.data, 1)
            predict_list.extend(pred.tolist())

    # label_list = y_val.tolist()
    if report:
        print(classification_report(label_list, predict_list))
    if acc:
        return accuracy_score(label_list, predict_list)

    return precision_recall_fscore_support(label_list, predict_list, average='weighted')


# def adjust_learning_rate(optimizer, epoch, args):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# In[8]:


best_val_acc = 0.0
best_epoch = 0
last_loss = 0
try:
    for epoch in range(200):
        loss_sum = 0.0
        correct = 0
        start_time = time.time()
        model.train()  # necessary if dropout is used
        for i, (X_batch, y_batch, pos) in enumerate(train_loader):
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()
            pos = pos.cuda()

            optimizer.zero_grad()
            outputs, loss = model(X_batch, y_batch, pos)
            loss_sum += loss.item()

            prediction = torch.argmax(outputs.data, 1)
            correct += (prediction == y_batch).sum().item()

            loss.backward()
            optimizer.step()
        loss_sum /= len(train_loader)
        train_acc = correct / len(train_dataset)
        model.eval()
        val_acc = evaluate(acc=True, eval_nonzero=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model_best.load_state_dict(model.state_dict())
        print("Epoch {:3d}: loss={:.4f} "
              "train_acc={:.3f} "
              "val_acc={:.3f} "
              "time={:.2f}".format(epoch + 1, loss_sum, train_acc,
                                   val_acc,
                                   time.time() - start_time))
        if abs(loss_sum - last_loss) <= 1e-4:
            print("Early stopping at epoch ", epoch + 1)
            break
        last_loss = loss_sum
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print("Best epoch {:3d} best val_acc {:.3f}".format(best_epoch, best_val_acc))
# evaluate(acc=False, report=True)

# In[9]:


test_id_list, test_en_pos_list, X_test = make_input_data(test_path, test=True)
# X_test = X_test[:100]
test_dataset = torch.utils.data.TensorDataset(X_test, test_en_pos_list)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=500)
result_list = []
model_best = model_best.cuda()

# lgmloss.to("cpu")
# model_best = model
with torch.no_grad():
    model_best.eval()
    for i, (X_batch, pos) in enumerate(test_loader):
        X_batch = X_batch.cuda()
        pos = pos.cuda()
        outputs = model_best(X_batch, None, pos)

        prediction = torch.argmax(outputs.data, 1)
        result_list.extend(prediction.tolist())

out_file_name = "./result_pcnn_a.txt"
out_path = Path(out_file_name)
if out_patorch.exists():
    print("File existed! Will rename it")
    out_patorch.rename(out_file_name + ".bak")

with open(out_file_name, 'w', encoding='utf8') as f:
    for i, label in zip(test_id_list, result_list):
        f.write("{}\t{}\r\n".format(i, label))
print("Finished at {:s}".format(time.ctime()))
