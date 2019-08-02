#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import torch as th
import torch.nn as nn
import torch.nn.functional as F
# import torchtext as thtext
import geoopt as gt

import numpy as np
import scipy.sparse
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# In[2]:


default_dtype = th.float64
th.set_default_dtype(default_dtype)

cuda_device = th.device('cuda:3')
th.cuda.set_device(device=cuda_device)

base_path = '/data/blchen/text/CCKS2019-IPRE/'
save_path = os.path.join(base_path, 'preprocessed/sent')
result_path = os.path.join(base_path, 'result')
test_path = os.path.join(base_path, 'sent_relation_test.txt')

# ## Load data

# In[3]:


X_train = np.load(os.path.join(save_path, 'X_train.npy'))
X_dev = np.load(os.path.join(save_path, 'X_dev.npy'))

y_train = scipy.sparse.load_npz(os.path.join(save_path, 'y_train.npz'))
y_dev = scipy.sparse.load_npz(os.path.join(save_path, 'y_dev.npz'))

# In[4]:


X_train = th.LongTensor(X_train)
X_dev = th.LongTensor(X_dev)

y_train = th.DoubleTensor(y_train.todense())
y_dev = th.DoubleTensor(y_dev.todense())

# In[5]:


train_dataset = th.utils.data.TensorDataset(X_train, y_train)
train_data_loader = th.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

dev_dataset = th.utils.data.TensorDataset(X_dev, y_dev)
dev_data_loader = th.utils.data.DataLoader(dev_dataset, batch_size=4)

for X_train_batch, y_train_batch in train_data_loader:
    print('X_train shape', X_train_batch.shape, 'y_train shape', y_train_batch.shape)
    break
print('train_batch_num', len(train_data_loader))
for X_dev_batch, y_dev_batch in dev_data_loader:
    print('X_dev shape', X_dev_batch.shape, 'y_dev shape', y_dev_batch.shape)
    break
print('dev_batch_num', len(dev_data_loader))
print('train/dev split', len(X_train) / (len(X_train) + len(X_dev)))

# ## Network

# In[6]:


PROJ_EPS = 1e-5
EPS = 1e-15


def project_hyp_vec(x):
    # To make sure hyperbolic embeddings are inside the unit ball.
    norm = th.sum(x ** 2, dim=-1, keepdim=True)

    return x * (1. - PROJ_EPS) / th.clamp(norm, 1. - PROJ_EPS)


def asinh(x):
    return th.log(x + (x ** 2 + 1) ** 0.5)


def acosh(x):
    return th.log(x + (x ** 2 - 1) ** 0.5)


def atanh(x):
    return 0.5 * th.log((1 + x) / (1 - x))


def poinc_dist(u, v):
    m = mob_add(-u, v) + EPS
    atanh_x = th.norm(m, dim=-1, keepdim=True)
    dist_poincare = 2.0 * atanh(atanh_x)
    return dist_poincare


def euclid_dist(u, v):
    return th.norm(u - v, dim=-1, keepdim=True)


def mob_add(u, v):
    v = v + EPS

    norm_uv = 2 * th.sum(u * v, dim=-1, keepdim=True)
    norm_u = th.sum(u ** 2, dim=-1, keepdim=True)
    norm_v = th.sum(v ** 2, dim=-1, keepdim=True)

    denominator = 1 + norm_uv + norm_v * norm_u
    result = (1 + norm_uv + norm_v) / denominator * u + (1 - norm_u) / denominator * v

    return project_hyp_vec(result)


def mob_scalar_mul(r, v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    nomin = th.tanh(r * atanh(norm_v))
    result = nomin / norm_v * v

    return project_hyp_vec(result)


def mob_mat_mul(M, x):
    x = project_hyp_vec(x)
    Mx = x.matmul(M)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def mob_mat_mul_d(M, x, d_ball):
    x = project_hyp_vec(x)
    Mx = x.view(x.shape[0], -1).matmul(M.view(M.shape[0] * d_ball, M.shape[0] * d_ball)).view(x.shape)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)

    return project_hyp_vec(th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx)


def lambda_x(x):
    return 2. / (1 - th.sum(x ** 2, dim=-1, keepdim=True))


def exp_map_x(x, v):
    v = v + EPS
    second_term = th.tanh(lambda_x(x) * th.norm(v) / 2) / th.norm(v) * v
    return mob_add(x, second_term)


def log_map_x(x, y):
    diff = mob_add(-x, y) + EPS
    return 2. / lambda_x(x) * atanh(th.norm(diff, dim=-1, keepdim=True)) / th.norm(diff, dim=-1, keepdim=True) * diff


def exp_map_zero(v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    result = th.tanh(norm_v) / norm_v * v

    return project_hyp_vec(result)


def log_map_zero(y):
    diff = project_hyp_vec(y + EPS)
    norm_diff = th.norm(diff, dim=-1, keepdim=True)
    return atanh(norm_diff) / norm_diff * diff


def mob_pointwise_prod(x, u):
    # x is hyperbolic, u is Euclidean
    x = project_hyp_vec(x + EPS)
    Mx = x * u
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x, dim=-1, keepdim=True)

    result = th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx
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
        return th.zeros((batch_size, hidden_size, d_ball), dtype=default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_rnn_state(inputs.shape[0], self.hidden_size)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.transition(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)


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
        z = th.sigmoid(log_map_zero(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(log_map_zero(r))

        r_point_h = mob_pointwise_prod(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)
        # h_tilde = th.tanh(log_map_zero(h_tilde)) # non-linearity

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
        return th.zeros((batch_size, hidden_size, self.d_ball), dtype=default_dtype, device=cuda_device)

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)


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


# In[7]:


word_embed = th.randn(270734, 128, 8)
label_embed = th.randn(35, 128, 8)

# In[8]:


net = HyperIM(50, word_embed, label_embed, d_ball=8, hidden_size=128, if_gru=True)
net = net.to(cuda_device)

loss = nn.BCEWithLogitsLoss()
optim = th.optim.Adam(net.parameters(), lr=1e-4)

# In[9]:


net.load_state_dict(th.load('/data/blchen/text/CCKS2019-IPRE/net/sent/hyper-2.pt',
                            map_location=lambda storage, loc: storage.cuda(3)))


# ## Evaluation

# In[ ]:


def precision_k(pred, label, k=[1, 3, 5]):
    batch_size = pred.shape[0]

    precision = []
    for _k in k:
        p = 0
        for i in range(batch_size):
            p += label[i, pred[i, :_k]].mean().item()
        precision.append(p * 100 / batch_size)

    return precision


def evaluate(result):
    p1, p3 = 0, 0

    with th.no_grad():
        for batch_idx, (X_batch, y_batch) in enumerate(dev_data_loader):
            _batch_size = X_batch.shape[0]
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

            output = net(X_batch)
            pred = output.topk(k=5)[1]

            _p1, _p3 = precision_k(pred, y_batch, k=[1, 3])
            p1 += _p1
            p3 += _p3

    batch_idx += 1
    p1 /= batch_idx
    p3 /= batch_idx

    result[-1].append([p1, p3])

    return result


# def evaluate(result):
#     f, p, r = 0, 0, 0

#     with th.no_grad():
#         for batch_idx, (X_batch, y_batch) in enumerate(dev_data_loader):

#             _batch_size = X_batch.shape[0]
#             X_batch = X_batch.cuda()
#             y_batch = y_batch.numpy()

#             output = net(X_batch).topk(k=5)[1].cpu().numpy()
#             pred = np.zeros(y_batch.shape)
#             pred[output] = 1

#             _f = f1_score(y_batch, pred, average='micro')
#             _p = precision_score(y_batch, pred, average='micro')
#             _r = recall_score(y_batch, pred, average='micro')
#             f += _f
#             p += _p
#             r += _r


#     batch_idx += 1
#     f /= batch_idx
#     p /= batch_idx
#     r /= batch_idx

#     result[-1].append([f, p, r])

#     return result


# ## Train

# In[ ]:


result = []

# In[ ]:


for e in tqdm(range(3, 11)):
    for batch_idx, (X_batch, y_batch) in enumerate(train_data_loader):
        X_batch = X_batch.cuda()
        y_batch = y_batch.cuda()

        optim.zero_grad()
        l = loss(net(X_batch), y_batch)
        l.backward()
        optim.step()

    result.append(['epoch', e])
    result = evaluate(result)

    th.save(net.state_dict(), '/data/blchen/text/CCKS2019-IPRE/net/sent/hyper-' + str(e) + '.pt')

# In[ ]:


result

# ## Prediction

# In[ ]:


X_test = np.load(os.path.join(save_path, 'X_test.npy'))
X_test = th.LongTensor(X_test)

test_data_loader = th.utils.data.DataLoader(X_test, batch_size=4)

for X_test_batch in test_data_loader:
    print('X_train shape', X_test_batch.shape)
    break
print('train_batch_num', len(test_data_loader))

# In[ ]:


result = []

with th.no_grad():
    for batch_idx, X_batch in enumerate(test_data_loader):

        _batch_size = X_batch.shape[0]
        X_batch = X_batch.cuda()

        output = net(X_batch)
        pred = output.topk(k=1)[1]
        for p in pred.tolist():
            result.append(p[0])

# In[ ]:


for r in result:
    if (r != 0):
        print(r)

# In[ ]:


with open(test_path, 'r') as f:
    with open(os.path.join(result_path, 'result-hyper-2.txt'), 'w') as fw:
        cnt = 0
        for line in f:
            fw.write(line.strip() + '\t' + str(result[cnt]) + '\n')
            cnt += 1

# In[ ]:
