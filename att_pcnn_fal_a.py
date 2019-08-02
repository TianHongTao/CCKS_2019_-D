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

# from tqdm import tqdm_notebook as tqdm


# In[2]:


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


model = Net(n_classes, embd_dim, max_sen_length,
            alpha1=0.01, lambda1=0.1, gamma=2, alpha2=class_freq_inverse).cuda()

model_best = Net(n_classes, embd_dim, max_sen_length,
                 alpha1=0.01, lambda1=0.1, gamma=2, alpha2=class_freq_inverse)  # deep copy the model with best acc

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
if out_path.exists():
    print("File existed! Will rename it")
    out_path.rename(out_file_name + ".bak")

with open(out_file_name, 'w', encoding='utf8') as f:
    for i, label in zip(test_id_list, result_list):
        f.write("{}\t{}\r\n".format(i, label))
print("Finished at {:s}".format(time.ctime()))
