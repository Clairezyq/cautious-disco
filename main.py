import numpy as np
from numpy import *
import scipy.sparse as sp
import torch
import numpy as np
from tqdm import tqdm,trange
import pandas as pd
import random
import gc
import matplotlib.pyplot as plt

import math
from torch.nn.parameter import Parameter

from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim

####################################################################################################

def normalize(mx):
    """"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """compute L=D^-0.5 * (A+I) * D^-0.5"""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj += sp.eye(adj.shape[0])
    degree = np.array(adj.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    norm_adj = d_hat.dot(adj).dot(d_hat)
    return sparse_mx_to_torch_sparse_tensor(norm_adj)

def normalize_adj_noi(adj):
    """compute L=D^-0.5 * (A+I) * D^-0.5"""
    adj = sp.coo_matrix(adj, dtype=np.float32)
    degree = np.array(adj.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    norm_adj = d_hat.dot(adj).dot(d_hat)
    return sparse_mx_to_torch_sparse_tensor(norm_adj)
####################################################################################################
class GraphConvolution(Module):
  
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj_):
        support = torch.matmul(input, self.weight)
        output = torch.spmm(adj_, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc10 = GraphConvolution(nfeat, nhid)
        self.gc11 = GraphConvolution(nhid, nclass)
        
        self.gc20 = GraphConvolution(nfeat, nhid)
        self.gc21 = GraphConvolution(nhid, nclass)
        
        self.gc30 = GraphConvolution(nfeat, nhid)
        self.gc31 = GraphConvolution(nhid, nclass)
        
        self.dropout = dropout
        
        self.fuison_weight0 = Parameter(torch.rand(nclass, 16))
        self.fuison_weight1 = Parameter(torch.rand(nclass, 16))
        self.fuison_weight2 = Parameter(torch.rand(nclass, 16))
        
        self.layer_norm = nn.LayerNorm(normalized_shape = 16, eps = 1e-6)
        self.dropout_layer = nn.Dropout(p = dropout)

    def forward(self, x, adj_normal, adj_flow, ad_poi):
        x0 = self.gc11(F.dropout(F.relu(self.gc10(x, adj_normal)), self.dropout, training = self.training), 
                       adj_normal)
        x1 = self.gc21(F.dropout(F.relu(self.gc20(x, adj_flow)), self.dropout, training = self.training), 
                       adj_flow)
        x2 = self.gc31(F.dropout(F.relu(self.gc30(x, ad_poi)), self.dropout, training = self.training), 
                       ad_poi)

        x_out = torch.matmul(x0, self.fuison_weight0) + torch.matmul(x1, self.fuison_weight1) \
                + torch.matmul(x2, self.fuison_weight2)

        return self.dropout_layer(self.layer_norm(x_out + x))
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len = 20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    


class CopyTaskModel(nn.Module):

    def __init__(self, d_model = 16, n_heads = 4):
        super(CopyTaskModel, self).__init__()

        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_heads, dropout = 0.4),
                                             num_layers = 1,
                                             norm = nn.LayerNorm(normalized_shape = d_model, eps = 1e-6))

# 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout = 0.4)
 
        self.predictor = nn.Linear(d_model, 8)

    def forward(self, src):

        src = self.positional_encoding(src)
        out = self.encoder(src)

        return self.predictor(out)
    
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.embedding1 = nn.Embedding(num_embeddings = 8, embedding_dim = 3)
        self.embedding2 = nn.Embedding(num_embeddings = 5, embedding_dim = 3)
        self.linear0 = nn.Linear(in_features = 3, out_features = 4, bias=True)
        self.linear1 = nn.Linear(in_features = 3, out_features = 4, bias=True)
        self.linear2 = nn.Linear(in_features = 1, out_features = 8, bias=True)

    def forward(self, X_feature, X_week, X_stamp):
        
        X1 = self.embedding1(X_week)
        X1 = self.linear0(X1.reshape([97,3]))
        
        X2 = self.embedding2(X_stamp)
        X2 = self.linear1(X2.reshape([97,3]))
        
        X3 = self.linear2(X_feature.reshape([97,1]))
        
        return torch.cat((X3,X2,X1),dim = 1)
    
class GCN_encoder(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.Embedding = Embedding()
        self.model_graphlearning = GCN(nfeat, nhid, nclass, dropout)
        self.model_tencoder = CopyTaskModel()
        self.linear0 = nn.Linear(in_features = 8, out_features = 4, bias=True)
        self.linear1 = nn.Linear(in_features = 4, out_features = 1, bias=True)

    def forward(self, feature_tensor, week_tensor, stamptensor, a0, a1, a2, k):
        
        result0 = torch.stack((self.model_graphlearning(
                           self.Embedding(feature_tensor[0], week_tensor[k], stamptensor[k]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[1], week_tensor[k+1], stamptensor[k+1]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[2], week_tensor[k+2], stamptensor[k+2]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[3], week_tensor[k+3], stamptensor[k+3]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[4], week_tensor[k+4], stamptensor[k+4]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[5], week_tensor[k+5], stamptensor[k+5]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[6], week_tensor[k+6], stamptensor[k+6]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[7], week_tensor[k+7], stamptensor[k+7]), a0, a1, a2)), 
                              dim=0)
        result1 = self.model_tencoder(result0)
        result2  = self.linear1(self.linear0(result1))
        
        
        return result1, result2, result2[-1,:,:]



####################################################################################################



A_normal = np.load('A_normal.npy')
A_mobility =  np.load('mobility_graph.npy')
A_poi =  np.load('A_poi.npy')

# adj0 = normalize_adj(A_normal)
# adj1 = normalize_adj_noi(A_mobility)
# adj2 = normalize_adj(A_poi)

adj0 = sp.coo_matrix(A_normal, dtype=np.float32)
adj0 = adj0 + adj0.T.multiply(adj0.T > adj0) - adj0.multiply(adj0.T > adj0)
adj0 = sparse_mx_to_torch_sparse_tensor(normalize(adj0 + sp.eye(adj0.shape[0])))

adj1 = normalize_adj_noi(A_poi)

adj2 = sp.coo_matrix(A_poi, dtype=np.float32)
adj2 = adj2 + adj2.T.multiply(adj2.T > adj2) - adj2.multiply(adj2.T > adj2)
adj2 = sparse_mx_to_torch_sparse_tensor(normalize(adj2 + sp.eye(adj2.shape[0])))

taxi_tensor_in = np.load('taxi_tensor_in.npy')
taxi_tensor_in = (taxi_tensor_in-taxi_tensor_in.mean())/taxi_tensor_in.std()
service_tensor = np.load('service_tensor.npy')
service_tensor = (service_tensor-service_tensor.mean())/service_tensor.std()
crime_tensor = np.load('crime_tensor.npy')
crime_tensor = (crime_tensor-crime_tensor.mean())/crime_tensor.std()
crash_tensor = np.load('crash_tensor.npy')
crash_tensor = (crash_tensor-crash_tensor.mean())/crash_tensor.std()

week_index = ([3,3,3,3]+[4,4,4,4]+[5,5,5,5]+[6,6,6,6]+[7,7,7,7]+[1,1,1,1]+[2,2,2,2])*25 \
            +[3,3,3,3]+[4,4,4,4]+[5,5,5,5]+[6,6,6,6]+[7,7,7,7]+[1,1,1,1]

hour_index = ([0]+[1]+[2]+[3])*181

week_ff = np.zeros([724,97,1])

for i in tqdm(range(724)):
    week_ff[i] = week_index[i]
    

stamp_ff = np.zeros([724,97,1])
for i in tqdm(range(724)):
    stamp_ff[i] = hour_index[i]

inflow = np.zeros([716,8,97])
for i in tqdm(range(716)):
    inflow[i] = taxi_tensor_in[i:i+8]
inservice = np.zeros([716,8,97])
for i in tqdm(range(716)):
    inservice[i] = service_tensor[i:i+8]

####################################################################################################


model_forward = GCN_encoder(nfeat = 16, nhid = 32, nclass = 16, dropout = 0.4)

optimizer = optim.Adam(model_forward.parameters(), lr = 0.001, weight_decay = 5e-4)

criterion = nn.L1Loss()
def train(epoch):
    model_forward.train()
    optimizer.zero_grad()
    lossall = []
    loss_train = 0
    batch_ = random.sample(list(range(trainingdate)), 128)
    for i in batch_:
        mask0 = torch.FloatTensor(np.random.random(size=(8, 97)))
        mask1 = mask0 * (mask0 >= 0.3)
        taxi_tensor_in1 = torch.FloatTensor(inflow[i]) * (mask1 > 0)
        service_tensor1 = torch.FloatTensor(inservice[i]) * (mask1 > 0)

        _, _, output0 = model_forward(torch.FloatTensor(taxi_tensor_in1), torch.IntTensor(week_ff), 
                                      torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        _, _, output1 = model_forward(torch.FloatTensor(service_tensor1), torch.IntTensor(week_ff), 
                                      torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        
        loss = criterion(output0 * (mask0 < 0.3), torch.FloatTensor(taxi_tensor_in)[i:i+8] * (mask0 < 0.3)) \
              + criterion(output1 * (mask0 < 0.3), torch.FloatTensor(service_tensor)[i:i+8] * (mask0 < 0.3))
        loss_train = loss_train + loss
        lossall.append(float(loss.cpu()))

    loss_train.backward()
    optimizer.step()
    return lossall


path = './modelpretrain.pth'
torch.save(model_forward.state_dict(), path)


####################################################################################################
class crime_prediction(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.Embedding = Embedding()
        self.model_graphlearning = GCN(nfeat, nhid, nclass, dropout)
        self.model_tencoder = CopyTaskModel()

        for param in self.parameters():
            param.requires_grad = False

        self.prompt0 = Parameter(torch.rand(97, 16))
        self.prompt1 = Parameter(torch.rand(97, 16))
        self.prompt2 = Parameter(torch.rand(97, 16))
        self.linear0 = nn.Linear(in_features = 8, out_features = 4, bias=True)
        self.linearout0 = nn.Linear(in_features = 4, out_features = 2, bias=True)
        self.linearout1 = nn.Linear(in_features = 2, out_features = 1, bias=True)

    def forward(self, feature_tensor, week_tensor, stamptensor, a0, a1, a2, k):
        
        result0 = torch.stack((self.model_graphlearning(self.prompt0, a0, a1, a2), 
                               self.model_graphlearning(self.prompt1, a0, a1, a2),
                               self.model_graphlearning(self.prompt2, a0, a1, a2),
                        self.model_graphlearning(
                           self.Embedding(feature_tensor[k], week_tensor[k], stamptensor[k]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+1], week_tensor[k+1], stamptensor[k+1]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+2], week_tensor[k+2], stamptensor[k+2]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+3], week_tensor[k+3], stamptensor[k+3]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+4], week_tensor[k+4], stamptensor[k+4]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+5], week_tensor[k+5], stamptensor[k+5]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+6], week_tensor[k+6], stamptensor[k+6]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+7], week_tensor[k+7], stamptensor[k+7]), a0, a1, a2)), 
                              dim=0)
        
        result1 = self.model_tencoder(result0)
        result2  = self.linearout1(self.linearout0(self.linear0(result1)))
        
        
        return result2[-1,:,:]

####################################################################################################
    
class taxi_prediction(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super().__init__()
        self.Embedding = Embedding()
        self.model_graphlearning = GCN(nfeat, nhid, nclass, dropout)
        self.model_tencoder = CopyTaskModel()

        for param in self.parameters():
            param.requires_grad = False

        self.prompt0 = Parameter(torch.rand(97, 16))
        self.prompt1 = Parameter(torch.rand(97, 16))
        self.prompt2 = Parameter(torch.rand(97, 16))
        self.linear0 = nn.Linear(in_features = 8, out_features = 4, bias=True)
        self.linearout0 = nn.Linear(in_features = 4, out_features = 2, bias=True)
        self.linearout1 = nn.Linear(in_features = 2, out_features = 1, bias=True)

    def forward(self, feature_tensor, week_tensor, stamptensor, a0, a1, a2, k):
        
        result0 = torch.stack((self.model_graphlearning(self.prompt0, a0, a1, a2), 
                               self.model_graphlearning(self.prompt1, a0, a1, a2),
                               self.model_graphlearning(self.prompt2, a0, a1, a2),
                        self.model_graphlearning(
                           self.Embedding(feature_tensor[k], week_tensor[k], stamptensor[k]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+1], week_tensor[k+1], stamptensor[k+1]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+2], week_tensor[k+2], stamptensor[k+2]), a0, a1, a2), 
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+3], week_tensor[k+3], stamptensor[k+3]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+4], week_tensor[k+4], stamptensor[k+4]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+5], week_tensor[k+5], stamptensor[k+5]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+6], week_tensor[k+6], stamptensor[k+6]), a0, a1, a2),
                       self.model_graphlearning(
                           self.Embedding(feature_tensor[k+7], week_tensor[k+7], stamptensor[k+7]), a0, a1, a2)), 
                              dim=0)
        
        result1 = self.model_tencoder(result0)
        result2  = self.linearout1(self.linearout0(self.linear0(result1)))
        
        
        return result2[-1,:,:]

####################################################################################################

model_crime = crime_prediction(nfeat = 16, nhid = 32, nclass = 16, dropout = 0.4)
crime_state_dict = model_crime.state_dict()

pretrained_dict = torch.load('./modelpretrain.pth')
pretrained_dict_l = {k: v for k, v in pretrained_dict.items() if k in crime_state_dict}
for i in pretrained_dict_l:
    crime_state_dict[i] = pretrained_dict_l[i]
    
model_crime.load_state_dict(crime_state_dict)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_crime.parameters()), lr = 0.0005, weight_decay = 5e-4)
criterion = nn.L1Loss()
def train(epoch):
    model_crime.train()
    optimizer.zero_grad()
    lossall = []
    loss_train = 0
    for i in range(20):
        output0 = model_crime(torch.FloatTensor(crime_tensor), torch.IntTensor(week_ff), 
                               torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        loss = criterion(output0.reshape([97]), torch.FloatTensor(crime_tensor)[i+8])
        loss_train = loss_train + loss
        lossall.append(float(loss.cpu()))

    loss_train.backward()
    optimizer.step()

    return lossall

def test():
    model_crime.eval()
    loss_test = 0
    for i in range(20,715):
        output0 = model_crime(torch.FloatTensor(crime_tensor), torch.IntTensor(week_ff), 
                               torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        loss = criterion(output0.reshape([97]), torch.FloatTensor(crime_tensor)[i+8])
        loss_test = loss_test + loss
    loss_test = loss_test/len(range(20,715))
    return loss_test


torch.backends.cudnn.enabled = False
loss0 = []
loss1 = []
for epoch in tqdm(range(3000)):
    loss0.append(float(train(epoch)))
    loss1.append(float(test().cpu()))

####################################################################################################

model_taxi = crime_prediction(nfeat = 16, nhid = 32, nclass = 16, dropout = 0.4)
taxi_state_dict = model_taxi.state_dict()

pretrained_dict = torch.load('./modelpretrain.pth')
pretrained_dict_l = {k: v for k, v in pretrained_dict.items() if k in taxi_state_dict}
for i in pretrained_dict_l:
    taxi_state_dict[i] = pretrained_dict_l[i]
    
model_taxi.load_state_dict(taxi_state_dict)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_taxi.parameters()), lr = 0.0005, weight_decay = 5e-4)
criterion = nn.L1Loss()
def train(epoch):
    model_taxi.train()
    optimizer.zero_grad()
    lossall = []
    loss_train = 0
    for i in range(trainingdate, trainingdate+20):
        output0 = model_taxi(torch.FloatTensor(taxi_tensor_in), torch.IntTensor(week_ff), 
                               torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        loss = criterion(output0.reshape([97]), torch.FloatTensor(taxi_tensor_in)[i+8])
        loss_train = loss_train + loss
        lossall.append(float(loss.cpu()))

    loss_train.backward()
    optimizer.step()

    return lossall

def test():
    model_taxi.eval()
    loss_test = 0
    for i in range(trainingdate+20,715):
        output0 = model_taxi(torch.FloatTensor(taxi_tensor_in), torch.IntTensor(week_ff), 
                               torch.IntTensor(stamp_ff), adj0, adj1, adj2, i)
        loss = criterion(output0.reshape([97]), torch.FloatTensor(taxi_tensor_in)[i+8])
        loss_test = loss_test + loss
    loss_test = loss_test/len(range(trainingdate+20,715))
    return loss_test

torch.backends.cudnn.enabled = False
loss0 = []
loss1 = []
for epoch in tqdm(range(3000)):
    loss0.append(float(train(epoch)))
    loss1.append(float(test().cpu()))

