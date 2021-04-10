# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

pytorch version 2 
"""

from __future__ import print_function
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F
import glob
import time
import csv 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

# 提取数据  get data
csvx_list = glob.glob('D:/c1/c1/*.csv')

A = []
for i in range( len(csvx_list)):
    # print(csvx_list[1])
    # filename='csvx_list[i]'
    filename = csvx_list[i]
    with open(filename,'r') as f:
        row  = csv.reader(f)
        a = []

        j = 0
        for r in row :
            j += 1

            if j > 500 * 200:
                break

            if not j % 200:
                a.append([float(i) for i in r])
                # np.array(A)

    A.append(a)

A = np.array(A)
A = A.reshape(A.shape[0], A.shape[-1], -1)


pass
# 导入真实磨损值
df = pd.read_csv(r"D:\c1\c1_wear.csv")

# X = np.expand_dims(A, axis=2)#增加一维轴
X=A

Y = df.values[:, 1]
# 划分训练集，测试集

X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)
X_train=torch.from_numpy(X_train).float()
Y_train=torch.from_numpy(Y_train).float()
X_test=torch.from_numpy(X_test).float()
Y_test=torch.from_numpy(Y_test).float()
# 设置 批大小 batch_size
BATCH_SIZE = 10
# 利用 pytorch 框架下的 Dataloader 创建数据集
dataset_train = Data.TensorDataset(X_train,Y_train)
loader_train = Data.DataLoader(
    dataset = dataset_train,
    batch_size = BATCH_SIZE,
    shuffle = True,
    num_workers = 0
)
dataset_test = Data.TensorDataset(X_test,Y_test)
loader_test = Data.DataLoader(
    dataset = dataset_test,
    batch_size = 10,
    shuffle = False,
    num_workers = 0
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(7,16,kernel_size=3)
        self.relu1 = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv1d(16,16,kernel_size=3)
        self.relu2 = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3)
        self.conv3 = nn.Conv1d(16,64,3)
        self.relu3 = nn.ReLU(inplace = True)
        self.conv4 = nn.Conv1d(64,64,3)
        self.relu4 = nn.ReLU(inplace = True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3)
        self.conv5 = nn.Conv1d(64,128,3)
        self.relu5 = nn.ReLU(inplace = True)
        self.conv6 = nn.Conv1d(128,128,3)
        self.relu6 = nn.ReLU(inplace = True)
        self.maxpool3 = nn.MaxPool1d(kernel_size=3)
        self.conv7 = nn.Conv1d(128,64,3)
        self.relu7 = nn.ReLU(inplace = True)
        self.conv8 = nn.Conv1d(64,64,3)
        self.relu8 = nn.ReLU(inplace = True)
        self.maxpool4 = nn.MaxPool1d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256,1)
        #self.logsoftmax = nn.LogSoftmax()
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        #print('conv1:',x.size())
        x = self.conv2(x)
        x = self.relu2(x)
        #print('conv2:',x.size())
        x = self.maxpool1(x)
        #print('maxpool1:',x.size())
        x = self.conv3(x)
        x = self.relu3(x)
        #print('conv3:',x.size())
        x = self.conv4(x)
        x = self.relu4(x)
        #print('conv4:',x.size())
        x = self.maxpool2(x)
        #print('maxpool2:',x.size())
        x = self.conv5(x)
        x = self.relu5(x)
        #print('conv5:',x.size())
        x = self.conv6(x)
        x = self.relu6(x)
        #print('conv5:',x.size())
        x = self.maxpool3(x)
        #print('maxpool3:',x.size())
        x = self.conv7(x)
        x = self.relu7(x)
        #print('conv7:',x.size())
        x = self.conv8(x)
        x = self.relu8(x)
        #print('conv8:',x.size())
        x = self.maxpool4(x)
        #print('maxpool4:',x.size())
        x = self.flatten(x)
        #print('flatten:',x.size())
        x = self.dense(x)
        #x = self.logsoftmax(x)
        
        return x

def sunshi(X,Y):
    loss = criterion(X[0],Y[0])
    return loss

model = CNN()

import torch.optim as optim
criterion = torch.nn.MSELoss(reduction='mean')
#criterion = nn.SmoothL1Loss(reduction='mean')
#criterion = torch.nn.L1Loss(reduction='mean')
    
optimizer = optim.Adam(model.parameters(),lr=0.001)
iterations = 2000
loss_tend = []
y_predict_store = []
for i in range(iterations):
    print('*'*60)
    print('the epoch :',i+1)
    y_pred_store = []
    tmd = []
    loss_sum = 0
    start = time.time()
    loss_list = []
    model.train()
    for k,(X_train1,Y_train1) in enumerate(loader_train):
        y_pred = []
        y_pred = model(X_train1)
        loss = sunshi(y_pred,Y_train1)
        print('the result of y_pred and Y_train1: ',y_pred,Y_train1)
        for p in range(len(y_pred)):
            temp = y_pred[p].item()

            y_pred_store.append(temp)
        loss1 = loss.item()
        loss_list.append(loss1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    tmd = np.array(loss_list)
    print(np.size(tmd))
    loss_mean = np.sum(tmd)/252
    loss_tend.append(loss_mean)
    print('loss_sum:',loss_mean)
    print('the time of this epoch :',end-start)
print('print y_pred_store: ',y_pred_store)
print('print loss_tend:',loss_tend)
"""
    重新编写测试集
"""
for i,(X_test1,Y_test1) in enumerate(loader_test):
    model.eval()
    predict = model(X_test1)
    loss = criterion(predict,Y_test1)
    for w in range(len(predict)):
        temp_test = predict[w].item() 
        y_predict_store.append(temp_test)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(y_predict_store)

"""
   绘制新的曲线
"""
# 绘制损失曲线
plt.figure(1)
x_label = range(1,iterations+1)
plt.title('the figure of loss of res2net test of using batch')
plt.plot(x_label,loss_tend,color='red')
# 绘制 实际预测对照图
plt.figure(2)
x_label1 = range(1,len(Y_train)+1)
plt.title('the wear of real and predicted in train set')
plt.plot(x_label1,y_pred_store,color='green')
plt.plot(x_label1,Y_train,color='red')
"""
   重新绘制测试集
"""
plt.figure(3)
x_label2 = range(1,len(Y_test)+1)
plt.title('the wear of predict in test set')
plt.plot(x_label2,Y_test,color = 'blue')
plt.plot(x_label2,y_predict_store,color='green')