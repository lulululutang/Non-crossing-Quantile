# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:20:34 2021

@author: wenlutang
"""



#%%
from gen_uni import gen_univ, plot_func
from dnn import DNN
from lossfunctions import QuantileLoss,check

import torch
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#import imageio
from tqdm import tqdm
#import pandas as pd
import math
import random
from sklearn.linear_model import QuantileRegressor
#from sklearn.kernel_ridge import KernelRidge
#%%
# class GKR:
    
#     def __init__(self, x, y, b):
#         self.x = x
#         self.y = y
#         self.b = b
    
#     '''Implement the Gaussian Kernel'''
#     def gaussian_kernel(self, z):
#         return (1/math.sqrt(2*math.pi))*math.exp(-0.5*z**2)
    
#     '''Calculate weights and return prediction'''
#     def predict(self, X):
#         kernels = [self.gaussian_kernel((xi-X)/self.b) for xi in self.x]
#         weights = [len(self.x) * (kernel/np.sum(kernels)) for kernel in kernels]
#         return np.dot(weights, self.y)/len(self.x)
#%%
# Data Generation
torch.manual_seed(2021)
random.seed(2021)
SIZE=3000; model='noncon';
#taus=[0.3]
dataset = gen_univ(SIZE,model='noncon',error='normal',df=2,sigma=1)
x=dataset[:][0].data.numpy();y=dataset[:][1].data.numpy();
#eps=dataset[:][2].data.numpy();
ind=x.squeeze().argsort();
y_max=max(y);y_min=min(y);y_r=y_max-y_min;

# set the desired miscoverage error
alpha=0.1
rep=1
#%%
# A quick view on the data
fig, ax = plt.subplots(figsize=(10,5))
plt.cla()
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.xticks(fontproperties = 'Times New Roman', size = 14)
ax.set_title('%s'% model.title(),fontdict={'family':'Times New Roman','size':26})
ax.set_xlabel(r'$X$',fontdict={'size':16})
ax.set_ylabel(r'$Y$',fontdict={'size':16})
ax.scatter(x[ind], y[ind], color = "k")
#%%
# Data splitting
#Divide the data into proper training set and calibration set
length=[]
cover=[]
length2=[]
cover2=[]
for k in range(rep):
    random.seed(k+2021)
#random.seed(2021)
    n_train= 2000
    idx = np.random.permutation(SIZE)
    n_half = int(np.floor(n_train/2))
    idx_train, idx_cal, idx_test = idx[:n_half], idx[n_half:2*n_half], idx[2*n_half:SIZE]
        
    data_train = torch.utils.data.Subset(dataset, idx_train)
    data_cal = torch.utils.data.Subset(dataset, idx_cal)
    data_test = torch.utils.data.Subset(dataset, idx_test)
        
    y_cal = y[idx_cal] 
    x_cal = x[idx_cal] 
    x_test = x[idx_test]
    y_test = y[idx_test]
    x_train = x[idx_train]
    y_train = y[idx_train]
        
    alpha_lo=0.05
    alpha_hi=0.95
        #%%
    # Prepare dataloader
    # Prepare dataloader
    BATCH_SIZE = int(n_half/4)
    EPOCH =1000
    loader = Data.DataLoader(
    dataset=data_train, 
    batch_size=BATCH_SIZE, 
    shuffle=False, num_workers=0,)
        #%%
        # Create DNN
    net1 = DNN(width_vec=[1,256,256,256])     # define the network
    #net2 = DNN(width_vec=[1,256,256,256]) 
    #print(net1)  # net architecture
    optimizer1 = torch.optim.Adam(net1.parameters(), lr=0.001)
    #optimizer2 = torch.optim.Adam(net2.parameters(), lr=0.001)
        #loss_ls = torch.nn.MSELoss(reduction='mean')  
        #loss_L1 = torch.nn.L1Loss(reduction='mean')   
        #loss_huber = torch.nn.SmoothL1Loss(reduction='mean') 
        #loss_cauchy = CauchyLoss(k=1,reduction='mean')
        #loss_tukey = TukeyLoss(t=4.685,reduction='mean')
    loss_check = QuantileLoss(quantiles=[alpha_lo,alpha_hi]) 
    #loss_check = check(taus=alpha_lo,reduction='mean') 
        # Tarin the DNN
    my_images = []
    losses=[]
    #losses2=[]
    pre_loss=[]
    #pre_loss2=[]
    
    
    #%%
    for t in tqdm(range(EPOCH),total=EPOCH):
            for step, (x_batch, y_batch,_) in enumerate(loader):
                net1.train()
                x_batch, y_batch = Variable(x_batch.float()), Variable(y_batch.float())
                prediction = net1(x_batch)     # input x and predict based on x
                loss = loss_check(prediction, y_batch)     # must be (1. nn output, 2. target)
                losses.append(loss.data.numpy())
                optimizer1.zero_grad()   # clear gradients for next train
                loss.backward()         # backpropagation, compute gradients
                optimizer1.step()        # apply gradients
                #net2.train()
                #prediction2 = net2(x_batch)     # input x and predict based on x
                #loss2 = loss_check2(prediction2, y_batch)     # must be (1. nn output, 2. target)
                #losses2.append(loss2.data.numpy())
                #optimizer2.zero_grad()   # clear gradients for next train
                #loss2.backward()         # backpropagation, compute gradients
                #optimizer2.step()        # apply gradients
                if step == 0:
                    pre_loss.append(loss.data.numpy())
                    #pre_loss2.append(loss2.data.numpy())
        
    net1.eval()
    #net2.eval()
    
    
        #%%
    prediction_hi=net1(data_cal[:][0].data).data.numpy()[:,1].reshape(len(y_cal),1) # Evaluation
    prediction_lo=net1(data_cal[:][0].data).data.numpy()[:,0].reshape(len(y_cal),1)
        #for i in range(len(loss_list)):
        #bayes_risks[j] = loss_L1(torch.abs(eps),torch.zeros(eps.size()))
        #cal_loss = np.abs(y_cal-prediction_cal.data.numpy())
    cal_loss=np.maximum(prediction_lo-y_cal, y_cal-prediction_hi)
    k = math.ceil(((n_train/2.0)+1)*(1-alpha))
    sorted_cal_loss = sorted(cal_loss)
    q = sorted_cal_loss[k-1]
        #sorted_bayes_loss =sorted(eps[idx_cal])
        #find the test preidtion
        #prediction_train=net(data_train[:][0].data)
      #  prediction_test=net(data_test[:][0].data)
        #find the confidence interval of prediction
    y_u=net1(data_test[:][0]).data.numpy()[:,1].reshape(len(y_cal),1)
    y_l=net1(data_test[:][0]).data.numpy()[:,0].reshape(len(y_cal),1)
    y_upper=y_u+q
    y_lower=y_l-q
    #    plot_func(x = range(50),y=y_train,y_u=y_upper,y_l=y_lower, y_min=y_min, y_max=y_max,  pred=prediction_cal.data.numpy(), max_show=50, shade_color='blue',
    #    method_name="CI_NN:",title="",
    #    filename="illustration_split_NNCI_t.png",save_figures=False)
    length_CI= np.mean(y_upper-y_lower)
        # compute and display the average coverage
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    cover_CI= in_the_range / len(y_test)
    length.append(length_CI)
    cover.append(cover_CI)
        #%%
        #train Bspline on sin function
    quantiles=[alpha_lo, alpha_hi]
    qr1=QuantileRegressor(quantile=alpha_lo,alpha=0)
    qr2=QuantileRegressor(quantile=alpha_hi,alpha=0)
    q_l=qr1.fit(x_train,  y_train.squeeze(-1))
    #q_l=qr1.fit(x_train,  y_train)
    q_h=qr2.fit(x_train,  y_train.squeeze(-1))
    pred_lo=q_l.predict(x_cal).reshape(len(x_cal),1) 
    pred_hi=q_h.predict(x_cal).reshape(len(x_cal),1) 
    # Evaluation
    cal_loss2 = np.maximum(pred_lo-y_cal, y_cal-pred_hi)
    k2 = math.ceil(((n_train/2.0)+1)*(1-alpha))
    sorted_cal_loss2 = sorted(cal_loss2)
    q2 = sorted_cal_loss2[k2-1]
    #find the test preidtion
    y_u1=q_h.predict(x_test).reshape(len(x_test),1) 
    y_l1=q_l.predict(x_test).reshape(len(x_test),1) 
    #find the confidence interval of prediction
    y_upper2=y_u1+q2
    y_lower2=y_l1-q2
    length_CI2=np.mean(y_upper2-y_lower2)
    # compute and display the average coverage
    in_the_range2 = np.sum((y_test >= y_lower2) & (y_test <= y_upper2))
    cover_CI2= in_the_range2 / len(y_test)
    length2.append(length_CI2)
    cover2.append(cover_CI2)
#%%
print(length_CI)
print(cover_CI)
print(np.std(cover))
print(np.std(length))
print(np.sum((prediction_hi-prediction_lo>0)))
print(np.mean(cover2))
print(np.mean(length2))
print(np.std(cover2))
print(np.std(length2))
# print(np.mean(cover2))
# print(np.mean(length2))
# print(np.std(cover2))
# print(np.std(length2))
#plt.scatter(range(len(pre_loss)),pre_loss)
np.savetxt('result_cubic', (length,cover, length2, cover2))
#np.savetxt('result2_cubic', (length2,cover2))
# plot_func(x = x_test,y=y_test,y_u=y_upper,y_l=y_lower, y_min=y_min+2, y_max=y_max-2,  pred=np.array([np.squeeze(y_l),np.squeeze(y_u)]).T, max_show=399, shade_color='blue',
#   method_name="NCCI_NN:",title="Non-crossing quantile interval of univariate setting (4)",
#   filename="illustration_split_NNCI.png",save_figures=True,
#   label_observations="")
# # plot_func(x = x_test,y=y_test,y_u=y_upper2,y_l=y_lower2, y_min=-10, y_max=10,  pred=prediction_test2, max_show=399, shade_color='blue',
# #   method_name="Kernel regression:",title="Kernel regression interval of univariate setting (1)",
# #   filename="illustration_split_qr.png",save_figures=True,
# #   label_observations="")
# plot_func(x = x_test,y=y_test,y_u=y_upper2,y_l=y_lower2, y_min=y_min+2, y_max=y_max-2,  pred=np.array([np.squeeze(y_l1),np.squeeze(y_u1)]).T, max_show=399, shade_color='blue',
#   method_name="Quantile regression:",title="Quantile regression interval of univariate setting (4)",
#   filename="illustration_split_qr.png",save_figures=True,
#   label_observations="")
# plot_func(x = x_test,y=y_test,y_u=y_upper2,y_l=y_lower2, y_min=y_min, y_max=y_max,  pred=net1(data_test[:][0]).data.numpy(), max_show=399, shade_color='blue',
#   method_name="Linear_Regression interval:",title="interval of linear regression",
#   filename="illustration_split_linear.png",save_figures=False,
#   label_observations="")

plot_func(x = x_test,y=y_test,y_u=y_upper,y_l=y_lower, y_u2=y_upper2,y_l2=y_lower2, y_min=y_min, y_max=y_max,  pred=net1(data_test[:][0]).data.numpy(), pred2=np.array([np.squeeze(y_l1),np.squeeze(y_u1)]).T, max_show=600, shade_color='blue',
  shade_color2="green", method_name="NC-CQR interval",method_name2="QR interval",title="Comparison of QR and NC-CQR interval",
  filename="illustration_split_sin.png",save_figures=False,
  label_observations="")


