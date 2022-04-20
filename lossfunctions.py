# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:11:55 2021

@author: wenlutang
"""

import torch
#from torch import nn

 
class check(torch.nn.Module):
    def __init__(self,taus:0.5,reduction='mean'):
        super(check,self).__init__()
        self.reduction = reduction 
        self.tau = taus
    # def derive(self,x,y):
    #     diff = torch.sub(y,x)
    #     index = (diff>0).float()
    #     totloss = (self.tau-1)*(1-index)+self.tau*index
    #     return totloss
    def forward(self,x,y):
        #size = y.size()[0]
        diff = torch.sub(y,x)
        index = (diff>0).float()
        totloss = diff*(self.tau-1)*(1-index)+self.tau*index*diff
        if self.reduction=='mean':
            totloss=torch.mean(totloss)
        return totloss
    
class QuantileLoss(torch.nn.Module):
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, x, y):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """        
        totloss = []
        for i, q in enumerate(self.quantiles):
            #tau=self.quantiles[i]
            diff = torch.sub(y,x[:,i].reshape(len(y),1))
            index = (diff>0).float()
            totloss.append(diff*(q-1)*(1-index)+q*index*diff)
            
        index=x[:, 0]-x[:, 1]
        a=torch.zeros([250,2])
        a[:,0]=index
        loss=torch.mean(torch.sum(torch.cat(totloss, dim=1), dim=1))#+10*torch.mean(torch.pow(torch.max(a, 1)[0],2))
        
        # assert not y.requires_grad
        # assert x.size(0) == y.size(0)
        # losses = 0
        # size = y.size()[0]
        # #index =[[]]

        # for i, q in enumerate(self.quantiles):
        #     errors = y - x[:, i]
        #     losses=losses+torch.sum(torch.max((q-1) * errors, q * errors).unsqueeze(1))
            

        #index=x[:, 0]-x[:, 1]
        #a=torch.zeros([250,2])
        #a[:,0]=index
        #loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))#+torch.mean(torch.pow(torch.max(a, 1)[0],2))
        #loss=losses/size
        #print(losses,losses.shape)
        return loss
    

class AllQuantileLoss(torch.nn.Module):
    """ Pinball loss function
    """
    def __init__(self, quantiles):
        """ Initialize
        Parameters
        ----------
        quantiles : pytorch vector of quantile levels, each in the range (0,1)
        """
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """ Compute the pinball loss
        Parameters
        ----------
        preds : pytorch tensor of estimated labels (n)
        target : pytorch tensor of true labels (n)
        Returns
        -------
        loss : cost function value
        """
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []

        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
            

        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss