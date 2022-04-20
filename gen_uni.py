# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:01:56 2021

@author: wenlutang
"""



import numpy as np
import torch
import torch.utils.data as Data
import matplotlib.pyplot as plt
#import torch.autograd.Function as Function




def gen_univ(size=2**10,model='sine',error='expx',df=2,sigma=1):    
    x = torch.rand([size,1]).float()
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.randn([size,1]),
            'cauchy':torch.from_numpy(np.random.standard_cauchy([size,1])),
            'sinex':torch.randn(x.shape)*((torch.sin(np.pi*x))),
            'expx':torch.randn(x.shape)*(torch.exp((x-0.5)*2)),
            #'expx':torch.exp(1-x)
            }
    eps=errors[error].float()
    bsign=torch.Tensor(np.array([int(x >= 0.5) for x in x])).reshape((size,1))
    ys={#'sine':2*x*torch.sin(4*np.pi*x)+sigma*eps,
        'sine':2*torch.sin(4*np.pi*x)+sigma*eps,
         'linear':2*x+sigma*eps,
         'exp': torch.exp(2*x)+sigma*eps,
         'quad': (4-4*torch.abs(x-0.5))+sigma*eps,
         'change': 10*x+5*bsign.mul(eps)+(1-bsign).mul(eps),
         'noncon': 5*bsign.mul(x)+5*(1-bsign).mul(x-1)+sigma*eps,
        }
    y=ys[model].float()
    return Data.TensorDataset(x, y, eps)



    
def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              y_u2=None,
              y_l2=None,
              y_min=None,
              y_max=None,
              pred=None,
              pred2=None,
              max_show=None,
              shade_color="",
              shade_color2="",
              method_name="",
              method_name2="",
              title="",
              filename=None,
              save_figures=False,
              label_observations="Observations",
              label_estimate="Predicted value"):
    
    
    x_ = x[:max_show]
    y_ = y[:max_show]
    if y_u is not None:
        y_u_ = y_u[:max_show]
    if y_l is not None:
        y_l_ = y_l[:max_show]
    if y_u2 is not None:
        y_u2_ = y_u2[:max_show]
    if y_l2 is not None:
        y_l2_ = y_l2[:max_show]
    if pred is not None:
        pred_ = pred[:max_show]
    if pred2 is not None:
        pred2_ = pred2[:max_show]

 #   fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    # plt.plot(x_[inds,:], y_[inds], 'k.', alpha=.3, markersize=7,
    #          fillstyle='full', label=label_observations)
    plt.plot(x_[inds,:], y_[inds], 'm.', alpha=.2, markersize=4,
             fillstyle='full', label=label_observations)
        
    
    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x_[inds], x_[inds][::-1]]),
                 np.concatenate([y_u_[inds], y_l_[inds][::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' prediction interval')
        
    if (y_u2 is not None) and (y_l2 is not None):
        plt.fill(np.concatenate([x_[inds], x_[inds][::-1]]),
                 np.concatenate([y_u2_[inds], y_l2_[inds][::-1]]),
                 alpha=.3, fc=shade_color2, ec='None',
                 label = method_name2 + ' prediction interval')
    
    if pred is not None:
        if pred_.ndim == 2:
            plt.plot(x_[inds,:], pred_[inds,0], 'tab:blue', lw=2, alpha=0.9,
                     label=u'NC-CQR 5% and 95% quantiles')
            plt.plot(x_[inds,:], pred_[inds,1], 'tab:blue', lw=2, alpha=0.9)
        else:
            plt.plot(x_[inds,:], pred_[inds], 'k', lw=2, alpha=0.9,
                     label=label_estimate)
            
    if pred2 is not None:
        if pred2_.ndim == 2:
            plt.plot(x_[inds,:], pred2_[inds,0], 'g', lw=2, alpha=0.9,
                     label=u'QR 5% and 95% quantiles')
            plt.plot(x_[inds,:], pred2_[inds,1], 'g', lw=2, alpha=0.9)
        else:
            plt.plot(x_[inds,:], pred_[inds], 'k', lw=2, alpha=0.9,
                     label=label_estimate)
    
    #plt.ylim([-10, 10])
    plt.ylim([y_min-4, y_max+1])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='lower right',fontsize=8)
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=500)
    
    plt.show()







