#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:50:00 2019

@author: santiago
"""
import torch
import torch.nn as nn
import torch.nn.functional as F




import torch
import torch.tensor as tensor
 
a = torch.ones((2,3))  #建立tensor
#tensor([[1., 1., 1.],
#        [1., 1., 1.]])

a1 = (a.norm(p=2,dim=0,keepdim=True))  
#tensor([[1.4142, 1.4142, 1.4142]])

a2 = (a.norm(p=2,dim=1,keepdim=True))

#tensor([[1.7321],
#        [1.7321]]) 
print(a)
print(a1)
print(a2)

print("--------------------------------------")

#torch.eq
t=torch.eq(torch.tensor([[1, 2],
                         [3, 4]]),
         torch.tensor([[1, 1], 
                       [4, 4]]))


# =============================================================================
#相同返回1，不同返回0
# tensor([[1, 0],
#         [0, 1]], dtype=torch.uint8)
# =============================================================================
print(t)
print(t.sum())#tensor(2)

print(t.sum().item())#2


#topk
x = torch.arange(1., 6.)
print(x)
#tensor([ 1.,  2.,  3.,  4.,  5.])
print(torch.topk(x, 3))

#values=tensor([5., 4., 3.]),
#indices=tensor([4, 3, 2]))

output = torch.tensor([[-5, 2],
                               [-4, -3],
                               [-1, -5],
                               [-2, -4]])


print(output.topk(1))
# =============================================================================
# values=tensor([[ 2],
#         [-3],
#         [-1],
#         [-2]]),
# indices=tensor([[1],
#         [1],
#         [0],
#         [0]]))
# =============================================================================
