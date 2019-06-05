#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 13:25:52 2019

@author: santiago
"""
import onnx

import torch
import torchvision


import os
from CNNArchitecture import Net

#save another format model
dummy_input = torch.randn(20,3,128,64)
# net definition
net = Net(get_features=True,num_classes=751)##mars 625 ,market1501 751

assert os.path.isfile("./checkpoint/ckpt.pytorch"), "Error: no checkpoint file found!"
print('Loading from checkpoint/ckpt.pytorch')
checkpoint = torch.load("./checkpoint/ckpt.pytorch",map_location='cpu')
net_dict = checkpoint['net_dict']
net.load_state_dict(net_dict)
net.train(False)
input_names="Placeholder"
output_names="div"

#torch.onnx.export(net, dummy_input, "a.onnx", verbose=True,training=False)#input_names=input_names)#, output_names=output_names)


torch.onnx.export(net, dummy_input, "a.onnx", verbose=True,training=False,input_names=input_names,export_params=True)


model = onnx.load("a.onnx") # Check that the IR is well formed 
onnx.checker.check_model(model) # Print a human readable representation of the graph 
onnx.helper.printable_graph(model.graph)
