#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: onnxmodel.py
@time: 2021/2/2 下午5:55
@desc:
'''
import sys, os

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

import argparse
import numpy as np
from time import  time

# -*-coding: utf-8 -*-

import os, sys
# sys.path.append("/work/LIB/CPP/onnx/libonnxruntime.so")
import onnxruntime

class ONNXModel():
    def __init__(self, onnx_file):
        # self.onnx_session = onnxruntime.InferenceSession(onnx_file, providers=[ 'CUDAExecutionProvider',
        #                                                    'CPUExecutionProvider'])
        self.onnx_session = onnxruntime.InferenceSession(onnx_file)

        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image:np.ndarray):
        input_feed = {}
        for name in input_name:
            input_feed[name] = image
        return input_feed

    def get_input_feed2(self, input_name, images):
        input_feed = {}
        for name, image in zip(input_name, images):
            input_feed[name] = image
        return input_feed

    def forward(self, image:np.ndarray):
        start = time()
        input_feed = self.get_input_feed(self.input_name, image)
        for index in input_feed:
            # print("key: ", index," : ", input_feed[index].shape)
            run_time = time() - start
            print("time: {}".format(run_time))
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return scores

    def forward2(self, images):
        input_feed = self.get_input_feed2(self.input_name, images)
        for index in input_feed:
            # print("key: ", index," : ", input_feed[index].shape)
            pass
        start = time()
        scores = self.onnx_session.run(self.output_name, input_feed=input_feed)
        run_time = time() - start
        print("time: {} ms".format(run_time * 1000))
        return scores



def to_numpy(tensor):
    print(tensor.device)
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
