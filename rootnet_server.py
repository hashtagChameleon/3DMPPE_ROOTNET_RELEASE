#!/usr/bin/env python3

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import json
import pickle
import codecs
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
from http.server import BaseHTTPRequestHandler, HTTPServer

model_path = '/home/levishai_g/pose_estimation/models/snapshot_18.pth.tar'

sys.path.insert(0, 'main')
sys.path.insert(0, 'data')
sys.path.insert(0, 'common')
from config import cfg
from model import get_pose_net
from utils.pose_utils import process_bbox
from dataset import generate_patch_image

class Rootnet():
    def __init__(self, model_path = model_path, gpu_ids = '0'):
        cfg.set_args(gpu_ids)
        cudnn.benchmark = True

        assert osp.exists(model_path), 'Cannot find model at ' + model_path
        print(f'Loading Rootnet model {model_path}')
        self.model = DataParallel(get_pose_net(cfg, False)).cuda()
        ckpt = torch.load(model_path)
        self.model.load_state_dict(ckpt['network'])
        self.model.eval()
        print(f'Model loaded succesfully')

    def process_image(self, original_img, bbox_list):
        # prepare input image
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
        original_img_height, original_img_width = original_img.shape[:2]

        person_num = len(bbox_list)

        # normalized camera intrinsics
        focal = [1500, 1500] # x-axis, y-axis
        princpt = [original_img_width/2, original_img_height/2] # x-axis, y-axis
        # print('focal length: (' + str(focal[0]) + ', ' + str(focal[1]) + ')')
        # print('principal points: (' + str(princpt[0]) + ', ' + str(princpt[1]) + ')')

        # for cropped and resized human image, forward it to RootNet
        root_depth_list = []
        for n in range(person_num):
            bbox = process_bbox(np.array(bbox_list[n]), original_img_width, original_img_height)
            img, img2bb_trans = generate_patch_image(original_img, bbox, False, 0.0)
            img = transform(img).cuda()[None,:,:,:]
            k_value = np.array([math.sqrt(cfg.bbox_real[0]*cfg.bbox_real[1]*focal[0]*focal[1]/(bbox[2]*bbox[3]))]).astype(np.float32)
            k_value = torch.FloatTensor([k_value]).cuda()[None,:]

            # forward
            with torch.no_grad():
                root_3d = self.model(img, k_value) # x,y: pixel, z: root-relative depth (mm)
            img = img[0].cpu().numpy()
            root_3d = root_3d[0].cpu().numpy()

            root_depth_list.append(root_3d[2])

        return root_depth_list

rootnet = Rootnet()

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        data_string = self.rfile.read(int(self.headers['Content-Length']))
        data = json.loads(data_string)
        result = rootnet.process_image(pickle.loads(codecs.decode(data['image'].encode(), 'base64')), data['bbox_list'])
        response = { "root_depth_list": result }
        # print(f'calculated {response}')

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        js = json.dumps(response, cls=NumpyEncoder)
        self.wfile.write(js.encode(encoding='utf_8'))

if __name__ == '__main__':
    server_address = ('127.0.0.1', 3000)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    httpd.serve_forever()
