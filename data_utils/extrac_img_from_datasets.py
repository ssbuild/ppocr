# coding=utf8
# @Time    : 2025/5/18 14:05
# @Author  : tk
# @FileName: extrac_img_from_datasets
import json
import os.path

import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastdatasets.lmdb import LMDB,DB,load_dataset,NumpyWriter
from numpy.linalg import norm
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default="")
parser.add_argument('--output_dir', type=str, default='./output_img')
parser.add_argument('--max', type=int, default=50)
args = parser.parse_args()


def extract_img(db_path, out_path):
    dataset = load_dataset.RandomDataset(db_path,
                                         data_key_prefix_list=('input',))

    index = 0
    dataset = dataset.parse_from_numpy_writer()
    print(len(dataset))
    length = min(len(dataset), args.max)
    for i in tqdm(range(length), total=length):
        d = dataset[i]
        # nparr = np.frombuffer(d["img"], np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)

        index += 1
        with open(f"{out_path}/img_{index}.jpg", mode='wb') as f:
            f.write(d["img"].tobytes())



def extract_det(db_path, out_path):
    dataset = load_dataset.RandomDataset(db_path,
                                         data_key_prefix_list=('input',))

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    length = min(len(dataset), 10)
    for i in tqdm(range(length), total=length):
        d = dataset[i]
        label = str(d["label"], encoding='utf-8')

        label = json.loads(label)
        boxes = label['boxes']
        # print(d.keys())
        # print(boxes)

        nparr = np.frombuffer(d["img"], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
        # for box in boxes:
        #     box = numpy.asarray(box)
        #     pts = []
        #     for x, y in zip(box[0::2], box[1::2]):
        #         pts.append([int(x), int(y)])
        #
        #     pts = np.asarray(pts)
        #     print(pts, pts.shape)
        #     cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)


        cv2.imshow('', img)
        cv2.waitKey(2 * 1000)
        # print(i, d)


def extract_rec(db_path, out_path):
    dataset = load_dataset.RandomDataset(db_path,
                                         data_key_prefix_list=('input',))

    dataset = dataset.parse_from_numpy_writer().shuffle(10)
    print(len(dataset))
    length = min(len(dataset), 10)
    for i in tqdm(range(length), total=length):
        d = dataset[i]
        label = str(d["label"], encoding='utf-8')

        nparr = np.frombuffer(d["img"], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
        print(label)


if __name__ == "__main__":
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    extract_img(args.dataset_dir, args.output_dir)