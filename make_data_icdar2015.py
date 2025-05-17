import argparse
import copy
import io
import json
import math
import os
import random
import re
from sys import platform
import cv2
import numpy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastdatasets.lmdb import LMDB,DB,load_dataset,NumpyWriter
from numpy.linalg import norm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default="ch4_training_images")
parser.add_argument('--label_dir', type=str, default="ch4_training_localization_transcription_gt")
parser.add_argument('--output_dir', type=str, default='./output_dir')
args = parser.parse_args()


def align_quadrilateral(img, pts_src):
    """
    根据四个顶点进行透视变换，得到矩形区域。
    自动计算矩形的宽高。

    :param img: 输入图像
    :param pts_src: 输入图像中的四边形顶点（顺时针或逆时针）
    :return: 透视变换后的矩形图像
    """
    # 计算宽度和高度
    pts_src = np.asarray(pts_src, dtype=np.float32)

    w1 = norm(pts_src[0] - pts_src[1])
    w2 = norm(pts_src[2] - pts_src[3])
    h1 = norm(pts_src[0] - pts_src[3])
    h2 = norm(pts_src[1] - pts_src[2])

    # 目标矩形的宽高（选择最大值）
    width = int(max(w1, w2))
    height = int(max(h1, h2))

    # 目标矩形的四个顶点
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped

def gen_one_img(image_file, label_file):
    with open(label_file, mode='r', encoding='utf-8-sig') as f:
        label_str = f.read()
    label_list = label_str.split('\n')

    boxes, texts = [], []
    for label_line in label_list:
        if not label_line:
            continue
        labels_one = label_line.split(',', 8)

        box = numpy.asarray(labels_one[:8])
        pts = []
        for x, y in zip(box[0::2], box[1::2]):
            pts.append([int(x), int(y)])

        boxes.append(pts)
        texts.append(labels_one[8])
    with open(image_file, mode='b+r') as f:
        img_bytes = f.read()


    det_label = {
        "boxes": boxes,
        "texts": texts,
    }
    det_label_bin = bytes(json.dumps(det_label, ensure_ascii='utf-8'), encoding='utf-8')

    det_data = {
        "img": np.asarray(img_bytes),
        "label": np.asarray(det_label_bin)
    }

    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR_BGR)

    rec_data_list = []
    for box, text in zip(boxes, texts):
        rec_img = align_quadrilateral(img, box)
        rec_label_bin = bytes(text, encoding='utf-8')
        rec_img_bin = cv2.imencode(".jpg", rec_img)[1].tobytes()
        rec_data = {
            "img": np.asarray(rec_img_bin),
            "label": np.asarray(rec_label_bin)
        }
        rec_data_list.append(rec_data)
    return (det_data, rec_data_list)


det_nr = 0
rec_nr = 0
def gen_img(image_dir, label_dir):
    global det_nr,rec_nr
    batch_list = []
    fs_det = NumpyWriter(os.path.join(args.output_dir, "det_ldb"), map_size=10 * 1024 * 1024 * 1024)
    fs_rec = NumpyWriter(os.path.join(args.output_dir, "rec_ldb"), map_size=2 * 1024 * 1024 * 1024)
    def produce_data_batch(data):
        global det_nr,rec_nr
        if data is not None:
            batch_list.append(data)
        if data is None or len(batch_list) >= 100:
            b_det, b_rec = [], []
            for det_data,rec_data_list in batch_list:
                b_det.append((f"input{det_nr}", det_data))
                det_nr += 1

                for rec_data in rec_data_list:
                    b_rec.append((f"input{rec_nr}", rec_data))
                    rec_nr += 1

            b_keys,b_vals = [],[]
            for k,v in b_det:
                b_keys.append(k)
                b_vals.append(v)
            fs_det.put_batch(b_keys, b_vals)

            b_keys, b_vals = [], []
            for k,v in b_rec:
                b_keys.append(k)
                b_vals.append(v)
            fs_rec.put_batch(b_keys, b_vals)

            batch_list.clear()

    label_file_list = os.listdir(label_dir)
    for label_file in label_file_list:
        result = re.match(r'gt_img_(\d+).txt', label_file)
        image_file = "img_{}.jpg".format(result.group(1))
        image_file = os.path.join(image_dir, image_file)
        label_file = os.path.join(label_dir, label_file)
        d = gen_one_img(image_file, label_file)
        produce_data_batch(d)

    produce_data_batch(None)

    print(det_nr, rec_nr)
    fs_det.get_writer.put('total_num', str(det_nr))
    fs_rec.get_writer.put('total_num', str(rec_nr))

    fs_det.close()
    fs_rec.close()


# ---------- 主程序 ----------
if __name__ == '__main__':

    # gen_img(args.image_dir, args.label_dir)

    def test_random(db_path):
        dataset = load_dataset.RandomDataset(db_path,
                                             data_key_prefix_list=('input',))

        dataset = dataset.parse_from_numpy_writer().shuffle(10)
        print(len(dataset))
        length = min(len(dataset), 10)
        for i in tqdm(range(length), total=length):
            d = dataset[i]
            label = str(d["label"], encoding='utf-8')

            if db_path.find('det') != -1:
                label = json.loads(label)
                boxes = label['boxes']
                print(d.keys())
                print(boxes)

                nparr = np.frombuffer(d["img"], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
                for box in boxes:
                    box = numpy.asarray(box)
                    pts = []
                    for x,y in zip(box[0::2],box[1::2]):
                        pts.append([int(x),int(y)])

                    pts = np.asarray(pts)
                    print(pts, pts.shape)
                    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            else:
                nparr = np.frombuffer(d["img"], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR_BGR)
                print(label)

            cv2.imshow('', img)
            cv2.waitKey(2 * 1000)
            #print(i, d)


    test_random(os.path.join(args.output_dir, "det_ldb"))
    test_random(os.path.join(args.output_dir, "rec_ldb"))