# coding=utf8
# @Time    : 2025/5/4 9:23
# @Author  : tk
# @FileName: record_dataset
import numpy as np
import cv2
import math
import os
import json
import random
import traceback
from paddle.io import Dataset
from .imaug import transform, create_operators

from fastdatasets.lmdb import DB as LDB,load_dataset as LOAD_DATASET

class MyDetDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(MyDetDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.delimiter = dataset_config.get("delimiter", "\t")
        self.data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        self.need_reset = False
        self.ds_width = False

        dataset = LOAD_DATASET.RandomDataset(self.data_dir,
                                             data_key_prefix_list=('input',))


        print(len(dataset))
        self.data_len = len(dataset)
        dataset = dataset.parse_from_numpy_writer()
        if self.mode == "train" and self.do_shuffle:
            dataset = dataset.shuffle(-1)
        self.dataset = dataset
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)


    def shuffle_data_random(self):
        random.seed(self.seed)
        return


    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            idx = np.random.randint(self.__len__())
            d = self.dataset[idx]
            label = json.loads(str(d["label"], encoding='utf-8'))
            label = [{'points': box, 'transcription': text} for box, text in zip(label['boxes'], label['texts'])]
            img = d["img"].tobytes()
            data = {"img_path": '', 'filename': '', 'image': img,
                    "label": label}
            data = transform(data, load_data_ops)
            if data is None:
                continue
            if "polys" in data.keys():
                if data["polys"].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        try:
            d = self.dataset[idx]
            assert d is not None, d.keys()
            label = json.loads(str(d["label"], encoding='utf-8'))
            label = [{'points': box, 'transcription': text} for box,text in zip(label['boxes'],label['texts'])]
            img = d["img"].tobytes()
            data = {"img_path": '', 'filename': '', 'image': img,
                    "label": label}
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    idx, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return self.data_len


class MyRecDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(MyRecDataSet, self).__init__()
        self.logger = logger
        self.mode = mode.lower()

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        loader_config = config[mode]["loader"]

        self.delimiter = dataset_config.get("delimiter", "\t")
        self.data_dir = dataset_config["data_dir"]
        self.do_shuffle = loader_config["shuffle"]
        self.seed = seed
        self.need_reset = False
        self.ds_width = False

        dataset = LOAD_DATASET.RandomDataset(self.data_dir,
                                             data_key_prefix_list=('input',))

        print(len(dataset))
        self.data_len = len(dataset)
        dataset = dataset.parse_from_numpy_writer()
        if self.mode == "train" and self.do_shuffle:
            dataset = dataset.shuffle(-1)
        self.dataset = dataset
        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.ext_op_transform_idx = dataset_config.get("ext_op_transform_idx", 2)


    def shuffle_data_random(self):
        random.seed(self.seed)
        random.shuffle(self.data_lines)
        return


    def get_ext_data(self):
        ext_data_num = 0
        for op in self.ops:
            if hasattr(op, "ext_data_num"):
                ext_data_num = getattr(op, "ext_data_num")
                break
        load_data_ops = self.ops[: self.ext_op_transform_idx]
        ext_data = []

        while len(ext_data) < ext_data_num:
            idx = np.random.randint(self.__len__())
            d = self.dataset[idx]
            label = str(d["label"])
            img = d["img"].tobytes()
            data = {"img_path": '', 'filename': '', 'image': img,
                    "label": label}
            data = transform(data, load_data_ops)
            if data is None:
                continue
            if "polys" in data.keys():
                if data["polys"].shape[1] != 4:
                    continue
            ext_data.append(data)
        return ext_data

    def __getitem__(self, idx):
        try:
            d = self.dataset[idx]
            label = str(d["label"])
            img = d["img"].tobytes()
            data = {"img_path": '', 'filename': '', 'image': img,
                    "label": label}
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops)
        except:
            self.logger.error(
                "When parsing line {}, error happened with msg: {}".format(
                    idx, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (
                np.random.randint(self.__len__())
                if self.mode == "train"
                else (idx + 1) % self.__len__()
            )
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return self.data_len




class MyMultiScaleDataSet(MyRecDataSet):
    def __init__(self, config, mode, logger, seed=None):
        super(MyMultiScaleDataSet, self).__init__(config, mode, logger, seed)
        self.ds_width = config[mode]["dataset"].get("ds_width", False)
        if self.ds_width:
            raise "not support ds_width"
            self.wh_aware()

    def wh_aware(self):
        data_line_new = []
        wh_ratio = []
        for lins in self.data_lines:
            data_line_new.append(lins)
            lins = lins.decode("utf-8")
            name, label, w, h = lins.strip("\n").split(self.delimiter)
            wh_ratio.append(float(w) / float(h))

        self.data_lines = data_line_new
        self.wh_ratio = np.array(wh_ratio)
        self.wh_ratio_sort = np.argsort(self.wh_ratio)
        self.data_idx_order_list = list(range(len(self.data_lines)))

    def resize_norm_img(self, data, imgW, imgH, padding=True):
        img = data["image"]
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(
                img, (imgW, imgH), interpolation=cv2.INTER_LINEAR
            )
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")

        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((3, imgH, imgW), dtype=np.float32)
        padding_im[:, :, :resized_w] = resized_image
        valid_ratio = min(1.0, float(resized_w / imgW))
        data["image"] = padding_im
        data["valid_ratio"] = valid_ratio
        return data

    def __getitem__(self, properties):
        # properties is a tuple, contains (width, height, index)
        img_height = properties[1]
        idx = properties[2]
        if self.ds_width and properties[3] is not None:
            wh_ratio = properties[3]
            img_width = img_height * (
                1 if int(round(wh_ratio)) == 0 else int(round(wh_ratio))
            )
            file_idx = self.wh_ratio_sort[idx]
        else:
            file_idx = idx
            img_width = properties[0]
            wh_ratio = None


        try:
            d = self.dataset[file_idx]
            label = str(d["label"])
            img = d["img"].tobytes()
            data = {"img_path": '', 'filename': '', 'image': img,
                    "label": label}
            data["ext_data"] = self.get_ext_data()
            outs = transform(data, self.ops[:-1])
            if outs is not None:
                outs = self.resize_norm_img(outs, img_width, img_height)
                outs = transform(outs, self.ops[-1:])
        except:
            self.logger.error(
                "When parsing idx {}, error happened with msg: {}".format(
                    file_idx, traceback.format_exc()
                )
            )
            outs = None
        if outs is None:
            # during evaluation, we should fix the idx to get same results for many times of evaluation.
            rnd_idx = (idx + 1) % self.__len__()
            return self.__getitem__([img_width, img_height, rnd_idx, wh_ratio])
        return outs