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

class DetDataSet(Dataset):
    def __init__(self, config, mode, logger, seed=None):
        super(DetDataSet, self).__init__()
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
