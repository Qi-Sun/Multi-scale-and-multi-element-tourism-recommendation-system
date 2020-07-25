# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime

if __name__ == "__main__":
    place_type_dict = json.load(open("./place_type_dict.json", 'r'))
    place_predict = {}
    with open("../lstm-dense-clean/place_lstm_dense_model_nobos_0421a/place_show_res.txt", 'r') as rf:
        for line in rf.readlines():
            res_line = line.split(',')
            place_id = res_line[1]
            place_rank = int(res_line[2]) + 1
            place_type = place_type_dict[place_id]
            if place_type not in place_predict:
                place_predict[place_type] = {'cnt': 0, 'top1': 0, 'top5': 0, 'top10': 10, 'mrr': 0}
            place_predict[place_type]['cnt'] += 1
            if place_rank <= 1:
                place_predict[place_type]['top1'] += 1
            if place_rank <= 5:
                place_predict[place_type]['top5'] += 1
            if place_rank <= 10:
                place_predict[place_type]['top10'] += 1
            place_predict[place_type]['mrr'] += (1.0 / place_rank)
    for key in place_predict:
        print("Type:%s Cnt:%d Top1:%.3f Top5:%.3f Top10:%.3f MRR:%.3f" % (
            key, place_predict[key]['cnt'],
            place_predict[key]['top1'] / place_predict[key]['cnt'],
            place_predict[key]['top5'] / place_predict[key]['cnt'],
            place_predict[key]['top10'] / place_predict[key]['cnt'],
            place_predict[key]['mrr'] / place_predict[key]['cnt']))
