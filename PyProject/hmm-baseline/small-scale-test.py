# -*- coding: utf-8 -*-

import hmmlearn.hmm as hmm
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import random


def play_the_data_poi_id():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT ss_poi_ids FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020 WHERE train_or_test = 'train';"
    db_cursor.execute(select_sql)
    poiid_counter= {}
    for record in tqdm(db_cursor.fetchall()):
        poiids = record["ss_poi_ids"]
        for poiid in poiids.split(";"):
            if poiid not in poiid_counter:
                poiid_counter[poiid] = 0
            poiid_counter[poiid] += 1
    poiid_list = [(x,poiid_counter[x]) for x in poiid_counter]
    poiid_list.sort(key=lambda x:-x[1])
    print("Count:%d" % len(poiid_list))
    print(poiid_list[:10])
    return
    pass

def play_the_data_poi_type():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT ss_poi_types_6 FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020 WHERE train_or_test = 'train';"
    db_cursor.execute(select_sql)
    poi_type_counter= {}
    for record in tqdm(db_cursor.fetchall()):
        poi_types = record["ss_poi_types_6"]
        for poi_type in poi_types.split(","):
            if poi_type not in poi_type_counter:
                poi_type_counter[poi_type] = 0
        for poi_type in poi_type_counter:
            if poi_type in poi_types:
                poi_type_counter[poi_type] += 1
    print(poi_type_counter)
    return
    pass

if __name__ == "__main__":
    # play_the_data_poi_id()
    play_the_data_poi_type()
    pass