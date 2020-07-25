# -*- coding: utf-8 -*-

import pymysql
import json
import pickle
from tqdm import tqdm
import random


def read_db_trip_into_file(filename, movement_field="ls_route_f", train_or_test="train"):
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT * FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020 WHERE train_or_test = '%s';" % (
        train_or_test)
    large_scale_trips = []
    db_cursor.execute(select_sql)
    for line in tqdm(db_cursor.fetchall()):
        one_trip = line[movement_field].split(";")
        if movement_field == "ls_route_f":
            one_trip = one_trip[1:-1]
        features = [line["feature_dcr_f"], line["feature_ttt_f"], line["feature_ttd_f"], line["feature_tvs_f"]]
        large_scale_trips.append([features, one_trip])
    json.dump(large_scale_trips, open(filename, 'w'))
    return large_scale_trips


def get_hmm_test_pair(trip_test_list):
    hmm_test_pair = []
    for features, trip in trip_test_list:
        trip_length = len(trip)
        select_index = random.randint(1, trip_length - 1)
        hmm_test_pair.append([features, (trip[:select_index], trip[select_index], trip)])
    return hmm_test_pair


if __name__ == "__main__":
    # 大尺度
    ls_train_set = read_db_trip_into_file("../data/ls_train_feature_trips.json", movement_field="ls_route_f",
                                          train_or_test="train")
    ls_test_set = read_db_trip_into_file("../data/ls_test_feature_trips.json", movement_field="ls_route_f",
                                         train_or_test="test")
    ls_test_pair = get_hmm_test_pair(ls_test_set)
    json.dump(ls_test_pair, open("../data/ls_test_feature_trips_pair.json", 'w'))
    # 中尺度，shp中的节点
    mms_train_set = read_db_trip_into_file("../data/mms_train_feature_trips.json", movement_field="mms_route_f",
                                           train_or_test="train")
    mms_test_set = read_db_trip_into_file("../data/mms_test_feature_trips.json", movement_field="mms_route_f",
                                          train_or_test="test")
    mms_test_pair = get_hmm_test_pair(mms_test_set)
    json.dump(mms_test_pair, open("../data/mms_test_feature_trips_pair.json", 'w'))
    # 中小尺度，景区节点
    ms_train_set = read_db_trip_into_file("../data/ms_train_feature_trips.json", movement_field="ms_route_f",
                                          train_or_test="train")
    ms_test_set = read_db_trip_into_file("../data/ms_test_feature_trips.json", movement_field="ms_route_f",
                                         train_or_test="test")
    ms_test_pair = get_hmm_test_pair(ms_test_set)
    json.dump(ms_test_pair, open("../data/ms_test_feature_trips_pair.json", 'w'))
