# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime


def get_place_type():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT ms_place_id,poi_type FROM suzhou.travel_poi_checkin_weibos_suzhou;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    place_type_cnt = {}
    db_cursor.execute(select_weibo_sql)
    for record in tqdm(db_cursor.fetchall()):
        place_id = record["ms_place_id"]
        place_type = record["poi_type"]
        if place_id and place_type:
            if place_id not in place_type_cnt:
                place_type_cnt[place_id] = {"交通": 0, "住宿": 0, "吸引物": 0, "娱乐": 0, "购物": 0, "饮食": 0}
            if place_type in place_type_cnt[place_id]:
                place_type_cnt[place_id][place_type] += 1
    place_type_sort = {}
    for place_id in tqdm(place_type_cnt):
        place_type_sort[place_id] = sorted([(k,place_type_cnt[place_id][k]) for k in place_type_cnt[place_id]],
                                           key=lambda x: -x[1])
    place_type_sum = {"交通": 0, "住宿": 0, "吸引物": 0, "娱乐": 0, "购物": 0, "饮食": 0}
    empty_cnt = 0
    place_dict_type = {}
    for place_id in tqdm(place_type_sort):
        if place_type_sort[place_id][0][1] != 0:
            cur_type = place_type_sort[place_id][0][0]
            place_type_sum[cur_type]  += 1
            place_dict_type[place_id] = cur_type
        else:
            empty_cnt += 1
    print(empty_cnt)
    print(place_type_sum)
    return place_dict_type

if __name__ == "__main__":
    place_type_dict = get_place_type()
    json.dump(place_type_dict,open("./place_type_dict.json",'w'))
