# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np


def get_small_scale_movement():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ss_action_d3_st_to_study_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_checkin_weibo_sql = "SELECT annotation_place_poiid,annotation_place_title FROM suzhou.travel_poi_checkin_weibos_suzhou WHERE id in (SELECT id FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s) ORDER BY id;"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_all_2020 SET ss_poi_count = %s , ss_poi_titles = '%s', ss_poi_ids = '%s' WHERE aid = %s;"
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_checkin_weibo_sql % cur_aid)
        poi_title_list = []
        poi_id_list = []
        for line in action_cursor.fetchall():
            if line["annotation_place_poiid"]:
                if not poi_id_list or poi_id_list[-1] != line["annotation_place_poiid"]:
                    poi_id_list.append(line["annotation_place_poiid"])
                    poi_title_list.append(line["annotation_place_title"].replace("'", "の"))
        poi_titles = ';'.join(poi_title_list) if len(';'.join(poi_title_list)) < 512 else "###"
        poi_ids = ';'.join(poi_id_list) if len(';'.join(poi_id_list)) < 2048 else "###"
        action_cursor.execute(update_aid_sql % (str(len(poi_id_list)), poi_titles, poi_ids, cur_aid))
    return


def get_poi_type():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT poiid,type FROM pois_suzhou_new WHERE type is not null and type not in ('机构','产业');"
    poi_type_dict = {}
    db_cursor.execute(select_sql)
    for record in tqdm(db_cursor.fetchall()):
        poi_type_dict[record["poiid"]] = record["type"]
    return poi_type_dict


def get_poi_list_type(poi_type_dict):
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    update_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    update_cursor = update_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT aid,ss_poi_ids from suzhou.sz_ls_ms_ss_action_info_all_2020;"
    update_sql = "update suzhou.sz_ls_ms_ss_action_info_all_2020 set ss_poi_types = '%s', ss_poi_types_6 = '%s', ss_poi_count_6 = %s where aid = %s;"
    db_cursor.execute(select_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        cur_poiids = record["ss_poi_ids"].split(";")
        poi_type_list = []
        poi_type6_list = []
        for poi in cur_poiids:
            if poi in poi_type_dict:
                poi_type6_list.append(poi_type_dict[poi])
                poi_type_list.append(poi_type_dict[poi])
            else:
                poi_type_list.append("NULL")
        update_cursor.execute(update_sql % (';'.join(poi_type_list),','.join(poi_type6_list),str(len(poi_type6_list)),cur_aid) )
    return


if __name__ == '__main__':
    # 提取小尺度
    get_small_scale_movement()
    # 获取poi对应的要素类型
    poi_type_dict = get_poi_type()
    json.dump(poi_type_dict, open("./poi_type_dict.json", "w"))
    # 将poi类型写入数据库
    poi_type_dict = json.load(open("./poi_type_dict.json", "r"))
    get_poi_list_type(poi_type_dict)
    pass
