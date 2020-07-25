# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np


def get_medium_scale_movement_old():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ms_action_d3_st_to_study_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT sz_spot FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s ORDER BY time;"
    update_aid_sql = "UPDATE suzhou.sz_ms_action_d3_st_to_study_2020 SET route = '%s' WHERE aid = %s;"
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % cur_aid)
        spot_list = []
        for line in action_cursor.fetchall():
            if line["sz_spot"]:
                spot_list.append(line["sz_spot"])
        action_cursor.execute(update_aid_sql % (';'.join(spot_list), cur_aid))
    return


def get_medium_scale_movement():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ls_ms_ss_action_info_all_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT sz_ms_tourism_place FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s ORDER BY time;"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_all_2020 SET mms_route = '%s', mms_route_f = '%s', mms_place_count = %s WHERE aid = %s;"
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % cur_aid)
        spot_list = []
        spot_f_list = []
        for line in action_cursor.fetchall():
            spot_list.append(line["sz_ms_tourism_place"] if line["sz_ms_tourism_place"] else "NULL")
            if line["sz_ms_tourism_place"]:
                if not spot_f_list or spot_f_list[-1] != line["sz_ms_tourism_place"]:
                    spot_f_list.append(line["sz_ms_tourism_place"])
        spot_list_str = ';'.join(spot_list)
        spot_f_list_str = ';'.join(spot_f_list)
        if len(spot_list_str) >= 512:
            spot_list_str = "###"
        if len(spot_f_list_str) >= 512:
            spot_f_list_str = "###"
        action_cursor.execute(update_aid_sql % (spot_list_str, spot_f_list_str, len(spot_f_list), cur_aid))
    return


def get_medium_scale_movement_detials():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid,route FROM suzhou.sz_ms_action_d3_st_to_study_2020;"
    update_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    update_cursor = update_conn.cursor(pymysql.cursors.DictCursor)
    update_sql = "UPDATE suzhou.sz_ms_action_d3_st_to_study_2020 SET sz_spot_unique_count = %s , route_f = '%s' WHERE aid = %s ;"
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        unique_spot_list = []
        spot_list = record["route"].split(";")
        for spot in spot_list:
            if not unique_spot_list or unique_spot_list[-1] != spot:
                unique_spot_list.append(spot)
        update_cursor.execute(update_sql % (str(len(unique_spot_list)), ';'.join(unique_spot_list), cur_aid))
    return


if __name__ == "__main__":
    get_medium_scale_movement()
