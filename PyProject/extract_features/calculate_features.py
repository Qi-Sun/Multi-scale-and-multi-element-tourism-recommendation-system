# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime


def get_feature_ttt():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT time FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s ORDER BY time;"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_to_study_2020 SET feature_ttt = %s WHERE aid = %s;"
    # 查询
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % cur_aid)
        weibos = action_cursor.fetchall()
        ttt = (weibos[-1]["time"] - weibos[0]["time"]).total_seconds()  / 3600
        action_cursor.execute(update_aid_sql % (ttt, cur_aid))
    return


def get_feature_ttd():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT cityname,time FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s ORDER BY time;"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_to_study_2020 SET feature_ttd = %s WHERE aid = %s;"
    # 查询
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % cur_aid)
        weibos = action_cursor.fetchall()
        weibos = [x for x in weibos if x["cityname"] == '苏州市']
        ttd = (weibos[-1]["time"] - weibos[0]["time"]).total_seconds() / 3600
        action_cursor.execute(update_aid_sql % (ttd, cur_aid))
    return

def get_feature_tvs():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT count(distinct(ls_action_unified_id)) as cnt FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE userid = %s and ls_action_unified_id < %s and cityname = '苏州市';"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_to_study_2020 SET feature_tvs = %s WHERE aid = %s;"
    # 查询
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % (cur_aid // 1000000, cur_aid % 1000000))
        tmp_res = action_cursor.fetchall()
        tvs = tmp_res[0]["cnt"]
        action_cursor.execute(update_aid_sql % (tvs, cur_aid))
    return

if __name__ == "__main__":
    # get_feature_ttt()
    # get_feature_ttd()
    get_feature_tvs()
    pass
