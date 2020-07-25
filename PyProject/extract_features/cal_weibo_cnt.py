# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime


def get_feature_weibo_cnt():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT aid FROM suzhou.sz_ls_ms_ss_action_info_all_2020;"
    action_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                                  charset="utf8")
    action_cursor = action_conn.cursor(pymysql.cursors.DictCursor)
    select_weibo_sql = "SELECT count(*) as cnt FROM suzhou.travel_poi_users_weibodata_suzhou_sq_st_to_study WHERE aid = %s ORDER BY time;"
    update_aid_sql = "UPDATE suzhou.sz_ls_ms_ss_action_info_all_2020 SET weibo_cnt = %s WHERE aid = %s;"
    # 查询
    db_cursor.execute(select_aid_sql)
    for record in tqdm(db_cursor.fetchall()):
        cur_aid = record["aid"]
        action_cursor.execute(select_weibo_sql % cur_aid)
        weibos = action_cursor.fetchall()
        weibo_cnt = weibos[0]["cnt"]
        action_cursor.execute(update_aid_sql % (weibo_cnt, cur_aid))
    return

if __name__ == "__main__":
    get_feature_weibo_cnt()
