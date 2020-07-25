# -*- coding: utf-8 -*-
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt


def get_features():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT feature_dcr_f,feature_ttt_f,feature_ttd_f,feature_tvs_f FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020;"
    db_cursor.execute(select_aid_sql)
    f_dcr = []
    f_ttt = []
    f_ttd = []
    f_tvs = []
    features = []
    for record in db_cursor.fetchall():
        features.append(
            [record["feature_dcr_f"], record["feature_ttt_f"], record["feature_ttd_f"], record["feature_tvs"]])
    return features

def get_raw_features():
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_aid_sql = "SELECT feature_dcr,feature_ttt,feature_ttd,feature_tvs FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020;"
    db_cursor.execute(select_aid_sql)
    f_dcr = []
    f_ttt = []
    f_ttd = []
    f_tvs = []
    features = []
    for record in db_cursor.fetchall():
        features.append(
            [record["feature_dcr"], record["feature_ttt"], record["feature_ttd"], record["feature_tvs"]])
    return features

if __name__ == "__main__":
    # features = get_features()
    # pickle.dump(features,open("./tourist_features.pkl",'wb'))
    # tourist_features = pickle.load(open("./tourist_features.pkl", 'rb'))
    # tourist_features = np.array(tourist_features)
    # plt.boxplot([tourist_features[:, 0], tourist_features[:, 1], tourist_features[:, 2], tourist_features[:, 3]],
    #             labels=["dcr", 'ttt', 'ttd', 'tvs'])
    # plt.boxplot(tourist_features)
    # plt.show()
    features = get_raw_features()
    pickle.dump(features,open("./tourist_raw_features.pkl",'wb'))

    pass
