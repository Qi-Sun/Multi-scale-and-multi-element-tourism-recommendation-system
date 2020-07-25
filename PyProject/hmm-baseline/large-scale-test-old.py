# -*- coding: utf-8 -*-

import hmmlearn.hmm as hmm
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import random


class large_scale_hmm():
    def __init__(self, large_scale_trips, n_components):
        self.n_components = n_components
        self.state2index = {"<eos>": 0}
        self.index2state = ["<eos>"]
        max_length = 0
        for trip in large_scale_trips:
            for state in trip:
                if state not in self.state2index:
                    self.state2index[state] = len(self.index2state)
                    self.index2state.append(state)
            max_length = max(max_length, len(trip))

        self.large_scale_trips = [[self.state2index[state] for state in trip] + [0] * (max_length - len(trip)) for
                                  trip in large_scale_trips]
        self.hmm_model = None

    def fit(self, n_iter, save_file=None):
        start_time = datetime.datetime.now()
        print("Start Time: %s" % str(start_time))
        model2 = hmm.MultinomialHMM(n_components=self.n_components, n_iter=n_iter, tol=0.01)
        X2 = np.array(self.large_scale_trips)
        model2.fit(X2)
        print(model2.startprob_)
        print(model2.transmat_)
        print(model2.emissionprob_)
        print(model2.score(X2))
        if save_file:
            pickle.dump(model2, open(save_file, 'wb'), 2)
        end_time = datetime.datetime.now()
        self.hmm_model = model2
        print("Done!")
        print("Finish Time: %s" % str(end_time))
        return

    def test_one_trip(self):
        pass


def read_db_trip_into_file(filename, train_or_test="train"):
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT ls_route_f FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020 WHERE train_or_test = '%s';" % train_or_test
    large_scale_trips = []
    db_cursor.execute(select_sql)
    for line in tqdm(db_cursor.fetchall()):
        one_trip = line["ls_route_f"].split(";")[1:-1]
        large_scale_trips.append(one_trip)
    json.dump(large_scale_trips, open(filename, 'w'))

    return


def get_hmm_test_pair(trip_test_list):
    hmm_test_pair = []
    for trip in trip_test_list:
        trip_length = len(trip)
        select_index = random.randint(1, trip_length - 1)
        hmm_test_pair.append((trip[:select_index], trip[select_index], trip))
    return hmm_test_pair


def test_hmm_model(model: large_scale_hmm, test_data):
    hmm_model = model.hmm_model
    prop_obs = model.index2state[1:]
    state2index = model.state2index
    test_res = []
    inf_cnt = 0
    val_cnt = 0
    topk_list = [0] * len(prop_obs)
    mrr_list = []
    for test_prefix, test_res, _ in tqdm(test_data):
        test_prefix_indexs = [state2index[x] if x in state2index else 0 for x in test_prefix]
        cur_test_prop_list = []
        for obs in prop_obs:
            seen = np.array(test_prefix_indexs + [state2index[obs]])
            logprob, state_list = hmm_model.decode(seen.reshape(-1, 1), algorithm='viterbi')
            cur_test_prop_list.append((obs, logprob))
        cur_test_prop_list.sort(key=lambda x: -x[1])
        if cur_test_prop_list[0][1] == float('-inf'):
            inf_cnt += 1
        else:
            val_cnt += 1
            for i in range(len(cur_test_prop_list)):
                if cur_test_prop_list[i][0] == test_res:
                    topk_list[i] += 1
                    mrr_list.append(1.0 / (i+1))
                    break
    print("inf_cnt:%s" % inf_cnt)
    for k in [1,5,10]:
        print("Top%d:%.3f" % (k, sum(topk_list[:k])/ val_cnt ) )
    print("MRR:%.3f" % (sum(mrr_list) / val_cnt))
    return


if __name__ == "__main__":
    # 测试数据 划分
    # hmm_test_pair = get_hmm_test_pair(json.load(open("./large_scale_trips_test.json",'r')))
    # pickle.dump(hmm_test_pair,open("./large_scale_trips_hmm_test_pair.plk",'wb'),2)

    # 测试阶段
    # lshmm = pickle.load(open("./large_scale_hmm_comp_50_iter_500.plk", 'rb'))
    # hmm_test_pair = pickle.load(open("./large_scale_trips_hmm_test_pair.plk", 'rb'))
    # test_hmm_model(lshmm, hmm_test_pair)
    pass

if __name__ == "__main__":
    # 读取ls数据
    # read_db_trip_into_file("./large_scale_trips_train.json")
    # hmm 拟合
    # large_scale_trips = json.load(open("./large_scale_trips_train.json", 'r'))
    # lshmm = large_scale_hmm(large_scale_trips)
    # lshmm.fit(100,save_file="./large_scale_hmm_n_100.plk")
    # read_db_trip_into_file("./large_scale_trips_test.json",'test')
    # 测试数据 划分
    # hmm_test_pair = get_hmm_test_pair(json.load(open("./large_scale_trips_test.json",'r')))
    # pickle.dump(hmm_test_pair,open("./large_scale_trips_hmm_test_pair.plk",'wb'),2)
    # 最新拟合
    large_scale_trips = json.load(open("./large_scale_trips_train.json", 'r'))
    lshmm = large_scale_hmm(large_scale_trips,n_components=50)
    lshmm.fit(n_iter=1000)
    pickle.dump(lshmm,open("./large_scale_hmm_comp_50_iter_1k.plk",'wb'),2)
    print("Done")
    # 测试
    hmm_test_pair = pickle.load(open("./large_scale_trips_hmm_test_pair.plk", 'rb'))
    test_hmm_model(lshmm, hmm_test_pair)
    pass
