# -*- coding: utf-8 -*-

import hmmlearn.hmm as hmm
import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import random


class medium_scale_hmm():
    def __init__(self, medium_scale_trips, n_components):
        self.n_components = n_components
        self.state2index = {"<eos>": 0}
        self.index2state = ["<eos>"]
        max_length = 0
        self.lengths = []
        for trip in medium_scale_trips:
            for state in trip:
                if state not in self.state2index:
                    self.state2index[state] = len(self.index2state)
                    self.index2state.append(state)
            max_length = max(max_length, len(trip))
            self.lengths.append(len(trip))

        self.medium_scale_trips = [[self.state2index[state] for state in trip] + [0] * (max_length - len(trip)) for
                                   trip in medium_scale_trips]
        self.hmm_model = None

    def fit(self, n_iter, save_file=None, use_length=False):

        start_time = datetime.datetime.now()
        print("Start Time: %s" % str(start_time))
        model2 = hmm.MultinomialHMM(n_components=self.n_components, n_iter=n_iter, tol=0.01)
        X2 = np.array(self.medium_scale_trips)

        # 是否使用length参数
        lengths = None
        if use_length:
            lengths = self.lengths
            X2 = X2.reshape((-1, 1))

        model2.fit(X=X2, lengths=lengths)
        print(model2.startprob_)
        print(model2.transmat_)
        print(model2.emissionprob_)
        print(model2.score(X=X2, lengths=lengths))
        if save_file:
            pickle.dump(model2, open(save_file, 'wb'), 2)
        end_time = datetime.datetime.now()
        self.hmm_model = model2
        print("Start Time: %s" % str(start_time))
        print("Done!")
        print("Finish Time: %s" % str(end_time))
        return

    def test_hmm_model(self, test_data):
        hmm_model = self.hmm_model
        prop_obs = self.index2state[1:]
        state2index = self.state2index
        inf_cnt = 0
        val_cnt = 0
        topk_list = [0] * len(prop_obs)
        mrr_list = []
        for test_prefix, test_res, _ in tqdm(test_data):
            test_prefix_indexs = [state2index[x] if x in state2index else 0 for x in test_prefix]
            cur_test_prop_list = []
            for obs in prop_obs:
                seen = np.array(test_prefix_indexs + [state2index[obs]])
                logprob = hmm_model.score(seen.reshape(-1, 1))
                cur_test_prop_list.append((obs, logprob))
            cur_test_prop_list.sort(key=lambda x: -x[1])
            if cur_test_prop_list[0][1] == float('-inf'):
                inf_cnt += 1
            else:
                val_cnt += 1
                for i in range(len(cur_test_prop_list)):
                    if cur_test_prop_list[i][0] == test_res:
                        topk_list[i] += 1
                        mrr_list.append(1.0 / (i + 1))
                        break
        print("inf_cnt:%s" % inf_cnt)
        print("val_cnt:%s" % val_cnt)
        for k in [1, 5, 10]:
            print("Top%d:%.3f" % (k, sum(topk_list[:k]) / val_cnt))
        print("MRR:%.3f" % (sum(mrr_list) / val_cnt))
        return


def read_db_trip_into_file(filename, movement_field="ls_route_f", train_or_test="train"):
    db_conn = pymysql.connect(host="222.29.117.240", port=2048, user="root", database="suzhou", password="19950310",
                              charset="utf8")
    db_cursor = db_conn.cursor(pymysql.cursors.DictCursor)
    select_sql = "SELECT %s FROM suzhou.sz_ls_ms_ss_action_info_to_study_2020 WHERE train_or_test = '%s';" % (
        movement_field, train_or_test)
    large_scale_trips = []
    db_cursor.execute(select_sql)
    for line in tqdm(db_cursor.fetchall()):
        one_trip = line[movement_field].split(";")
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


if __name__ == "__main__":
    read_db_trip_into_file("./mms_model/mm_scale_trips_train.json", movement_field="mms_route_f",
                          train_or_test="train")
    read_db_trip_into_file("./mms_model/mm_scale_trips_test.json", movement_field="mms_route_f",
                           train_or_test="test")
    trip_test_list = json.load(open("./mms_model/mm_scale_trips_test.json", "r"))
    hmm_test_pair = get_hmm_test_pair(trip_test_list)
    pickle.dump(hmm_test_pair, open("./mms_model/mm_scale_trips_hmm_test_pair.plk", "wb"), 2)

    hmm_test_pair = pickle.load(open("./mms_model/mm_scale_trips_hmm_test_pair.plk", "rb"))
    medium_scale_trips = json.load(open("./mms_model/mm_scale_trips_train.json", "r"))
    n_components = 25
    n_iter = 1000
    use_lengths = True
    ms_hmm_base = medium_scale_hmm(medium_scale_trips=medium_scale_trips, n_components=n_components)
    ms_hmm_base.fit(n_iter=n_iter, save_file="./mms_model/mms_hmm_comp_%s_iter_%s_len_%s.pkl" % (
    n_components, n_iter, 1 if use_lengths else 0), use_length=use_lengths)
    ms_hmm_base.test_hmm_model(hmm_test_pair)

    print("Done!")
