# -*- coding: utf-8 -*-

import pymysql
import json
from tqdm import tqdm
import numpy as np
import datetime
import pickle
import random
import torch
from typing import List, Dict
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.nn.utils import clip_grad_norm
import os
import datetime
from torch.nn import init
import sys
import copy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class Corpus_dense(object):
    class Dictionary(object):
        def __init__(self):
            self.word2index = {}
            self.index2word = {}
            self.increment_index = 0
            # for word in ['<pad>', '<unk>', '<bos>', '<eos>']:
            #     self.add_word(word)

        def add_word(self, word):
            if word not in self.word2index:
                self.word2index[word] = self.increment_index
                self.index2word[self.increment_index] = word
                self.increment_index += 1

        def __len__(self):
            return len(self.word2index)

    def __init__(self):
        self.dictionary = self.Dictionary()
        self.train_corpus_list = None
        self.dev_corpus_list = None

    def load_data(self, another_corpus):
        self.dictionary = another_corpus.dictionary
        self.train_corpus_list = another_corpus.train_corpus_list
        self.dev_corpus_list = another_corpus.dev_corpus_list
        self.raw_data = another_corpus.raw_data
        self.dev_sample_count = another_corpus.dev_sample_count
        self.train_sample_count = another_corpus.train_sample_count
        self.train_sample_lengths, another_corpus.train_samples = another_corpus.train_sample_lengths, another_corpus.train_samples
        self.dev_sample_lengths, another_corpus.dev_samples = another_corpus.dev_sample_lengths, another_corpus.dev_samples
        return self

    # get word representation
    def get_train_or_dev_packed_seq(self, subset_sentences_list, test_stage=False):
        # 使用packed_pad_seq
        corpus_vector = []
        lengths = []
        tourist_feature = []
        for feature, sentence in subset_sentences_list:
            # if test_stage:
            #     sentence = ['<bos>'] + sentence
            # else:
            #     sentence = ['<bos>'] + sentence + ['<eos>']

            corpus_vector.append(torch.LongTensor([self.dictionary.word2index[x] for x in sentence]))
            if not test_stage:
                lengths.append(len(sentence) - 1)
            else:
                lengths.append(len(sentence))
            tourist_feature.append(feature)
        corpus_vector = torch.nn.utils.rnn.pad_sequence(corpus_vector, batch_first=True, padding_value=0)
        # corpus_Tensor = torch.nn.utils.rnn.pack_padded_sequence(input=corpus_vector, lengths=lengths,batch_first=True, enforce_sorted=False)
        return lengths, corpus_vector, tourist_feature

    def read_corpus_from_list(self, sentences_list):
        # init
        self.dictionary = self.Dictionary()
        self.raw_data = sentences_list
        # get unique words
        for tourist_feature, sentence in sentences_list:
            for word in sentence:
                self.dictionary.add_word(word)
        # 打乱顺序
        random.shuffle(sentences_list)
        # 划分训练集和验证集个数
        self.dev_sample_count = len(sentences_list) // 5
        self.train_sample_count = len(sentences_list) - self.dev_sample_count
        # 将训练集和验证集都打包
        self.train_sample_lengths, self.train_samples, self.train_tourist_features = self.get_train_or_dev_packed_seq(
            sentences_list[:self.train_sample_count], test_stage=False)
        self.dev_sample_lengths, self.dev_samples, self.dev_tourist_features = self.get_train_or_dev_packed_seq(
            sentences_list[self.train_sample_count:], test_stage=False)


class LSModel(torch.nn.Module):

    def __init__(self, corpus: Corpus_dense, network_framework: Dict):
        super().__init__()

        self.corpus = corpus

        # 网络结构
        self.num_epochs = 20 if "num_epochs" not in network_framework else network_framework["num_epochs"]
        self.batch_size = 64 if "batch_size" not in network_framework else network_framework["batch_size"]
        self.embed_size = 16 if "embed_size" not in network_framework else network_framework["embed_size"]
        self.hidden_size = 16 if "hidden_size" not in network_framework else network_framework["hidden_size"]
        self.num_lstm_layers = 1 if "num_lstm_layers" not in network_framework else network_framework["num_lstm_layers"]

        # 保存
        self.save_path = "./ls_model/" if "save_path" not in network_framework else network_framework["save_path"]
        self.save_model_file = "lsmodel_epoch_%s_embed_%s_hidden_%s.pkl"

        # 网络
        self.layer_embed = torch.nn.Embedding(num_embeddings=len(self.corpus.dictionary),
                                              embedding_dim=self.embed_size)
        self.layer_lstm = torch.nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, batch_first=True,
                                        num_layers=self.num_lstm_layers)
        self.layer_dense_predict = torch.nn.Linear(in_features=self.hidden_size,
                                                   out_features=len(self.corpus.dictionary))

    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes,
                                                 sorted_indices=packed_sequence.sorted_indices,
                                                 unsorted_indices=packed_sequence.unsorted_indices)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.simple_elementwise_apply(self.layer_embed, x)
        # x = self.layer_embed(x)

        # Forward propagate LSTM
        packed_out, (h, c) = self.layer_lstm(x, h)
        out, length_info = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Reshape output to (batch_size * sequence_length, hidden_size)
        # out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out_nopad = torch.cat([out[i][:length_info[i], :] for i in range(length_info.size()[0])], dim=0)
        # Decode hidden states of all time steps
        out_linear = self.layer_dense_predict(out_nopad)
        return out_linear, (h, c)


def train_stage(network_framework: Dict):
    # 分割线
    print('-' * 77)
    print(network_framework)
    # 语料
    corpus = None
    corpus_path = network_framework["corpus_path"]
    if not os.path.exists(corpus_path) or network_framework["create_corpus"]:
        corpus = Corpus_dense()
        medium_scale_trips = json.load(open(network_framework["train_data_path"], 'r'))
        corpus.read_corpus_from_list(medium_scale_trips)
        pickle.dump(corpus, open(corpus_path, "wb"))
    else:
        corpus = pickle.load(open(corpus_path, 'rb'))

    # 构建网络
    lstmLM = LSModel(corpus=corpus, network_framework=network_framework)
    for name, parameters in lstmLM.named_parameters():
        print(name, ':', parameters.size())
    print("LSTMLM have {} paramerters in total".format(sum(x.numel() for x in lstmLM.parameters())))

    # 定义损失函数和优化器
    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstmLM.parameters(), lr=network_framework["learning_rate"],
                           weight_decay=network_framework["weight_decay"])

    # 定义函数：截断反向传播
    def detach(states):
        return [state.detach() for state in states]

    # 保存路径
    if not os.path.exists(lstmLM.save_path):
        os.mkdir(lstmLM.save_path)

    # 计算设备
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print(device)
    lstmLM = lstmLM.to(device)

    # 验证集上结果比较
    dev_loss_nbest = []
    checkpoint_file_dict = {}

    # 参数声明
    num_lstm_layers = network_framework["num_lstm_layers"]
    embed_size = network_framework["embed_size"]
    batch_size = network_framework["batch_size"]
    hidden_size = network_framework["hidden_size"]
    num_epochs = network_framework["num_epochs"]
    learning_rate = network_framework["learning_rate"]
    train_sample_count = corpus.train_sample_count
    dev_sample_count = corpus.dev_sample_count
    num_batches = train_sample_count // batch_size
    report_frequency = network_framework["report_frequency"]
    num_best_models = network_framework["best_model_count"]
    report_epochs = network_framework["report_epochs"]
    # 训练和验证数据
    train_corpus_lengths, train_corpus_tensor, train_tourist_features = corpus.train_sample_lengths, corpus.train_samples, corpus.train_tourist_features
    dev_corpus_lengths, dev_corpus_tensor, dev_tourist_features = corpus.dev_sample_lengths, corpus.dev_samples, corpus.dev_tourist_features

    # 准备训练
    start_time = datetime.datetime.now()

    for epoch in range(network_framework["num_epochs"]):
        # 初始化隐状态和细胞状态
        states = (torch.zeros(num_lstm_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_lstm_layers, batch_size, hidden_size).to(device))

        # 迭代
        for i in range(0, train_sample_count - batch_size + 1, batch_size):
            # Get mini-batch inputs and targets
            inputs = train_corpus_tensor[i:i + batch_size, :-1]
            cur_batch_lengths = torch.LongTensor(train_corpus_lengths[i:i + batch_size])
            targets = train_corpus_tensor[i:i + batch_size, 1:]
            input_tourist_feature = train_tourist_features[i:i + batch_size]

            # Forward pass
            states = detach(states)
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=inputs.to(device),
                                                                    lengths=cur_batch_lengths.to(device),
                                                                    batch_first=True,
                                                                    enforce_sorted=False)
            outputs, states = lstmLM(packed_inputs, states)
            targets_nopad = torch.cat([targets[k][:cur_batch_lengths[k]] for k in range(len(cur_batch_lengths))],
                                      dim=0).to(device)
            loss = criterion(outputs, targets_nopad.reshape(-1))

            # Backward and optimize
            lstmLM.zero_grad()
            loss.backward()
            clip_grad_norm(lstmLM.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // batch_size
            if step % report_frequency == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

        # dev test
        with torch.no_grad():
            total_dev_loss = 0
            for j in range(0, len(dev_corpus_lengths), batch_size):
                end_index = min(j + batch_size, len(dev_corpus_lengths))
                dev_states = (torch.zeros(num_lstm_layers, end_index - j, hidden_size).to(device),
                              torch.zeros(num_lstm_layers, end_index - j, hidden_size).to(device))
                dev_inputs = dev_corpus_tensor[j:end_index, :-1]
                cur_dev_batch_lengths = torch.LongTensor(dev_corpus_lengths[j:end_index])
                dev_input_tourist_feature = dev_tourist_features[j:end_index]
                dev_targets = dev_corpus_tensor[j:end_index, 1:]
                # Forward pass
                dev_states = detach(dev_states)
                dev_packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=dev_inputs.to(device),
                                                                            lengths=cur_dev_batch_lengths.to(device),
                                                                            batch_first=True,
                                                                            enforce_sorted=False)
                dev_outputs, dev_states = lstmLM(dev_packed_inputs, dev_states)
                dev_targets_nopad = torch.cat(
                    [dev_targets[k][:cur_dev_batch_lengths[k]] for k in range(len(cur_dev_batch_lengths))],
                    dim=0).to(device)
                dev_loss = criterion(dev_outputs, dev_targets_nopad.reshape(-1))
                total_dev_loss += dev_loss.item() * (end_index - j)
            total_dev_loss = total_dev_loss / len(dev_corpus_lengths)
            if len(dev_loss_nbest) < num_best_models:
                dev_loss_nbest.append(total_dev_loss)
            elif dev_loss_nbest[-1] > total_dev_loss:
                old_dev_loss = dev_loss_nbest.pop(-1)
                os.remove(checkpoint_file_dict[old_dev_loss])
                dev_loss_nbest.append(total_dev_loss)
            else:
                pass
            if dev_loss_nbest[-1] == total_dev_loss:  # this means that a better model was founded
                # save model
                cp_file_name = os.path.join(lstmLM.save_path,
                                            'ms_model_lstm_{}_bs_{}_lr_{}_embed_{}_hidden_{}_epoch_{}.mdl'.format(
                                                num_lstm_layers,
                                                batch_size,
                                                int(learning_rate * 1e5),
                                                embed_size,
                                                hidden_size,
                                                epoch + 1
                                            ))
                checkpoint_file_dict[total_dev_loss] = cp_file_name
                torch.save(lstmLM.state_dict(), cp_file_name)
                print('Better Dev Loss: {:.4f}. Saved in {}'.format(total_dev_loss, cp_file_name))
            dev_loss_nbest.sort()
        if epoch + 1 in report_epochs:
            yield [checkpoint_file_dict[best_loss] for best_loss in dev_loss_nbest]
        print("Epoch {} Done".format(epoch + 1))
    print("Start at : %s" % start_time)
    print("All Done! : %s" % datetime.datetime.now())
    print("Model Info:")
    for best_loss in dev_loss_nbest:
        print("Filename : %s Loss : %.3f " % (checkpoint_file_dict[best_loss], best_loss))
    yield [checkpoint_file_dict[best_loss] for best_loss in dev_loss_nbest]


def test_stage(model_file: str, network_framework: Dict):
    # 语料
    corpus_path = network_framework["corpus_path"]
    corpus = pickle.load(open(corpus_path, 'rb'))
    # 构建网络
    lstmLM = LSModel(corpus=corpus, network_framework=network_framework)
    lstmLM.load_state_dict(torch.load(model_file))
    # 测试数据
    test_pair_list = pickle.load(open(network_framework["test_data_path"], "rb"))

    # 测试单条数据
    def get_one_sample_res(sentence: List, target: str):
        if any([x not in corpus.dictionary.word2index for x in sentence + [target]]):
            return None

        num_lstm_layers = network_framework["num_lstm_layers"]
        hidden_size = network_framework["hidden_size"]
        test_states = (torch.zeros(num_lstm_layers, 1, hidden_size),
                       torch.zeros(num_lstm_layers, 1, hidden_size))
        predict_score = []

        test_input_length, test_input = corpus.get_train_or_dev_packed_seq([sentence], test_stage=True)
        with torch.no_grad():
            test_input_length = torch.LongTensor(test_input_length)
            test_packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=test_input,
                                                                         lengths=test_input_length,
                                                                         batch_first=True,
                                                                         enforce_sorted=False)
            test_outputs, test_states = lstmLM(test_packed_inputs, test_states)
            predict_score = [(i, test_outputs[-1][i]) for i in range(len(test_outputs[-1]))]
        predict_score.sort(key=lambda x: -x[1])
        return [x[0] for x in predict_score]

    inf_cnt = 0
    val_cnt = 0
    topk_list = [0] * len(corpus.dictionary)
    mrr_list = []
    for test_prefix, test_res, _ in test_pair_list:
        predict_score = get_one_sample_res(test_prefix, test_res)
        if not predict_score:
            inf_cnt += 1
        else:
            val_cnt += 1
            for i in range(len(predict_score)):
                if predict_score[i] == corpus.dictionary.word2index[test_res]:
                    topk_list[i] += 1
                    mrr_list.append(1.0 / (i + 1))
                    break
    print("模型：%s" % model_file)
    print("inf_cnt:%s" % inf_cnt)
    print("val_cnt:%s" % val_cnt)
    for k in [1, 5, 10]:
        print("Top%d:%.3f" % (k, sum(topk_list[:k]) / val_cnt))
    print("MRR:%.3f" % (sum(mrr_list) / val_cnt))
    return {"MRR": sum(mrr_list) / val_cnt, "Top1": sum(topk_list[:1]) / val_cnt,
            "Top5": sum(topk_list[:5]) / val_cnt, "Top10": sum(topk_list[:10]) / val_cnt,
            "inf_cnt": inf_cnt, "val_cnt": val_cnt}


def test_stage_batch(model_file: str, network_framework: Dict):
    # 语料
    corpus_path = network_framework["corpus_path"]
    corpus = pickle.load(open(corpus_path, 'rb'))
    # 构建网络
    lstmLM = LSModel(corpus=corpus, network_framework=network_framework)
    lstmLM.load_state_dict(torch.load(model_file))
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    lstmLM = lstmLM.to(device)
    # 测试数据
    # test_pair_list = pickle.load(open(network_framework["test_data_path"], "rb"))
    test_pair_list = json.load(open(network_framework["test_data_path"], "r"))

    # 测试多条数据
    def get_one_batch_sample_res(sentences_and_tragets: List):
        sample_to_test = []
        sample_target = []
        for test_tourist_feature, test_trips in sentences_and_tragets:
            sentence, target, _ = test_trips
            if any([x not in corpus.dictionary.word2index for x in sentence + [target]]):
                pass
            else:
                sample_to_test.append([test_tourist_feature, sentence])
                sample_target.append(target)

        num_lstm_layers = network_framework["num_lstm_layers"]
        hidden_size = network_framework["hidden_size"]
        test_states = (torch.zeros(num_lstm_layers, len(sample_to_test), hidden_size).to(device),
                       torch.zeros(num_lstm_layers, len(sample_to_test), hidden_size).to(device))
        predict_score_list = []

        test_input_length, test_input, test_feature = corpus.get_train_or_dev_packed_seq(
            sample_to_test, test_stage=True)
        with torch.no_grad():
            test_input_length = torch.LongTensor(test_input_length).to(device)
            test_packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=test_input.to(device),
                                                                         lengths=test_input_length.to(device),
                                                                         batch_first=True,
                                                                         enforce_sorted=False)
            test_outputs, test_states = lstmLM(test_packed_inputs, test_states)
            for index, last_word_index in enumerate(torch.cumsum(test_input_length, dim=0)):
                predict_score = [(i, test_outputs[last_word_index.item() - 1][i]) for i in range(len(test_outputs[-1]))]
                predict_score.sort(key=lambda x: -x[1])
                predict_score_list.append(
                    ([x[0] for x in predict_score], corpus.dictionary.word2index[sample_target[index]]))
        return predict_score_list

    inf_cnt = 0
    val_cnt = 0
    topk_list = [0] * len(corpus.dictionary)
    mrr_list = []
    for j in range(0, len(test_pair_list), 256):
        end_index = min(j + 256, len(test_pair_list))
        predict_score_list_res = get_one_batch_sample_res(test_pair_list[j:end_index])
        val_cnt += len(predict_score_list_res)
        inf_cnt += (end_index - j - len(predict_score_list_res))
        for predict_score, target_index in predict_score_list_res:
            target_rank = predict_score.index(target_index)
            topk_list[target_rank] += 1
            mrr_list.append(1.0 / (target_rank + 1))
    print("模型：%s" % model_file)
    print("inf_cnt:%s" % inf_cnt)
    print("val_cnt:%s" % val_cnt)
    for k in [1, 5, 10]:
        print("Top%d:%.3f" % (k, sum(topk_list[:k]) / val_cnt))
    print("MRR:%.3f" % (sum(mrr_list) / val_cnt))
    return {"MRR": sum(mrr_list) / val_cnt, "Top1": sum(topk_list[:1]) / val_cnt,
            "Top5": sum(topk_list[:5]) / val_cnt, "Top10": sum(topk_list[:10]) / val_cnt,
            "inf_cnt": inf_cnt, "val_cnt": val_cnt}


if __name__ == "__main__":
    Network_Framework = {"num_epochs": 1600,
                         "num_lstm_layers": 1,
                         "batch_size": 32,
                         "embed_size": 16,
                         "hidden_size": 8,
                         "learning_rate": 2e-5,
                         "report_frequency": 9999,
                         "best_model_count": 3,
                         "create_corpus": True,
                         "weight_decay": 0,
                         "feature_size": 4,
                         "corpus_path": "./ms_model/corpus.pkl",
                         "test_data_path": "../data/mms_test_feature_trips_pair.json",
                         "train_data_path": "../data/mms_train_feature_trips.json",
                         "save_path": "./ms_grid_search_model/",
                         "save_root_path": "place_lstm_model_nobos_0508a",
                         "report_epochs": []}
    # best_model_files = train_stage(Network_Framework)
    # for best_model_file in best_model_files:
    #     test_stage(best_model_file, Network_Framework)
    model_path_id = 0
    res_list = []
    for lr in [2e-5, 1e-5]:
        for embed in [8, 16, 12]:
            for hid in [8, 12, 4]:
                for bs in [64, 32]:
                    for wd in [0]:
                        framework = copy.deepcopy(Network_Framework)
                        model_path_id += 1
                        while os.path.exists("./%s/model_%s" % (framework["save_root_path"], model_path_id)):
                            model_path_id += 1
                        sys.stdout.write("Model % s Start At %s" % (model_path_id, datetime.datetime.now()))
                        print("Model % s Start At %s" % (model_path_id, datetime.datetime.now()))

                        framework["learning_rate"] = lr
                        framework["embed_size"] = embed
                        framework["hidden_size"] = hid
                        framework["batch_size"] = bs
                        framework["weight_decay"] = wd
                        framework["report_epochs"] = list(range(5, 30, 5)) + list(range(30, 100, 10)) + list(
                            range(100, 1000, 100)) + list(range(1000, 1600, 50))
                        framework["save_path"] = "./%s/model_%s" % (framework["save_root_path"], model_path_id)
                        framework["corpus_path"] = os.path.join(framework["save_path"], "corpus.pkl")
                        if not os.path.exists(framework["save_path"]):
                            os.mkdir(framework["save_path"])
                        cur_model_res = []
                        for best_model_files in train_stage(framework):
                            for best_model_file in best_model_files:
                                test_res = test_stage_batch(best_model_file, framework)
                                cur_res = [test_res, os.path.join(best_model_file), framework]
                                cur_model_res.append(cur_res)
                                res_list.append(cur_res)
                        json.dump(cur_model_res,
                                  open(os.path.join(framework["save_path"], "model_and_test_info.json"), 'w'))

                        print("Model % s End At %s" % (model_path_id, datetime.datetime.now()))
    print(sorted(res_list, key=lambda x: -x[0]["MRR"])[0])
    pickle.dump(res_list, open("./%s/gs_model_to_%s.pkl" % (Network_Framework["save_root_path"], model_path_id), 'wb'))
    json.dump(res_list, open("./%s/gs_model_to_%s.json" % (Network_Framework["save_root_path"], model_path_id), 'w'))
