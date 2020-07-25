# -*-coding:utf8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
from torch.nn.utils import clip_grad_norm
import os
import codecs
import random
from torch.nn.functional import softmax, log_softmax


class Corpus(object):
    class Dictionary(object):
        def __init__(self):
            self.word2index = {}
            self.index2word = {}
            self.increment_index = 0
            for word in ['<pad>', '<unk>', '<bos>', '<eos>']:
                self.add_word(word)

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

    def read_corpus_file(self, file_path):
        # init
        self.dictionary = self.Dictionary()
        # get unique words
        with codecs.open(filename=file_path, mode='r', encoding='utf8') as rf:
            for line in rf:
                line = line.strip('\n')
                for word in line.split(' '):
                    self.dictionary.add_word(word)
        # get word representation
        # 手动添加 <pad>
        # corpus_vector = []
        # with codecs.open(filename=file_path, mode='r', encoding='utf8') as rf:
        #     for line in rf:
        #         line = line.strip('\n').split(' ')
        #         line = ['<bos>'] + line[:seq_max_length - 2] + ['<eos>'] + ['<pad>'] * max(seq_max_length - 2 - len(line), 0)
        #         corpus_vector.append([self.dictionary.word2index[x] for x in line])
        # corpus_Tensor =  torch.LongTensor(corpus_vector)
        # 使用packed_pad_seq
        corpus_vector = []
        lengths = []
        with codecs.open(filename=file_path, mode='r', encoding='utf8') as rf:
            for line in rf:
                line = line.strip('\n').split(' ')
                line = ['<bos>'] + line + ['<eos>']
                corpus_vector.append(torch.LongTensor([self.dictionary.word2index[x] for x in line]))
                lengths.append(len(line) - 1)
        corpus_vector = torch.nn.utils.rnn.pad_sequence(corpus_vector, batch_first=True, padding_value=0)
        # corpus_Tensor = torch.nn.utils.rnn.pack_padded_sequence(input=corpus_vector, lengths=lengths,batch_first=True, enforce_sorted=False)
        return lengths, corpus_vector

    def read_corpus_train_dev_files(self, train_file_path, dev_file_path):
        self.train_corpus_list = []
        self.dev_corpus_list = []
        # init
        self.dictionary = self.Dictionary()
        # get unique words
        with codecs.open(filename=train_file_path, mode='r', encoding='utf8') as rf:
            for line in rf:
                line = line.strip('\n')
                for word in line.split(' '):
                    self.dictionary.add_word(word)
        # get train vector
        with codecs.open(filename=train_file_path, mode='r', encoding='utf8') as rf:
            for line in rf:
                line = line.strip('\n').split(' ')
                line = ['<bos>'] + line + ['<eos>']
                self.train_corpus_list.append([self.dictionary.word2index[x] for x in line])

        # get dev vector
        with codecs.open(filename=dev_file_path, mode='r', encoding='utf8') as rf:
            for line in rf:
                line = line.strip('\n').split(' ')
                line = ['<bos>'] + line + ['<eos>']
                self.dev_corpus_list.append(
                    [self.dictionary.word2index[x] if x in self.dictionary.word2index else self.dictionary.word2index[
                        '<unk>'] for x in line])

        # summary
        print('train size:{}  dev size:{}'.format(len(self.train_corpus_list), len(self.dev_corpus_list)))
        return

    def get_corpus_tensor(self, dataset='train', shuffle=True):
        corpus_vector = []
        lengths = []
        corpus_dataset = self.train_corpus_list
        if dataset != 'train':
            corpus_dataset = self.dev_corpus_list
        if shuffle:
            random.shuffle(corpus_dataset)
        for line in corpus_dataset:
            corpus_vector.append(torch.LongTensor(line))
            lengths.append(len(line) - 1)
        corpus_vector = torch.nn.utils.rnn.pad_sequence(corpus_vector, batch_first=True, padding_value=0)
        return lengths, corpus_vector


class LstmLM(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, lstm_layer_num):
        super(LstmLM, self).__init__()

        # NN parameters
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.lstm_layer_num = lstm_layer_num
        self.hidden_size = hidden_size
        # Hyper-parameters
        # embed_size = 128
        # hidden_size = 1024
        # num_layers = 1
        # num_epochs = 5
        # num_samples = 1000  # number of words to be sampled
        # batch_size = 20
        # seq_length = 30
        # learning_rate = 0.002

        # NN
        self.layer_embed = nn.Embedding(self.vocab_size, self.embedding_size)
        self.layer_lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.lstm_layer_num, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.simple_elementwise_apply(self.layer_embed, x)
        # x = self.layer_embed(x)

        # Forward propagate LSTM
        packed_out, (h, c) = self.layer_lstm(x, h)
        out, length_info = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Reshape output to (batch_size * sequence_length, hidden_size)
        # out = out.reshape(out.size(0) * out.size(1), out.size(2))
        out_nopad = torch.cat([out[i][:length_info[i], :] for i in range(length_info.size()[0])], dim=0)
        # Decode hidden states of all time steps
        out_linear = self.linear(out_nopad)
        return out_linear, (h, c)


def main():
    # 设备配置
    # Device configuration
    torch.cuda.set_device(7)  # 这句用来设置pytorch在哪块GPU上运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # prepare data
    corpus = Corpus()
    corpus.read_corpus_train_dev_files('./debug_test/corpus/train', './debug_test/corpus/dev')
    train_corpus_lengths, train_corpus_tensor = corpus.get_corpus_tensor(dataset='train', shuffle=True)
    dev_corpus_lengths, dev_corpus_tensor = corpus.get_corpus_tensor(dataset='dev', shuffle=False)
    num_sample, _ = train_corpus_tensor.size()
    # hyper parameters
    embed_size = 256
    vocab_size = len(corpus.dictionary)
    hidden_size = 128
    num_layers = 2
    num_epochs = 10
    batch_size = 128
    num_batches = num_sample // batch_size
    seq_length = 30
    learning_rate = 0.002
    num_checkpoint = 10
    model_save_path = './debug_test/model_gpu_test'
    checkpoint_file_dict = {}
    # preparation
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    # MODEL
    lstmLM = LstmLM(embedding_size=embed_size, vocab_size=vocab_size, hidden_size=hidden_size,
                    lstm_layer_num=num_layers)

    for name, parameters in lstmLM.named_parameters():
        print(name, ':', parameters.size())
    print("LSTMLM have {} paramerters in total".format(sum(x.numel() for x in lstmLM.parameters())))

    # 定义损失函数和优化器
    # loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstmLM.parameters(), lr=learning_rate)

    # 定义函数：截断反向传播
    def detach(states):
        return [state.detach() for state in states]

    dev_loss_nbest = []

    for epoch in range(num_epochs):
        # 初始化隐状态和细胞状态
        states = (torch.zeros(num_layers, batch_size, hidden_size),
                  torch.zeros(num_layers, batch_size, hidden_size))

        dev_states = (torch.zeros(num_layers, batch_size, hidden_size),
                      torch.zeros(num_layers, batch_size, hidden_size))

        train_corpus_lengths, train_corpus_tensor = corpus.get_corpus_tensor(dataset='train', shuffle=True)

        for i in range(0, num_sample - batch_size + 1, batch_size):
            # Get mini-batch inputs and targets
            inputs = train_corpus_tensor[i:i + batch_size, :-1]
            cur_batch_lengths = torch.LongTensor(train_corpus_lengths[i:i + batch_size])
            targets = train_corpus_tensor[i:i + batch_size, 1:]

            # Forward pass
            states = detach(states)
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=inputs, lengths=cur_batch_lengths,
                                                                    batch_first=True,
                                                                    enforce_sorted=False)
            print(packed_inputs.data.size(), packed_inputs.batch_sizes.size(), packed_inputs.sorted_indices.size(),
                  packed_inputs.unsorted_indices.size())
            outputs, states = lstmLM(packed_inputs, states)
            targets_nopad = torch.cat([targets[i][:cur_batch_lengths[i]] for i in range(len(cur_batch_lengths))], dim=0)
            loss = criterion(outputs, targets_nopad.reshape(-1))

            # Backward and optimize
            lstmLM.zero_grad()
            loss.backward()
            clip_grad_norm(lstmLM.parameters(), 0.5)
            optimizer.step()

            step = (i + 1) // batch_size
            if step % 64 == 0:
                print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

            # dev test
            with torch.no_grad():
                total_dev_loss = 0
                for j in range(0, len(dev_corpus_lengths) - batch_size + 1, batch_size):
                    dev_inputs = dev_corpus_tensor[j:j + batch_size, :-1]
                    cur_dev_batch_lengths = torch.LongTensor(dev_corpus_lengths[j:j + batch_size])
                    dev_targets = dev_corpus_tensor[j:j + batch_size, 1:]
                    # Forward pass
                    dev_states = detach(dev_states)
                    dev_packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=dev_inputs,
                                                                                lengths=cur_dev_batch_lengths,
                                                                                batch_first=True,
                                                                                enforce_sorted=False)
                    dev_outputs, dev_states = lstmLM(dev_packed_inputs, dev_states)
                    dev_targets_nopad = torch.cat(
                        [dev_targets[i][:cur_dev_batch_lengths[i]] for i in range(len(cur_dev_batch_lengths))],
                        dim=0)
                    dev_loss = criterion(dev_outputs, dev_targets_nopad.reshape(-1))
                    total_dev_loss += dev_loss.item()
                total_dev_loss = total_dev_loss / len(dev_corpus_lengths) * batch_size
                if len(dev_loss_nbest) < num_checkpoint:
                    dev_loss_nbest.append(total_dev_loss)
                elif dev_loss_nbest[-1] > total_dev_loss:
                    dev_loss_nbest.append(total_dev_loss)
                    old_dev_loss = dev_loss_nbest.pop(0)
                    os.remove(checkpoint_file_dict[old_dev_loss])
                if dev_loss_nbest[-1] == total_dev_loss:  # this means that a better model was founded
                    # save model
                    cp_file_name = os.path.join(model_save_path, 'model_e{}_b{}.pkl'.format(epoch + 1, (step + 1)))
                    checkpoint_file_dict[total_dev_loss] = cp_file_name
                    torch.save(lstmLM.state_dict(), cp_file_name)
                    print('Better Dev Loss: {:.4f}. Saved in {}'.format(total_dev_loss, cp_file_name))

def test_stage(model_file_name, corpus_file_name, context: List, word: str):
    train_corpus = Corpus()
    train_corpus.load_dictionary_from_file(corpus_file_name)
    # hyper parameters
    embed_size = 256
    vocab_size = len(train_corpus.dictionary)
    hidden_size = 128
    num_layers = 2
    batch_size = 128
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstmLM = LstmLM(embedding_size=embed_size, vocab_size=vocab_size, hidden_size=hidden_size,
                    lstm_layer_num=num_layers).to(device)
    lstmLM.load_state_dict(torch.load(model_file_name))

    def get_word_prob(model_to_test: LstmLM, corpus: Corpus, context: List, word: str):
        test_states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                       torch.zeros(num_layers, batch_size, hidden_size).to(device))
        target_wordid = corpus.dictionary.word2index[word]
        prob = 0
        with torch.no_grad():
            if context[0] != '<bos>':
                context = ['<bos>'] + context
            test_input, test_input_length = corpus.get_words_tensor(context)
            test_input = test_input.to(device)
            test_input_length = torch.LongTensor(test_input_length).to(device)
            test_packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=test_input,
                                                                         lengths=test_input_length,
                                                                         batch_first=True,
                                                                         enforce_sorted=False).to(device)
            test_outputs, test_states = model_to_test(test_packed_inputs, test_states)
            # print(test_outputs[-1])
            prob = log_softmax(test_outputs[-1], dim=0)[target_wordid]

        return prob

    return get_word_prob(lstmLM, train_corpus, context, word)


if __name__ == '__main__':
    main()
