import numpy as np
import pandas as pd
from konlpy.tag import Kkma,Komoran,Twitter
from pandas import Series
import re
from collections import defaultdict
import operator
import tensorflow as tf
import time as time
from gensim.models import word2vec
from Tensorflow import tool



#인코더 사이즈 만큼의 플레이스홀더를 생성해 인코더 인풉을 만든다.
#디코더,타겟,타겟무게를 만든다.
#타겟 무게는 제목(타겟)에 대응하는 요소가 <pad>일 경우 0, <pad> 가 아닐 경우 1로 채워넣은 리스트
#target_weights는 정답에 끼어있는 <pad>가 학습에 반영되지 않도록 도와주는 값
class seq2seq(object):

    def __init__(self, multi, hidden_size, num_layers, forward_only,
                 learning_rate, batch_size,
                 vocab_size, encoder_size, decoder_size):

        # variables
        self.source_vocab_size = vocab_size
        self.target_vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.global_step = tf.Variable(0, trainable=False)

        # networks
        W = tf.Variable(tf.random_normal([hidden_size, vocab_size]))
        b = tf.Variable(tf.random_normal([vocab_size]))
        output_projection = (W, b)
        self.encoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        self.decoder_inputs = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        self.targets = [tf.placeholder(tf.int32, [batch_size]) for _ in range(decoder_size)]
        self.target_weights = [tf.placeholder(tf.float32, [batch_size]) for _ in range(decoder_size)]

        # models
        if multi:
            single_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            #cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

        if not forward_only:
            self.outputs, self.states =tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=False)

            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]
            self.loss = []
            for logits, targets, target_weight in zip(self.logits, self.targets, self.target_weights):
                crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits)
                self.loss.append(crossentropy * target_weight)
            self.cost = tf.math.add_n(self.loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

        else:
            self.outputs, self.states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=True)
            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]

    def step(self, session, encoderinputs, decoderinputs, targets, targetweights, forward_only):
        input_feed = {}
        for l in range(len(encoder_inputs)):
            input_feed[self.encoder_inputs[l].name] = encoderinputs[l]
        for l in range(len(decoder_inputs)):
            input_feed[self.decoder_inputs[l].name] = decoderinputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = targetweights[l]
        if not forward_only:
            output_feed = [self.train_op, self.cost]
        else:
            output_feed = []
            for l in range(len(decoder_inputs)):
                output_feed.append(self.logits[l])
        output = session.run(output_feed, input_feed)
        if not forward_only:
            return output[1] # loss
        else:
            return output[0:] # outputs








#파일을 읽어 들여옴
title,contents = tool.loading_data("newdata.csv", eng=True, num=True, punc=False)
#학습 말뭉치를 읽어들임
word_to_ix, ix_to_word = tool.make_dict_all_cut(title + contents, minlength=0, maxlength=3, jamo_delete=True)

multi = False        #은닉층을 1개만 쌓을건지 여러개 쌓을건지 지정하는 변수
forward_only = False    #학습 때는 false, 추론(테스트)때는 true로 지정
hidden_size = 300
vocab_size = len(ix_to_word)    #말뭉치 사전 단어수
num_layers = 3                 #은닉층 개수
learning_rate = 0.0001           #학습률
batch_size = 5                #1회당 학습하는 데이터 수
encoder_size = 100              #인코더에 넣을 입력 시퀀스의 최대 길이
decoder_size = tool.check_doclength(title,sep=True) # 디코더에 넣을 시퀀스의 최대 길이
#입력 컨텐츠의 최대 길이를 반환
steps_per_checkpoint = 10

# 데이터 전처리에서 설명드렸듯 단어를 숫자로 바꾸는 과정을 한번에 해주는 함수
encoderinputs, decoderinputs, targets_, targetweights = tool.make_inputs(contents, title, word_to_ix, encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)




sess = tf.Session()

model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers,
                    learning_rate=learning_rate, batch_size=batch_size,
                    vocab_size=vocab_size,
                    encoder_size=encoder_size, decoder_size=decoder_size,
                    forward_only=forward_only)
sess.run(tf.global_variables_initializer())
step_time, loss = 0.0, 0.0
current_step = 0
start = 0
end = batch_size
while current_step < 300:
    if end > len(title):
        start = 0
        end = batch_size

    # Get a batch and make a step
    start_time = time.time()
    encoder_inputs, decoder_inputs, targets, target_weights = make_batch(encoderinputs[start:end],
                                                                              decoderinputs[start:end],
                                                                              targets_[start:end],
                                                                              targetweights[start:end])

    if current_step % steps_per_checkpoint == 0:
        for i in range(decoder_size - 2):
            decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
        output_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        predict = ' '.join(ix_to_word[ix] for ix in predict)
        real = [word[0] for word in targets]
        real = ' '.join(ix_to_word[ix] for ix in real)
        print('\n----\n step : %s \n time : %s \n LOSS : %s \n 예측 : %s \n 손질한 정답 : %s \n 정답 : %s \n----' %
              (current_step, step_time, loss, predict, real, title[start]))
        loss, step_time = 0.0, 0.0

    step_loss = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, False)
    step_time += time.time() - start_time / steps_per_checkpoint
    loss += np.mean(step_loss) / steps_per_checkpoint
    current_step += 1
    start += batch_size
    end += batch_size