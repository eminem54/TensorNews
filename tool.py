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
def normalize(text, english=True, number=True, punctuation=False):
    hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
    text = hangul.sub('', text)
    return text

def loading_contents(data_path,eng=True,num=True,punc=False):
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents=[]
    for doc in corpus:
        if type(doc[0]) is not str :
            continue
        if len(doc[0]) > 0 :
            tmpcontents = normalize(doc[0], english=eng, number=num, punctuation=punc)
            contents.append(tmpcontents)
    return contents


#제목, 본문 3개에 단어씩 파싱해서 데이터 저장
def make_dict_all_cut(contents, minlength, maxlength, jamo_delete=False):
    dict = defaultdict(lambda: [])
    for doc in contents:
        for idx, word in enumerate(doc.split()):
            if len(word) > minlength:
                normalizedword = word[:maxlength]
                if jamo_delete:
                    tmp = []
                    for char in normalizedword:
                        if ord(char) < 12593 or ord(char) > 12643:
                            tmp.append(char)
                    normalizedword = ''.join(char for char in tmp)
                if word not in dict[normalizedword]:
                    dict[normalizedword].append(word)
    dict = sorted(dict.items(), key=operator.itemgetter(0))[1:]
    words = []
    for i in range(len(dict)):
        word = []
        word.append(dict[i][0])
        for w in dict[i][1]:
            if w not in word:
                word.append(w)
        words.append(word)

    words.append(['<PAD>'])
    words.append(['<S>'])
    words.append(['<E>'])
    words.append(['<UNK>'])
    # word_to_ix, ix_to_word 생성
    ix_to_word = {i: ch[0] for i, ch in enumerate(words)}
    word_to_ix = {}
    for idx, words in enumerate(words):
        for word in words:
            word_to_ix[word] = idx
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s'
                  % (len(contents), len(ix_to_word)))
    return word_to_ix, ix_to_word


def add_dict_all_cut(ix_to_word, word_to_ix, contents, minlength, maxlength, jamo_delete=False):
    dict = defaultdict(lambda: [])
    for doc in contents:
        for idx, word in enumerate(doc.split()):
            if len(word) > minlength:
                normalizedword = word[:maxlength]
                if jamo_delete:
                    tmp = []
                    for char in normalizedword:
                        if ord(char) < 12593 or ord(char) > 12643:
                            tmp.append(char)
                    normalizedword = ''.join(char for char in tmp)
                if word not in dict[normalizedword]:
                    dict[normalizedword].append(word)
    dict = sorted(dict.items(), key=operator.itemgetter(0))[1:]
    words = []
    for i in range(len(dict)):
        word = []
        word.append(dict[i][0])
        for w in dict[i][1]:
            if w not in word:
                word.append(w)
        words.append(word)

    words.append(['<PAD>'])
    words.append(['<S>'])
    words.append(['<E>'])
    words.append(['<UNK>'])
    maxIndex = max(ix_to_word.keys())
    # word_to_ix, ix_to_word 생성
    for i, ch in enumerate(words):
        if ch[0] not in ix_to_word.values():
            ix_to_word.update({i+maxIndex: ch[0]})
            word_to_ix.update({ch[0]: i+maxIndex})
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s'
                  % (len(contents), len(ix_to_word)))
    return word_to_ix, ix_to_word



#데이터 읽어 오기
def loading_data(data_path, eng=True, num=True, punc=False):
    # data example : "title","content"
    # data format : csv, utf-8
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    title = []
    contents = []
    for doc in corpus:
        if type(doc[0]) is not str or type(doc[1]) is not str:
            continue
        if len(doc[0]) > 0 and len(doc[1]) > 0:
            tmptitle = normalize(doc[0], english=eng, number=num, punctuation=punc)
            tmpcontents = normalize(doc[1], english=eng, number=num, punctuation=punc)
            title.append(tmptitle)
            contents.append(tmpcontents)
    return title, contents


def check_doclength(docs, sep=True):
    max_document_length = 0
    for doc in docs:
        if sep:
            words = doc.split()
            document_length = len(words)
        else:
            document_length = len(doc)
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length