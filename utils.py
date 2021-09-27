# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm

import numpy as np
from transformers import BertTokenizer
from tqdm import tqdm
from config import Config


def load_data(data_file_name):
    with open(data_file_name,'r',encoding='utf-8') as f:
        lines = []
        words = []
        labels = []
        for line in f:

            contends = line.strip()
            word = line.strip().split(' ')[0]
            # print(word)
            label = line.strip().split(' ')[-1]
            # print(label)
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue
            if len(contends) == 0:
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                if  l:lines.append([l, w])  #此数据集存在空得文本
                # if not w:
                #     w="."
                # lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label)
        print(lines[-1]) #输出最后一个句子
    return lines



#examples包含三个东西  ：此列表 每一个样本包含【guid=guid, text=text, label=label】
def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label
def create_example(lines):
    examples = []
    for (index, line) in enumerate(lines):
        guid = "%s" % index  #句子索引、从o开始  str guid的大小即为句子的个数
        label = line[0]
        text = line[1]
        examples.append(InputExample(guid=guid, text=text, label=label))
    return examples

def get_labels():
    return Config().tags


class DataIterator(object):
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, pretrainning_model=False, seq_length=100, is_test=False):
        # 数据文件位置
        self.data_file = data_file
        self.data = get_examples(data_file) #读取到所有得数据，每一句子 、句子每个字符队应的标记以及句子索引
        self.batch_size = batch_size
        self.pretrainning_model = pretrainning_model
        self.seq_length = seq_length

        # 数据的个数
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引！！
        self.is_test = is_test

        if not self.is_test:   #是test就不shuffle
            self.shuffle()
        self.tokenizer = tokenizer
        self.label_map = {}
        for index, label in enumerate(get_labels()):
            self.label_map[label] = index
        print("标签个数：", len(get_labels()))
        print("样本个数：", self.num_records)   #句子的个数

    def convert_single_example(self, example_idx):    #根据索引将句子转换为模型能识别的数字的形式
        text_list = self.data[example_idx].text.split(" ")
        label_list = self.data[example_idx].label.split(" ")
        tokens = text_list
        labels = label_list    #获取到字符、和对应的标签列表

        # seq_length=128 则最多有126个字符
        # <cls>文本<sep>
        if len(tokens) >= self.seq_length - 1:  #如果句子长度超出最大限制   那么进行截取
            tokens = tokens[:(self.seq_length - 2)]
            labels = labels[:(self.seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []

        ntokens.append('[CLS]') #句子头添加一个cls符号
        segment_ids.append(0)
        label_ids.append(self.label_map['[CLS]'])  #添加cls对应的编号

        for index, token in enumerate(tokens):
            try:  # play + ## ing
                ntokens.append(self.tokenizer.tokenize(token.lower())[0])  # 全部转换成小写, 方便BERT词典
            except:
                ntokens.append('[UNK]')      #转换不了小写的改为unknown标签
            segment_ids.append(0)
            if index==2285:
                print("索引2285的句子为",tokens)
            # print(index,tokens)
            label_ids.append(self.label_map[labels[index]])

        tokens = ["[CLS]"] + tokens + ["[SEP]"]   #句子首尾加两个符号
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[SEP]"])    #标签列表最后添加sep对应的标号

        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)  #根据词典将token转换为对应的ids   数字化
        input_mask = [1] * len(input_ids)    #mask码，有字符的位置对应设为1
        while len(input_ids) < self.seq_length:    #若句子长度小于max，三个编码列表都补充零    标签列表添加pad符号
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(self.label_map["[PAD]"])
            ntokens.append("*NULL*")
            tokens.append("*NULL*")
        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        assert len(label_ids) == self.seq_length
        assert len(tokens) == self.seq_length
        return input_ids, input_mask, segment_ids, label_ids, tokens

    def shuffle(self):
        np.random.shuffle(self.all_idx)    #将句子token的索引列表打乱

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        label_ids_list = []
        tokens_list = []

        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]   #已经打乱了
            # print("168的idx为：",idx)
            res = self.convert_single_example(idx)  #根据id转为字典里对应的数字
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, label_ids, tokens = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            label_ids_list.append(label_ids)
            tokens_list.append(tokens)

            if self.pretrainning_model:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:   #若转换的句子数大于总数就退出
                break

        while len(input_ids_list) < self.batch_size:
            input_ids_list.append(input_ids_list[0])
            input_mask_list.append(input_mask_list[0])
            segment_ids_list.append(segment_ids_list[0])
            label_ids_list.append(label_ids_list[0])
            tokens_list.append(tokens_list[0])

        return input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list     #返回的是一个batch的数字输入形式


if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path=config.model_path,
                                              do_lower_case=True,
                                              never_split=["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"])
    train_iter = DataIterator(config.batch_size,
                              data_file=config.processed_data + 'new_train.txt',
                              pretrainning_model=config.pretrainning_model,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'test.txt',
                            pretrainning_model=config.pretrainning_model,
                            seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    for input_ids_list, input_mask_list, segment_ids_list, label_ids_list, tokens_list in tqdm(dev_iter):
        print("输入句子最后一个batch的id列表：",input_ids_list[0])
        print("mask编码：",input_mask_list[0])
        print("输入句子最后一个batch的token列表：", tokens_list[0])
        print("输入句子最后一个batch的对应的标签列表：",label_ids_list[0])   #  输出验证集第一个batch 的第一句话
        break   #第一个batch处理完自动处理下一个

