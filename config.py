# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Chile
# @Email   : realchilewang@foxmail.com
# @Software: PyCharm

#主要修改tags  和把数据清洗成对应的格式就行
class Config(object):
    def __init__(self):
        # 存模型和读数据参数
        self.data_type = 'bisai'  # 数据集类型 medicine, commerce, finance
        if self.data_type == 'medicine':
            self.processed_data = '/data1/home/fzq/projects/NER/data/ner_data/ner/medicine/ner/'  # 处理后的数据路径
            self.save_model = '/data1/home/fzq/projects/NER/data/ner_data/ner/medicine/ner/model_save/'  # 模型保存路径
            self.batch_size = 16
            self.sequence_length = 384
            # 标签列表
            self.tags = ["[PAD]", "[CLS]", "[SEP]", "O",
                         "B-DRUG", "I-DRUG",
                         "B-DRUG_INGREDIENT", "I-DRUG_INGREDIENT",
                         "B-DISEASE", "I-DISEASE",
                         "B-SYMPTOM", "I-SYMPTOM",
                         "B-SYNDROME", "I-SYNDROME",
                         "B-DISEASE_GROUP", "I-DISEASE_GROUP",
                         "B-FOOD", "I-FOOD",
                         "B-FOOD_GROUP", "I-FOOD_GROUP",
                         "B-PERSON_GROUP", "I-PERSON_GROUP",
                         "B-DRUG_GROUP", "I-DRUG_GROUP",
                         "B-DRUG_DOSAGE", "I-DRUG_DOSAGE",
                         "B-DRUG_TASTE", "I-DRUG_TASTE",
                         "B-DRUG_EFFICACY", "I-DRUG_EFFICACY",
                         ]

            # dym + crf  time/epoch:04:48 3.14s/it
            self.checkpoint_path = "/home/wangzhili/data/graduate/medicine/model/runs_1/1612598879/model_0.8357_0.8546_0.8284_736.bin"

            # ori + crf  time/epoch:04:55 3.22s/it
            # self.checkpoint_path = "/home/wangzhili/data/graduate/medicine/model/runs_0/1612598805/model_0.7727_0.8001_0.7768_736.bin"
        elif self.data_type == 'commerce':
            self.processed_data = '/data1/home/fzq/projects/NER/data/ner_data/ner/commerce/ner/'  # 处理后的数据路径
            self.save_model = '/data1/home/fzq/projects/NER/data/ner_data/ner/commerce/ner/model_save/'  # 模型保存路径
            self.tags = ["[PAD]", "[CLS]", "[SEP]", "O",
                         'B-company', 'I-company',
                         'B-scene', 'I-scene',
                         'B-name', 'I-name',
                         'B-game', 'I-game',
                         'B-organization', 'I-organization',
                         'B-position', 'I-position',
                         'B-book', 'I-book',
                         'B-government', 'I-government',
                         'B-address', 'I-address',
                         'B-movie', 'I-movie',
                         'B-mobile', 'I-mobile',
                         'B-email', 'I-email',
                         'B-vx', 'I-vx',
                         'B-QQ', 'I-QQ',
                         ]
            self.sequence_length = 384
            self.batch_size = 16

            # ori + crf  3.67s/it
            self.checkpoint_path = "/home/wangzhili/data/graduate/commerce/model/runs_0/1612611701/model_0.7746_0.8036_0.7833_1496.bin"

            # dym + crf  3.88s/it
            self.checkpoint_path = "/home/wangzhili/data/graduate/commerce/model/runs_1/1612611758/model_0.8147_0.8547_0.8333_1496.bin"
        elif self.data_type == 'finance':
            self.processed_data = '/data1/home/fzq/projects/NER/data/ner_data/ner/finance/ner/'  # 处理后的数据路径
            self.save_model = '/data1/home/fzq/projects/NER/data/ner_data/ner/finance/ner/model_save/'  # 模型保存路径
            self.tags = ["[PAD]", "[CLS]", "[SEP]", "O",
                         'B-ORG', 'I-ORG'
                         ]
            self.sequence_length = 512
            self.batch_size = 2

            # dym + crf  3.22s/it  weighted layer
            self.checkpoint_path = "/home/wangzhili/model_0.7037_0.8350_0.7631_1294.bin"

            # ori + crf  3.18s/it
            # self.checkpoint_path = "/home/wangzhili/model_0.7644_0.7947_0.7793_5176.bin"

        elif self.data_type == 'bisai':
            self.processed_data = '/data1/home/fzq/projects/NER/data/competetipn_data/'  # 处理后的数据路径
            self.save_model = '//data1/home/fzq/projects/NER/data/competetipn_data/model_save/'  # 模型保存路径
            self.tags = ["[PAD]", "[CLS]", "[SEP]", "O",
                         'B-ORG','M-ORG', 'E-ORG','S-ORG',
                         'B-GPE', 'M-GPE','E-GPE','S-GPE',
                         'B-PER','M-PER','E-PER','S-PER',
                         'B-LOC','M-LOC','E-LOC','S-LOC'
                         ]
            self.sequence_length =512
            self.batch_size =4

            # dym + crf  3.22s/it  weighted layer
            self.checkpoint_path = "/data1/home/fzq/projects/NER/data/competetipn_data/model_save/runs_3/1630212128/model_0.6264_0.6044_0.6115_18420.bin"

            # ori + crf  3.18s/it
            # self.checkpoint_path = "/home/wangzhili/model_0.7644_0.7947_0.7793_5176.bin"

        self.use_origin_bert = False
        self.warmup_proportion = 0.05
        self.pretrainning_model = 'roberta'

        self.decay_rate = 0.5
        self.train_epoch = 10

        self.learning_rate = 1e-4
        self.embed_learning_rate = 5e-5

        self.embed_dense = 512

        if self.pretrainning_model == 'nezha':
            model = '/data1/home/fzq/projects/NER/data/model/nezha-cn-base/nezha-cn-base/'
        elif self.pretrainning_model == 'roberta':
            model = '/data1/home/fzq/projects/NER/data/model/RoBERTa_zh_L12_PyTorch/'
        else:
            model = '/home/wangzhili/pretrained_model/Torch_model/pre_model_electra_base/'

        self.model_path = model
        self.bert_config_file = model + 'config.json'
        self.bert_file = model + 'pytorch_model.bin'
        self.vocab_file = model + 'vocab.txt'

        """
        下接结构
        """
        self.mid_struct = 'rtransformer'  # bilstm,idcnn,rtransformer,tener, base
        self.num_layers = 1  # 下游层数
        # bilstm
        self.lstm_hidden = 256  # Bilstm隐藏层size
        # idcnn
        self.filters = 128  # idcnn
        self.kernel_size = 9
        # Tener
        self.num_layers = 1
        self.tener_hs = 256
        self.num_heads = 4
        # rTansformer
        self.k_size = 32
        self.rtrans_heads = 4

        self.drop_prob = 0.1  # drop_out率
        # self.gru_hidden_dim = 64
        self.rnn_num = 256
        self.restore_file = None
        self.gradient_accumulation_steps = 1
        self.embed_name = 'bert.embeddings.word_embeddings.weight'  # 词
