import pandas as pd
import config
import collections
import pandas
config=config.Config()

# # #对训练集进行处理并进行划分
# data = pd.read_csv(config.processed_data+'char_ner_train.csv', encoding='utf-8')
#
# a=collections.Counter(data["tag"])
# print(a)
# data.fillna(value=",",inplace=True)
# for line in data.values:
#     if line[1]==",":
#         line[0]=""
#         line[1]=""
#
# dev=data[:50807]
# print(dev.info())
# train_new=data[50809:]
# # print(train_new.info())
#
# print(data.info())
# print(data.loc[50808,:])
#
# with open('new_train.txt', 'a+', encoding='utf-8') as f:
#     for line in train_new.values:
#
#         f.write('{0} {1}\n'.format(str(line[0]),str(line[1])))
# with open('dev.txt', 'a+', encoding='utf-8') as f:
#     for line in dev.values:
#
#         f.write('{0} {1}\n'.format(str(line[0]), str(line[1])))
#
# print(dev.loc[:18,"tag"])
# #将原始test转为文本并填充任意标签
# data = pd.read_csv(config.processed_data+'evaluation_public.csv', encoding='utf-8')
# data.fillna(value=",", inplace=True)
# for line in data.values:
#     if line[1] == ",":
#         line[0] = ""
#         line[1] = ""
# with open('test.txt', 'a+', encoding='utf-8') as f:
#     for line in data.values:
#         if line[1]!="":
#             line[0]=dev.loc[10,"tag"]
#         else:line[0]=""
#         # print(line[0])
#
#         # tag = object("o")
#         # f.write(line[0]+line[1] + '\n')
#         f.write('{0} {1}\n'.format(str(line[1]),str(line[0])))
#     f.close()






#读取数据并统计有多少个句子，并统计句子长度 maxlen=512
with open("test.txt",'r',encoding='utf-8') as f:
    lines = []
    words = []
    labels = []
    for index,line in enumerate(f):
        if len(lines)==2285:
            print("索引为",index)
        contends = line.strip()
        word = line.strip().split(' ')[0]
        # print(word)
        label = line.strip().split(' ')[-1]
        # print(label)
        if contends.startswith("-DOCSTART-"):
            words.append('')
            continue
        if len(contends) == 0:    #读取文本  得到【一句话对应标签、一句话】的形式
            l = ' '.join([label for label in labels if len(label) > 0])
            w = ' '.join([word for word in words if len(word) > 0])
            if  not l:
                print(index)
            lines.append([l, w])
            words = []
            labels = []
            continue

        words.append(word)
        labels.append(label)
    # df=pandas.DataFrame()
    print(len(lines))
    print("前十句话的lines形式为：",lines[:10])
    print("前十句话为：",lines[:10][1])
    print("前十句话每个字符对应的lable为：",lines[:10][0])
    a64,a128,a256,a512,a512_=0,0,0,0,0
    max_len=0
    for i in lines:
        if len(i[1])>max_len:max_len=len(i[1])
        if len(i[1])<64:
            a64+=1
        elif len(i[1])<128:
            a128+=1
        elif len(i[1])<256:
            a256+=1
        elif len(i[1])<512:
            a512+=1
        else:a512_+=1
    print("测试集的句子长度小于64、128、256、512、512+以及最大长度的个数为",a64,a128,a256,a512,a512_,max_len)

    #看lines是否有空白
    for index,line in enumerate(lines):
        if not line[0]:print(index)
    print(lines[:10])
    # # 生成测试集的小样本
    # with open("demo.txt", 'a+', encoding='utf-8') as p:
    #     for i in range(0,10):
    #         p.write(str(lines[i]))
    #     p.close()
    text=[]
    for i in range(len(lines)):
        text.append(lines[i][1])
    print(len(text))
    print(text[2285:2290])
    num=0
    for i in text:
        num += 1
        if not i :
            print(num)
            print("zunzai 空文本")


    f.close()

