import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
import json
import os, glob
import matplotlib.pyplot as plt
from translate import Translator

translator = Translator(to_lang="english")


# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 剔除标点符号,\xa0 空格
def pretreatment(comments):
    result_comments = []
    punctuation = '。，？！：%&~（）、；“”&|,.?!:%&~();""'
    for comment in comments:
        comment = str(comment)
        comment = ''.join([c for c in comment if c not in punctuation])
        comment = ''.join(comment.split())  # \xa0
        result_comments.append(comment)

    return result_comments


class Bert_BiLSTM(nn.Module):
    def __init__(self, bertpath, hidden_dim, output_size, n_layers, bidirectional=True, droput=0.5):
        super(Bert_BiLSTM, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # Bert Path
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        # LSTM layers  768 because of Bert_output's embedding dim is 768
        self.lstm = nn.LSTM(768, hidden_dim, n_layers, batch_first=True, bidirectional=bidirectional)

        # dropout layer
        self.dropout = nn.Dropout(droput)

        if bidirectional:
            self.fc = nn.Linear(hidden_dim*2, output_size)
        else:
            self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        # 生成bert字向量
        x = self.bert(x)[0]  # bert字向量

        lstm_out, (hidden_last, cn_last) = self.lstm(x, hidden)

        # print(lstm_out.shape)   torch.Size([2, 200, 768])
        # print(hidden_last.shape)   torch.Size([4, 2, 384])
        # print(cn_last.shape)   torch.Size([4, 2, 384])

        # 双向LSTM需要修改
        if self.bidirectional:
            # 正向最后一层，最后一个时刻
            hidden_last_L = hidden_last[-2]
            # print(hidden_last_L.shape)  #[32, 384]
            # 反向最后一层，最后一个时刻
            hidden_last_R = hidden_last[-1]
            # print(hidden_last_R.shape)   #[32, 384]
            # 进行拼接
            hidden_last_out = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
            # print(hidden_last_out.shape,'hidden_last_out')   #[32, 768]
        else:
            hidden_last_out = hidden_last[-1]

        out = self.dropout(hidden_last_out)   #  torch.Size([2, 768])
        out = self.fc(out)

        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        number = 1
        if self.bidirectional:
            number = 2

        hidden = (weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float(),
                  weight.new(self.n_layers * number, batch_size, self.hidden_dim).zero_().float()
                  )
        return hidden


class ModelConfig:
    batch_size = 32
    output_size = 6
    hidden_dim = 384
    n_layers = 2
    lr = 2e-5
    bidirectional = True  # 这里为True，为双向LSTM
    epochs = 1
    print_every = 10
    clip = 5  # gradient clipping
    bert_path = 'bert-base-chinese'  # 预训练bert路径
    save_path = 'bert_bilstm.pth'  # 模型保存路径


def train_model(config, data_train):
    net = Bert_BiLSTM(config.bert_path,
                      config.hidden_dim,
                      config.output_size,
                      config.n_layers,
                      config.bidirectional)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    net.train()
    for e in range(config.epochs):
        h = net.init_hidden(config.batch_size)
        counter = 0
        print(len(data_train))
        for inputs, labels in data_train:
            counter += 1

            h = tuple([each.data for each in h])
            # print(h)
            net.zero_grad()
            output = net(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            if counter % config.print_every == 0:
                print(f"counter = {counter} loss = {loss.item()}")
    torch.save(net.state_dict(), config.save_path)


def load_model(config):
    net = Bert_BiLSTM(config.bert_path,
                      config.hidden_dim,
                      config.output_size,
                      config.n_layers,
                      config.bidirectional)

    net.load_state_dict(torch.load(config.save_path))  # 模型重新加载

    return net


def predict(test_comment_list, config, net):
    result_comments = pretreatment(test_comment_list)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    result_comments_id = tokenizer(result_comments,
                                   padding=True,
                                   truncation=True,
                                   max_length=200,
                                   return_tensors='pt')
    tokenizer_id = result_comments_id['input_ids']
    inputs = tokenizer_id
    batch_size = inputs.size(0)
    h = net.init_hidden(batch_size)

    net.eval()
    with torch.no_grad():
        output = net(inputs, h)
        output = torch.nn.Softmax(dim=1)(output)
        pred_label = [label_trans_reverse[lab.item()] for lab in torch.max(output, 1)[1]]
        # printing output value, before rounding
        pred_posibility = [pos.item() for pos in torch.max(output, 1)[0]]
        return pred_label, pred_posibility


def read_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print('%s -> data over' % file)
    return data


def train_main(model_config, file):
    data = read_json(file)
    result_comments = pretreatment(list(item['content'] for item in data))
    tokenizer = BertTokenizer.from_pretrained(model_config.bert_path)

    result_comments_id = tokenizer(result_comments,
                                   padding=True,
                                   truncation=True,
                                   max_length=200,
                                   return_tensors='pt')
    X = result_comments_id['input_ids']
    Y = torch.from_numpy(np.array(list(label_trans[item['label']] for item in data))).float()

    train_data = TensorDataset(X, Y)
    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=model_config.batch_size,
                              drop_last=True)
    train_model(model_config, train_loader)


def draw_distribution(df, topic, keyword):
    index = np.arange(len(df))
    emo_type = len(df.columns)
    color = plt.get_cmap('Accent')(range(emo_type))
    # 建立簇状柱形图
    for i in range(emo_type):
        plt.bar(
            index + 1 / (emo_type + 1) * i,
            df[df.columns[i]],
            color=color[i],
            width=1 / (emo_type + 1))
    plt.xticks(index + 1 / 4, topic)
    # 添加图例
    plt.legend(df.columns, title='Sentiment')
    plt.title('Statistical chart of sentiment tendency of Weibo comments'.upper())
    plt.savefig(f'微博评论情感倾向统计图({keyword})')
    plt.show()


def predict_main(keyword):
    file_list = glob.glob(f'..\\comment_spider\\{keyword}\\*.xls')
    all_res = list()
    net = load_model(model_config)
    topic_list = []
    num = 1
    label_lst = []
    for file in file_list[:4]:
        df = pd.read_excel(file)
        test_comments = df['comment_content']
        prediction_label, posibility = predict(test_comments, model_config, net)
        # predict_dic = [dict(zip(['label', 'posibility'], [label[id], posibility[id]])) for id, _ in enumerate(label)]
        # result_dic = dict(zip(range(len(predict_dic)), predict_dic))
        distribution_result = list(prediction_label.count(emo) for emo in label)
        all_res.append(distribution_result)
        topic_label = f"topic{num}"
        num += 1
        topic = file.split('\\')[-1][:-4]
        # topic = '\n'.join([topic[i:i+6] for i in range(0, len(topic), 6)])
        label_lst.append(topic_label)
        topic_list.append(topic+"\n")
    df = pd.DataFrame(all_res, columns=label)
    # print(df)
    with open(keyword+'.txt', 'w', encoding='utf-8') as f:
        f.writelines(topic_list)
    draw_distribution(df, label_lst, keyword)


label = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
label_trans = dict(zip(label, range(len(label))))
# print(label_trans)
label_trans_reverse = dict(zip(range(len(label)), label))
# print(label_trans_reverse)


if __name__ == '__main__':
    model_config = ModelConfig()
    # file = './data/usual_train.txt'
    # train_main(model_config, file)
    predict_main('全球性别不平等报告')

