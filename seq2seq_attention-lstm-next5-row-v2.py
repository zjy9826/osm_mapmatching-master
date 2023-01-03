from itertools import chain
import torch.utils.data as data
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from LocalGat import  LocalGAT
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
# import Levenshtein
from sklearn.utils import shuffle
import pandas as pd
import math
import random



def get_onehot_neighbor(lis,num):
    onehot_encoded = []
    for value in lis:
        letter = [0 for _ in range(num)]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def get_onehot(lis,num):
    # lis = lis.astype(int)
    # print(lis.shape)
    onehot_encoded = []
    for value in lis:
        letter = [0 for _ in range(num)]
        letter[value[0]] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

def get_onehot_taxi(lis,num):
    lis = lis.astype(int)
    # print(lis.shape)
    onehot_encoded = []
    for value in lis:
        temp = []
        # print(value)
        for i in value:
            # print(i)
            letter = [0 for _ in range(num)]
            letter[i] = 1
            temp.append(letter)
        onehot_encoded.append(temp)
    
    return onehot_encoded

def get_dir_onehot(lis,num):
    # print(lis.shape)
    one_hot = []
    lis = lis.astype(int)
    for item in lis:
        # print(item)
        onehot_encoded = []
        for value in item:
            # print(value)
            letter = [0 for _ in range(num)]
            letter[value] = 1
            onehot_encoded.append(letter)
        one_hot.append(onehot_encoded)
        # break
    # print(np.array(one_hot).shape)
    return np.array(one_hot)


## 计算编辑距离
# def get_DE(s1,s2):
#     de = 0
#     for i in range(len(s1)):   
#         de += Levenshtein.distance(str(s1[i]),str(s2[i]))

#     return de/len(s1)


def get_data_roadsegment_ce():
    # 加载数据

    dataset = np.load("./1114/traj_dataset_window5_5fea_taxiid.npy",allow_pickle=True)

    rng = np.random.default_rng(12345)
    # 打乱数据顺序
    rng.shuffle(dataset)
    # dataset = dataset_emb

    # 划分数据集，8:2:2
    train_size = int(len(dataset) * 0.1)
    trainlist = dataset[:train_size]  # 训练集

    validationlist = dataset[train_size:int(len(dataset) * 0.15)]  # 验证集
    testlist = dataset[int(len(dataset) * 0.15):int(len(dataset) * 0.2)]  # 测试集

    # train_size = int(len(dataset) * 0.1)
    # trainlist = dataset[:train_size]  # 训练集

    # validationlist = dataset[train_size:int(len(dataset) * 0.15)]  # 验证集
    # testlist = dataset[int(len(dataset) * 0.15):int(len(dataset) * 0.2)]


    # 为了获得 targets
    # edgeid,node_u,node_v,angle,象限,taxiid
    L = []  #
    L.append(2)
    L.append(4)
    L.append(5)

    # 预处理
    length = 15  # 每个样本的长度
    look_back = length - 1
    next5 = -5

    """
    也就是说我需要在trainx中放入前14个，然后使用这个14个中后5个当做decoder的输入。
    """

    trainX = trainlist[:, :look_back, (L)]
    train_segment_id = trainlist[:, next5:, 4].astype(int)
    train_eid = trainlist[:, next5:, 0].astype(int)
    # trainY = get_onehot(train_segment_id, 8)  # onehot

    # trainX = trainlist[:, :look_back, :]
    # train_segment_id = trainlist[:, look_back:look_back + 1,0].astype(int)
    # trainY = get_onehot(train_segment_id, 118)  # onehot

    # print(train_segment_id.shape)

    validationX = validationlist[:, :look_back, (L)]
    validation_segment_id = validationlist[:, next5:, 4].astype(int)
    validation_eid = validationlist[:, next5:, 0].astype(int)
    # validationY = get_onehot(validation_segment_id, 8)  # onehot

    # validationX = validationlist[:, :look_back, :]
    # validation_segment_id = validationlist[:, look_back:look_back + 1, 0].astype(int)
    # validationY = get_onehot(validation_segment_id, 118)  # onehot

    #
    testX = testlist[:, :look_back, (L)]
    test_segment_id = testlist[:, next5:, 4]
    test_eid = testlist[:, next5:, 0].astype(int)
    # testY = get_onehot(test_segment_id, 8)  # onehot

    # testX = testlist[:, :look_back, :]
    # test_segment_id = testlist[:, look_back:look_back + 1, 0].astype(int)
    # testY = get_onehot(test_segment_id, 118)  # onehot

    return trainX, validationX, testX, train_segment_id, validation_segment_id, test_segment_id,train_eid,validation_eid,test_eid

class Encoder(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.1, num_layers=2)  # encoder

    def forward(self, enc_input):
        enc_input = enc_input.to(device)

        seq_len, batch_size, embedding_size = enc_input.size()

        h_0 = torch.rand(2, batch_size, self.hidden_size).to(device)
        c_0 = torch.rand(2, batch_size, self.hidden_size).to(device)
        # en_ht:[num_layers * num_directions,Batch_size,hidden_size]
        encode_output, (encode_ht, decode_ht) = self.encoder(enc_input, (h_0, c_0))
        return encode_output, (encode_ht, decode_ht)  # H,h0,c0

class Decoder_row(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.decoder_row = nn.LSTM(input_size=in_features, hidden_size=hidden_size, dropout=0.1, num_layers=2)  # encoder

    def forward(self, dec_input,h_0,c_0):

        dec_output, (h_0, c_0) = self.decoder_row(dec_input, (h_0, c_0))
        return dec_output, (h_0, c_0)  # H,h0,c0


class Decoder(nn.Module):
    def __init__(self, in_features,output_size, enc_hid_size, dec_hid_size, Attn):
        super().__init__()
        self.in_features = in_features
        self.Attn = Attn
        self.enc_hid_size = enc_hid_size
        self.dec_hid_size = dec_hid_size
        self.crition = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features + enc_hid_size, in_features)
        self.decoder = nn.LSTM(input_size=in_features, hidden_size=dec_hid_size, dropout=0.1, num_layers=2)  # encoder
        self.linear = nn.Linear(dec_hid_size,output_size)

    def forward(self, enc_output, dec_input, s):
        # s : [1, Batch_size , enc_hid_size ] s表示解码器的某一个隐含层的输出
        # enc_output : [seq_len, Batch_size,enc_hid_size]   对应于整个解码器的某一个输入
        # dec_input : [1, Batch_size, embed_size]  对应于解码器的某一个输入
        # dec_input = dec_input.unsqueeze(1)
        seq_len, Batch_size, embed_size = enc_output.size()
        atten = self.Attn(s, enc_output)  # atten : [Batch_size, seq_len]



        atten = atten.unsqueeze(2)  # atten : [Batch_size, seq_len, 1]
        atten = atten.transpose(1, 2)  # atten : [Batch_size, 1, seq_len]
        enc_output = enc_output.transpose(0, 1)
        ret = torch.bmm(atten, enc_output)  # ret : [Batch_size, 1, enc_hid_size]
        ret = ret.transpose(0, 1)  # ret : [1, Batch_size, enc_hid_size]
        # dec_input = dec_input.transpose(0, 1)  # dec_input : [1, Batch_size, embed_size]

        # print(ret.shape)
        # print(dec_input.shape)

        dec_input_t = torch.cat((ret, dec_input), dim=2)  # dec_input_t : [1, Batch_size, enc_hid_size+embed_size]
        dec_input_tt = self.fc(dec_input_t)  # dec_input_tt : [1, Batch_size, embed_size]
        c0 = torch.zeros(2, Batch_size, embed_size)
        s = s.to(device)
        c0 = c0.to(device)
        de_output, (s, _) = self.decoder(dec_input_tt, (s, c0))  # de_output:[1, Batch_size, dec_hid_size]
        de_output = de_output.transpose(0, 1)
        pre = self.linear(de_output.view(de_output.shape[0],-1))
        # pre = F.softmax(pre,dim=1)
        return pre, s


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(dec_hid_dim + enc_hid_dim, dec_hid_dim)
        self.fc2 = torch.nn.Linear(dec_hid_dim, 1)

    def forward(self, s, enc_output):
        # 将解码器的输出S和编码器的隐含层输出求相似性
        # s: [1, Batch_size, dec_hid_size]
        # enc_output: [seq_len, Batch_size, enc_hid_size ]
        seq_len, Batch_size, enc_hid_size = enc_output.size()
        # print(enc_output.size())
        # print(s.shape)
        s = s.repeat(int(seq_len/2), 1, 1)  # s: [seq_len, Batch_size, dec_hid_size]

        # print(s.shape)
        # print(enc_output.shape)

        a = torch.tanh(torch.cat((s, enc_output), 2))  # a: [Batch_size, seq_len, dec_hid_size + enc_hid_size ]
        a = self.fc1(a)  # a :  [Batch_size, seq_len, dec_hid_dim]
        a = self.fc2(a)  # a :  [Batch_size, seq_len, 1]
        a = a.squeeze(2)  # a :  [Batch_size, seq_len]
        return F.softmax(a, dim=1).transpose(0, 1)  # softmax 只进行归一化，不改变张量的维度

class temporal_att(nn.Module):
    def __init__(self,B):
        super().__init__()
        self.fc1 = torch.nn.Linear(5*B, 1).to(device="cuda")

    def forward(self,enc_output,h_p,c_p):
        # enc_output size :torch.Size([10, 20, 512])
        z = torch.cat((h_p, c_p,enc_output[0,:,:].unsqueeze(0)), dim=0).transpose(0,1)
        z = z.reshape(z.shape[0],-1)
        z = self.fc1(z)
        for i in range(1,enc_output.shape[0]):
            # print(f"h_p:{h_p.shape},c_p:{c_p.shape},enc:{enc_output[i,:,:].unsqueeze(0).shape}")
            x = torch.cat((h_p, c_p,enc_output[i,:,:].unsqueeze(0)), dim=0).transpose(0,1)
            x = x.reshape(x.shape[0],-1)
            # print(f"x shape :{x.shape}")
            x = self.fc1(x)
            # print(f"x shape :{x.shape}")
            z = torch.cat((z,x), dim=1)
        z = torch.tensor(z.unsqueeze(1),device="cuda")
        z = F.softmax(z, dim=0)

        # print(f" z shape :{z.shape}")
        # print(f" enc_output shape :{enc_output.shape}")
        
        att = torch.matmul(z, enc_output.transpose(0,1))
        return att.transpose(0,1)


class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, in_features, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        # 嵌入层
        self.node_emb = nn.Linear(37486,M)
        self.dir_emb = nn.Linear(8,D)
        # self.dir_w = nn.Parameter(torch.zeros(K, D,32))
        self.taxiid_emb = nn.Linear(10356,32)

        # LocalGat层  nfeat（输入维度M）, nhid（隐藏层 默认为8）, nclass（输出维度M）, dropout（0.6）, alpha（0.2）, nheads（8）
        self.localgat = LocalGAT(M,8,M,0.6,0.2,1)

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size+32, 8)
        self.crition = nn.CrossEntropyLoss()

    def forward(self, input,num_minbatch,flag): # 
        input = input.to(device)
        # dec_input = dec_input.to(device)

        # print(f"input :{input.shape}") # [51, 14, 2])
        # print(f"input[:,:,0].cpu().numpy() :{input[:,:,0].cpu().numpy().shape}")  # :(20, 14)
        ## 嵌入层 进行node，方向dir嵌入
        node_one_hot = torch.tensor(get_dir_onehot(input[:,:,0].cpu().numpy(),37486),dtype=torch.float,device="cuda")
        node_emb = self.node_emb(node_one_hot)
        dir_one_hot = torch.tensor(get_dir_onehot(input[:,:,1].cpu().numpy(),8),dtype=torch.float,device="cuda")
        # print(enc_dir_one_hot.shape)
        dir_emb = self.dir_emb(dir_one_hot)

        # 出租车id
        taxiid = input[:,0,2].unsqueeze(1)
        taxiid = torch.tensor(get_onehot_taxi(taxiid.cpu().numpy(),10356),dtype=torch.float,device="cuda")
        # print(taxiid.shape) # 32,1
        taxiid_emb = self.taxiid_emb(taxiid).transpose(1,0)
        
        enc_node_emb = node_emb[:,:-4,:]
        enc_dir_emb = dir_emb[:,:-4,:]
        dec_node_emb = node_emb[:,-5:,:]
        dec_dir_emb = dir_emb[:,-5:,:]

        # localGAT 层
        # 先获得各个节点的邻居节点
        wh = self._get_local_spatial_bygat(input,node_emb)
        # dec_wh = self._get_local_spatial_bygat(dec_input,dec_node_emb)
        enc_wh = wh[:,:-4,:]
        dec_wh = wh[:,-5:,:]
        # print(f"dec_wh shape:{dec_wh.unsqueeze(1).shape}")
        # exit()

        # 合并
        enc_input = torch.cat((enc_node_emb, enc_dir_emb,enc_wh), dim=2)
        # print(enc_input.shape)  torch.Size([32, 10, 128])
        dec_input = torch.cat((dec_node_emb, dec_dir_emb,dec_wh), dim=2)
        # print(enc_input.shape)  torch.Size([32, 10, 128])

        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]

        seq_len, Batch_size, embedding_size = dec_input.size()

        outputs = torch.zeros(seq_len,Batch_size,K).to(device="cuda")  # 初始化一个张量，用来存储解码器每一步的输出
        target_len, _, _ = dec_input.size()
        # 首先通过编码器的最后一步输出得到 解码器的第一个隐含层 ， 以及将编码器的所有的输出层作为后续提取注意力
        enc_output, (h_0, c_0) = self.encoder(enc_input)  # s : [1, Batch_size, enc_hid_size ]

        # print(f"enc_output size :{enc_output.shape},h_0:{h_0.shape},c_0:{c_0.shape}") 
        """ 
            enc_output size :torch.Size([10, 20, 512]),h_0:torch.Size([2, 20, 512]),c_0:torch.Size([2, 20, 512])
        """
        # exit()
        # print(f"target_len:{target_len}")

        #############之前自己做的##################
        # for i in range(0, target_len):
        #     dec_output_i, s = self.decoder(enc_output, dec_input[i, :, :].unsqueeze(0), s)
        #     # next_input = output if random.random() < teacher_ratio elif y
        #     print(f"dec_output_i.shape :{dec_output_i.shape}")  # torch.Size([20, 8])
        #     outputs[i] = dec_output_i
        
        ##############论文复现decoder#################
        
        tempor_att = temporal_att(B) # 滑动时间注意力。
        H = enc_output # 需要更新H
        dec_input_ = dec_input[0, :, :].unsqueeze(0)
        # print(f"dec_input_ :{dec_input_.shape}")  # ([1, 20, 768])
        u = tempor_att(H,h_0,c_0) # 获得注意力
        # print(f"u shape:{u.shape}")
        # print(f"dec_input[1, :, :] shape:{dec_input[i, :, :].unsqueeze(0).shape}")
        dec_in = torch.cat((dec_input_,u), dim=2)  # 拼接
        dec_output_i, (h_0,c_0) = self.decoder(dec_in,h_0,c_0)
        
            # 更新H
        # print(f"H {H.shape}")
        H = H[1:,:,:] # 删除第一个
        # H= np.delete(H.cpu().numpy(),0,axis=0) # 删除第一个
        # print(f"H {H.shape}")
        # H = np.insert(H,9,h_0,axis=0)  #插入到最后一个
        H = torch.cat((H,dec_output_i),dim=0)
        # print(f"H {H.shape}")
        all_in = torch.cat((dec_output_i,taxiid_emb),dim=2)
        # dec_output_i = self.fc(dec_output_i)
        dec_output_i = self.fc(all_in)
        outputs[0] = dec_output_i
        for i in range(1, target_len):
            if flag == "train":
                # 计划采样：
                if random.random()< math.pow(1.05,num_minbatch):
                    dec_input_ = dec_input[i, :, :].unsqueeze(0)
                else:
                    # 得到 r ：
                    r = torch.argmax(F.softmax(dec_output_i,dim=2),axis=2)
                    # print(f"r.shape :{r.shape}")  # (1, 20)
                    prenodeid = input[:,9,0]
                    # print(f"prenodeid.shape :{prenodeid.shape}")  # torch.Size([20])
                    # 得到 prenodeid：
                    node_id = self._get_eid_targetnode(prenodeid.cpu().tolist(),r.squeeze(0).cpu().tolist())
                    # print(f"node_id.shape :{node_id.unsqueeze(1).unsqueeze(2).cpu().numpy().shape}")
                    # 得到 node dir 嵌入
                    node1_one_hot = torch.tensor(get_dir_onehot(node_id.unsqueeze(1).cpu().numpy(),37486),dtype=torch.float,device="cuda")
                    node1_emb = self.node_emb(node1_one_hot)
                    # print(f"node1_emb emb :{node1_emb.shape}")
                    dir1_one_hot = torch.tensor(get_dir_onehot(r.transpose(1,0).cpu().numpy(),8),dtype=torch.float,device="cuda")
                    dir1_emb = self.dir_emb(dir1_one_hot)
                    # print(f"dir_emb emb :{dir1_emb.shape}")
                    # 得到 spatial 嵌入
                    wh_node_emb = self._get_local_spatial_bygat(node_id.unsqueeze(1).unsqueeze(2),node1_emb)
                    # print(f"wh_node emb :{wh_node_emb.shape}")
                    # 合并 node  dir spatial
                    dec_input_ = torch.cat((node1_emb, dir1_emb,wh_node_emb), dim=2).transpose(1,0)
                    # print(f"dec_in  :{dec_input_.shape}")
            if flag == "test":
                # 得到 r ：
                r = torch.argmax(F.softmax(dec_output_i,dim=2),axis=2)
                # print(f"r.shape :{r.shape}")  # (1, 20)
                prenodeid = input[:,9,0]
                # print(f"prenodeid.shape :{prenodeid.shape}")  # torch.Size([20])
                # 得到 prenodeid：
                node_id = self._get_eid_targetnode(prenodeid.cpu().tolist(),r.squeeze(0).cpu().tolist())
                # print(f"node_id.shape :{node_id.unsqueeze(1).unsqueeze(2).cpu().numpy().shape}")
                # 得到 node dir 嵌入
                node1_one_hot = torch.tensor(get_dir_onehot(node_id.unsqueeze(1).cpu().numpy(),37486),dtype=torch.float,device="cuda")
                node1_emb = self.node_emb(node1_one_hot)
                # print(f"node1_emb emb :{node1_emb.shape}")
                dir1_one_hot = torch.tensor(get_dir_onehot(r.transpose(1,0).cpu().numpy(),8),dtype=torch.float,device="cuda")
                dir1_emb = self.dir_emb(dir1_one_hot)
                # print(f"dir_emb emb :{dir1_emb.shape}")
                # 得到 spatial 嵌入
                wh_node_emb = self._get_local_spatial_bygat(node_id.unsqueeze(1).unsqueeze(2),node1_emb)
                # print(f"wh_node emb :{wh_node_emb.shape}")
                # 合并 node  dir spatial
                dec_input_ = torch.cat((node1_emb, dir1_emb,wh_node_emb), dim=2).transpose(1,0)
                # print(f"dec_in  :{dec_input_.shape}")

            u = tempor_att(H,h_0,c_0) # 获得注意力
            # print(f"u shape:{u.shape}")
            # print(f"dec_input[1, :, :] shape:{dec_input[i, :, :].unsqueeze(0).shape}")
            dec_in = torch.cat((dec_input_,u), dim=2)  # 拼接
            dec_output_i, (h_0,c_0) = self.decoder(dec_in,h_0,c_0)
                
            # 更新H
            # print(f"H {H.shape}")
            H = H[1:,:,:] # 删除第一个
            # H= np.delete(H.cpu().numpy(),0,axis=0) # 删除第一个
            # print(f"H {H.shape}")
            # H = np.insert(H,9,h_0,axis=0)  #插入到最后一个
            H = torch.cat((H,dec_output_i),dim=0)
            # print(f"H {H.shape}")
            all_in = torch.cat((dec_output_i,taxiid_emb),dim=2)
            # dec_output_i = self.fc(dec_output_i)
            dec_output_i = self.fc(all_in)

            outputs[i] = dec_output_i
           
        outputs = outputs.permute(1, 0, 2)
        # print(f"output :{outputs.shape}")   # torch.Size([256, 5, 8])

        return outputs


    def _get_local_spatial_bygat(self,enc_input,enc_node_emb):
        wh = []
        for index1,sub_input in enumerate(enc_input[:,:,0].cpu().numpy()):
            temp = []
            for index2,i in enumerate(sub_input):
                h_j = [n for n in G.neighbors(i)]
                # print(h_j)
                # print(np.array(get_onehot_neighbor(h_j,37486)).shape)
                h_j = torch.tensor(np.array(get_onehot_neighbor(h_j,37486)),dtype=torch.float,device="cuda")
                h_j = self.node_emb(h_j)
                # print(h_j.shape)
                # print(enc_node_emb[index1,index2,:].shape)
                sub_wh = self.localgat(enc_node_emb[index1,index2,:],h_j)
                # print(len(sub_wh.detach().cpu().numpy().tolist()))
                # exit()
                temp.append(sub_wh.detach().cpu().numpy().tolist())
            wh.append(temp)
        wh = np.array(wh)
        # print(f"wh shape:{wh.shape}")
        wh = torch.tensor(wh,dtype=torch.float,device="cuda")
        return wh

    def _get_eid_targetnode(self,pre_nodeid,direction):
        # print(pre_nodeid)
        # print(direction)
        edge_dir_matrix = pd.DataFrame(edge_dir_emb,index=None)
        nodeid = []
        for preid,r in zip(pre_nodeid,direction):
            # print(f"pre_node:{pre_nodeid},r:{r}")
            edge_sample = edge_dir_matrix[edge_dir_matrix[1]==preid].to_numpy()  # 符合条件的edge
            res = list(map(abs,edge_sample[:,4]-r))
            if(len(res)==0):
                nodeid.append(preid)
            else:
                min_index = np.argmin(res)  # 判断方向是否符合，取最小的
                nodeid.append(edge_sample[min_index][2])

        return torch.tensor(np.array(nodeid),dtype=torch.float,device="cuda")



def get_eid(pre_nodeid,direction,edge_dir_emb):
    # print(pre_nodeid)
    # print(direction)
    edge_dir_matrix = pd.DataFrame(edge_dir_emb,index=None)
    edgeid = []
    for r in direction:
        # print(f"pre_node:{pre_nodeid},r:{r}")
        edge_sample = edge_dir_matrix[edge_dir_matrix[1]==pre_nodeid].to_numpy()  # 符合条件的edge
        res = list(map(abs,edge_sample[:,4]-r))
        if(len(res)==0):
            edgeid.append(edgeid[-1])
        # print(res)
        else:
            min_index = np.argmin(res)  # 判断方向是否符合，取最小的
            pre_nodeid = edge_sample[min_index][2]
            edgeid.append(edge_sample[min_index][0])
    return edgeid


def test_model(model,testX,test_segment_id,node_mask_testx,batch_size):
    model.eval()

    y_list = []
    pre_list = []
    all_num = 0
    # for index in range(0, testX.shape[0], batch_size):  # 开始 结束 步长
    for step, (X_batch,y_batch,y_eid,node_mask_testx) in enumerate(test_loader):
        # 清除网络先前的梯度值
        with torch.no_grad():
            # 初始化隐藏层数据 GRU需要注释掉
            model.encoder.hidden_cell = (torch.zeros(2, batch_size, 128).to("cuda"),
                                    torch.zeros(2, batch_size, 128).to("cuda"))


            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch = y_batch.to("cuda")
            node_mask_testx = node_mask_testx.to("cuda")

            # 实例化模型
            # print(X_batch)
            # y_pred = model(X_batch,X_batch[:,9,:].unsqueeze(1)) #
            y_pred = model(X_batch,step,"train")
            # print(y_pred.shape)

            # y_pred = F.softmax(y_pred,dim=2).mul(node_mask_testx)
            y_pred = F.softmax(y_pred,dim=2)

            testYPredict_segmentID = np.argmax(y_pred.cpu().detach(), axis=2) # one-hot解码
            # print(testYPredict_segmentID.shape)
            
            # print(f"x_batch :{X_batch[:,9,0].shape}")  # torch.Size([20])
            # print(f"Predict_segmentID :{testYPredict_segmentID.shape}")  # torch.Size([20, 5])
            e = []
            for i in range(X_batch[:,9,0].shape[0]):
                # print(int(X_batch[:,9,0][i].cpu().tolist()))
                # print(testYPredict_segmentID[i].cpu().tolist())
                temp = get_eid(int(X_batch[:,9,0][i].cpu().tolist()),testYPredict_segmentID[i].cpu().tolist(),edge_dir_emb)
                e.append(temp)
            e = np.array(e)
            all_num += get_MRK(y_eid,e,1)

            y_list = y_list + y_eid.cpu().numpy().flatten().tolist()
            # pre_list = pre_list + testYPredict_segmentID.cpu().numpy().flatten().tolist()
            pre_list = pre_list + e.flatten().tolist()

    # 计算正确率
    mrk_test = all_num/len(y_list)
    acc = accuracy_score(y_list,pre_list)  # 准确率相当于 AMR  
    # de = get_DE(y_list,pre_list)

    return acc,mrk_test

    # precision = precision_score(y_list,pre_list,average='weighted')
    # recall = recall_score(y_list,pre_list,average='weighted')
    # f1 = f1_score(y_list,pre_list,average='weighted')
    # print('测试集的正确率：', rate)
    # return acc,precision,recall,f1

def get_MRK(y,y_pre,k):
    total = 0
    for a,b in zip(y,y_pre):
        temp = 0
        for i in range(y.shape[1]):
            if a[i]==b[i]:
                temp+=1
        if temp >= k:
            total +=5
    return total


def train_model(model,train_loader,testX,test_segment_id,node_mask_testx,map_dic,optimizer,loss_function):

    all_eva_list = []

    eva_acc = []
    eva_precision = []
    eva_recall = []
    eva_f1 = []

    
    # print(trainX.shape)

    for j in range(epochs):
        all_num = 0
        # rate = 0

        y_list_train = []
        pre_list_train = []

        # for index in range(0, trainX.shape[0], batch_size):  # 开始 结束 步长
        for step, (X_batch,y_batch_id,y_eid,node_mask) in enumerate(train_loader):
            model.train()
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            # 初始化隐藏层数据

            hidden_cell = (torch.zeros(2, batch_size, 128).to("cuda"),
                                torch.zeros(2, batch_size, 128).to("cuda"))


            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch_id = y_batch_id.to("cuda")
            node_mask = node_mask.to("cuda")
            # 实例化模型
            y_pred = model(X_batch,step,"train")

            # 计算损失，反向传播梯度以及更新模型参数
            # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值

            # print(f"y_pred:{y_pred.shape}")  y_pred:torch.Size([252, 5, 8])
            # print(f"y_batch_id:{y_batch_id.shape}")  y_batch_id:torch.Size([252, 5])
            y_pred = y_pred.transpose(2,1)
            single_loss = loss_function(y_pred,y_batch_id)

            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            y_pred = y_pred.transpose(2,1)

            y_pred = F.softmax(y_pred,dim=2)

            Predict_segmentID = np.argmax(y_pred.detach().cpu(), axis=2) # one-hot解码
            
            
            # print(f"x_batch :{X_batch[:,9,0].shape}")  # torch.Size([20])
            # print(f"Predict_segmentID :{Predict_segmentID.shape}")  # torch.Size([20, 5])
            e = []
            for i in range(X_batch[:,9,0].shape[0]):
                # print(int(X_batch[:,9,0][i].cpu().tolist()))
                # print(Predict_segmentID[i].cpu().tolist())
                temp = get_eid(int(X_batch[:,9,0][i].cpu().tolist()),Predict_segmentID[i].cpu().tolist(),edge_dir_emb)
                e.append(temp)
            e = np.array(e)

            all_num += get_MRK(y_eid,e,1)

            # y_list_train = y_list_train + list(chain.from_iterable(y_batch))
            y_list_train = y_list_train + y_eid.cpu().numpy().flatten().tolist()
            # pre_list_train = pre_list_train + Predict_segmentID.cpu().numpy().flatten().tolist()
            pre_list_train = pre_list_train + e.flatten().tolist()


        # 计算MRK
        mrk_train = all_num/len(y_list_train)
        # 计算正确率
        # rate = rightNum / all_num  # 训练集
        rate = accuracy_score(y_list_train,pre_list_train)
        # acc,precision,recall,f1 = test_model(model, testX, test_segment_id, node_mask_testx,batch_size)  # 测试集
        acc,mrk_test = test_model(model, testX, test_segment_id, node_mask_testx,batch_size)  # 测试集

        # print('训练集 epoch：{} 的(train)正确率：{},loss : {},test_acc:{},precision:{},recall:{},f1:{}'.format(i, rate, single_loss.item(),acc,precision,recall,f1))
        print(f'训练集 epoch：{j} 的(train)正确率：{rate},loss : {single_loss.item()},train_mrk_1:{mrk_train},test_acc:{acc},test_mrk:{mrk_test},de:{0}')

        # 保存 eva
        # eva_acc.append(acc)
        # eva_precision.append(precision)
        # eva_recall.append(recall)
        # eva_f1.append(f1)

    all_eva_list.append(eva_acc)
    # all_eva_list.append(eva_precision)
    # all_eva_list.append(eva_recall)
    # all_eva_list.append(eva_f1)
    # np.save("./result/roadnet_eva_seq2seq+att-lstm-2layers-SH.npy", all_eva_list)

def get_datasetLoader(X,Y_id,y_eid,node_mask):
    torch_dataset = data.TensorDataset(X,Y_id,y_eid,node_mask)
    # 把dataset放入DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=20,             # 每批提取的数量
        shuffle=True,             # 要不要打乱数据（打乱比较好）
        num_workers=4             # 多少线程来读取数据
    )
    return loader

def get_nodemask(dec_i):
    print(dec_i.shape)
    mask = []
    for i  in dec_i:
        # print(node_mask[i])
        mask.append(node_mask[i])

    return mask

def get_nodemask_next5(dec_i):
    print(dec_i.shape)
    mask = []
    for i  in dec_i:
        temp = []
        for j in i:
            temp.append(node_mask[j])
        mask.append(temp)
    # print(np.array(mask).shape)
    return mask

if __name__ == '__main__':
    # 获取相关data trainX,trainY,validationX,validationY,testX,testY
    # trainX, trainY, validationX, validationY, testX, testY, train_segment_id, validation_segment_id, test_segment_id = get_data_roadsegment()
    trainX, validationX, testX, train_segment_id, validation_segment_id, test_segment_id,train_eid,validation_eid,test_eid = get_data_roadsegment_ce()
    node_mask = np.load("./1114/node_mask.npy",allow_pickle=True)
    node_mask = node_mask.tolist()
    # print(node_mask)

    node_mask_trainx = torch.tensor(get_nodemask_next5(trainX[:,-5:,0])) 
    node_mask_testx = torch.tensor(get_nodemask_next5(testX[:,-5:,0]))

    trainX = torch.tensor(trainX)
    # trainY = torch.tensor(trainY)
    validationX = torch.tensor(validationX)
    # validationY = torch.tensor(validationY)
    testX = torch.tensor(testX)
    # testY = torch.tensor(testY)
    train_segment_id = torch.tensor(train_segment_id)
    validation_segment_id = torch.tensor(validation_segment_id)
    test_segment_id = torch.tensor(test_segment_id)

    train_eid = torch.tensor(train_eid)
    validation_eid = torch.tensor(validation_eid)
    test_eid = torch.tensor(test_eid)

    # 先将数据转换为 dataset。
    print(trainX.shape,train_segment_id.shape,train_eid.shape,node_mask_trainx.shape)
    train_loader = get_datasetLoader(trainX,train_segment_id,train_eid,node_mask_trainx)
    # validation_loader = get_datasetLoader(validationX,validation_segment_id,node_mask_testx)
    test_loader = get_datasetLoader(testX,test_segment_id,test_eid,node_mask_testx)

       
    # input_size = trainX.shape[2]

    K = 8

    output_size = 8

    M = 256
    D = 256  # M 是node节点嵌入维度  D 方向嵌入维度
    hidden_size = 512
    B = 512

    epochs = 500
    batch_size = 20

    # 定义相关模型
    attn = Attention(hidden_size, hidden_size)  # enc_hid_dim, dec_hid_dim
    enc = Encoder(M+D+M, hidden_size)  # input_dim, enc_hid_dim, dec_hid_dim
    # dec = Decoder(M+D+M,output_size, hidden_size, hidden_size, attn)  # output_dim, enc_hid_dim, dec_hid_dim, attention
    dec = Decoder_row(M+D+M+B, hidden_size)

    device = "cuda"

    # 图 G
    A = np.load("./1114/edge_info.npy")
    G = nx.Graph()
    G.add_edges_from(A)

    edge_dir_emb = np.load("./1114/edge_dir_emb.npy")

    model = Seq2seq(enc, dec, M+D,hidden_size).to(device)
    # 优化器
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.5,weight_decay=0.8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # 损失函数
    loss_function = nn.CrossEntropyLoss().to(device="cuda")
    # loss_function = nn.MSELoss()

    map_dic = None
    # #
    train_model(model,train_loader,testX, test_segment_id,node_mask_testx,map_dic,optimizer,loss_function)


