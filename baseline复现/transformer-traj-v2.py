import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from LocalGat import  LocalGAT
import networkx as nx
import Levenshtein


#  位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

#  获取数据
def get_data_roadsegment_ce():
    # 加载数据

    dataset = np.load("./1114/traj_dataset_window5_5fea_taxiid.npy",allow_pickle=True)

    rng = np.random.default_rng(12345)
    # 打乱数据顺序
    rng.shuffle(dataset)
    # dataset = dataset_emb

    # 划分数据集，6:2:2
    train_size = int(len(dataset) * 0.2)
    trainlist = dataset[:train_size]  # 训练集

    validationlist = dataset[int(len(dataset) * 0.6):int(len(dataset) * 0.65)]  # 验证集
    testlist = dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.85)]  # 测试集

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
    # length = 15  # 每个样本的长度
    # look_back = length - 1
    next5 = -5

    look_back = 10

    """
    也就是说我需要在trainx中放入前14个，然后使用这个14个中后5个当做decoder的输入。
    """

    trainX = trainlist[:, :, (L)]
    train_segment_id = trainlist[:, next5:, 4].astype(int)
    train_eid = trainlist[:, next5:, 0].astype(int)
    # trainY = get_onehot(train_segment_id, 8)  # onehot

    # trainX = trainlist[:, :look_back, :]
    # train_segment_id = trainlist[:, look_back:look_back + 1,0].astype(int)
    # trainY = get_onehot(train_segment_id, 118)  # onehot

    # print(train_segment_id.shape)

    validationX = validationlist[:, :, (L)]
    validation_segment_id = validationlist[:, next5:, 4].astype(int)
    validation_eid = validationlist[:, next5:, 0].astype(int)
    # validationY = get_onehot(validation_segment_id, 8)  # onehot

    # validationX = validationlist[:, :look_back, :]
    # validation_segment_id = validationlist[:, look_back:look_back + 1, 0].astype(int)
    # validationY = get_onehot(validation_segment_id, 118)  # onehot

    #
    testX = testlist[:, :, (L)]
    test_segment_id = testlist[:, next5:, 4]
    test_eid = testlist[:, next5:, 0].astype(int)
    # testY = get_onehot(test_segment_id, 8)  # onehot

    # testX = testlist[:, :look_back, :]
    # test_segment_id = testlist[:, look_back:look_back + 1, 0].astype(int)
    # testY = get_onehot(test_segment_id, 118)  # onehot

    return trainX, validationX, testX, train_segment_id, validation_segment_id, test_segment_id,train_eid,validation_eid,test_eid


class Nettarj_Transformer(nn.Module):
    def __init__(self):                                                                                                                                                                                                                                                                                                           
        super(Nettarj_Transformer,self).__init__()

        # 嵌入层：
        self.node_emb = nn.Linear(Node_num,M)
        self.dir_emb = nn.Linear(K,D)
        # self.dir_w = nn.Parameter(torch.zeros(K, D,32))
        self.taxiid_emb = nn.Linear(Taxi_num,32)
        # LocalGat层  nfeat（输入维度M）, nhid（隐藏层 默认为8）, nclass（输出维度M）, dropout（0.6）, alpha（0.2）, nheads（8）
        self.localgat = LocalGAT(M,8,M,0.6,0.2,1)
        
        self.d_model=128

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(768,dropout=0)

        # 定义Transformer
        self.transformer = nn.Transformer(768, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(768+32, K)

    def forward(self, input):
        # print(input.shape)  torch.Size([20, 15, 3])
        # 生成mask
        # print(input[:,10:,1].size())   torch.Size([20, 5])
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(input[:,9:14,1].size()[-1]).to(device)
        # src_key_padding_mask = Nettarj_Transformer.get_key_padding_mask(input[:,:10,1]).to(device)
        # tgt_key_padding_mask = Nettarj_Transformer.get_key_padding_mask(input[:,10:,1]).to(device)

        # print(f"tgt_mask:{tgt_mask}")

        # 对src和tgt进行编码
        ## 嵌入层 进行node，方向dir嵌入
        node_one_hot = torch.tensor(get_dir_onehot(input[:,:,0].cpu().numpy(),Node_num),dtype=torch.float,device="cuda")
        node_emb = self.node_emb(node_one_hot)
        dir_one_hot = torch.tensor(get_dir_onehot(input[:,:,1].cpu().numpy(),K),dtype=torch.float,device="cuda")
        # print(enc_dir_one_hot.shape)
        dir_emb = self.dir_emb(dir_one_hot)

        # 出租车id
        taxiid = input[:,0,2].unsqueeze(1)
        taxiid = torch.tensor(get_onehot_taxi(taxiid.cpu().numpy(),Taxi_num),dtype=torch.float,device="cuda")
        # print(taxiid.shape) # 32,1
        taxiid_emb = self.taxiid_emb(taxiid).transpose(1,0).repeat(5,1,1)
        # print(f"taxiid_emb.shape：{taxiid_emb.shape}")
        
        enc_node_emb = node_emb[:,:10,:]
        enc_dir_emb = dir_emb[:,:10,:]
        dec_node_emb = node_emb[:,9:14,:]
        dec_dir_emb = dir_emb[:,9:14,:]

        # localGAT 层
        # 先获得各个节点的邻居节点
        wh = self._get_local_spatial_bygat(input,node_emb)
        # dec_wh = self._get_local_spatial_bygat(dec_input,dec_node_emb)
        enc_wh = wh[:,:10,:]
        dec_wh = wh[:,9:14,:]
        # print(f"dec_wh shape:{dec_wh.unsqueeze(1).shape}")
        # exit()

        # 合并
        enc_input = torch.cat((enc_node_emb, enc_dir_emb,enc_wh), dim=2)
        # print(enc_input.shape)  torch.Size([32, 10, 128])
        dec_input = torch.cat((dec_node_emb, dec_dir_emb,dec_wh), dim=2)
        # print(enc_input.shape)  torch.Size([32, 10, 128])

        enc_input = enc_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]
        dec_input = dec_input.permute(1, 0, 2)  # [seq_len,Batch_size,embedding_size]

        src = enc_input
        tgt = dec_input  # src:torch.Size([10, 20, 768]),tat:torch.Size([5, 20, 768])

        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src).to(device)
        tgt = self.positional_encoding(tgt).to(device)

        # print(f"src:{src.shape},tat:{tgt.shape}")

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=None,
                               tgt_key_padding_mask=None)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        # print(f"out:{out.shape}")

        out = torch.cat((out,taxiid_emb),dim=2)

        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        # print(tokens.size())
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == -1] = -torch.inf
        return key_padding_mask

    def _get_local_spatial_bygat(self,enc_input,enc_node_emb):
        wh = []
        for index1,sub_input in enumerate(enc_input[:,:,0].cpu().numpy()):
            temp = []
            for index2,i in enumerate(sub_input):
                h_j = [n for n in G.neighbors(i)]
                # print(h_j)
                # print(np.array(get_onehot_neighbor(h_j,37486)).shape)
                h_j = torch.tensor(np.array(get_onehot_neighbor(h_j,Node_num)),dtype=torch.float,device="cuda")
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

#######修改softmax函数########
def _logsoftmax_self(x,mask):
    # 计算每行的最大值
    row_max = torch.max(x,dim=2,keepdim=True).values
    # print(row_max.shape)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    x_exp = x.exp()  # m * n
    # print(f"x_exp :{x_exp}")
    x_exp_mask = x_exp*mask
    partition = x_exp_mask.sum(dim=2, keepdim=True)  # 按列累加, m * 1
    # return torch.log((x_exp / partition))  # 广播机制, [m * n] / [m * 1] = [m * n]
    return x_exp / partition

###############路网约束的交叉熵损失函数####################
def my_celoss(output,target,mask):
    # output = output.transpose(2,1)
    
    log_softmax=_logsoftmax_self(output,mask)
    # print(f"log softmax :{log_softmax}")
    bsize =target.shape[0]
    loss=0
    
    #由于batchsize一般都不会很大，因此该for循环花费时间很少
    for b in range(bsize):
        # 找到target
        tar = target[b,:].unsqueeze(0).transpose(1,0)
        # print(f"tar shape:{tar.shape}")
        pre = log_softmax[b,:,:]
        # print(f"pre shape:{pre.shape}")
        # print(f"pre before :{pre}")
        pre = torch.gather(-pre,dim=1,index=tar)
        # pre = pre.gather(axis=0,tar)

        # print(f"pre :{pre}")
        # print(f"mead pre :{torch.mean(pre)}")

        current_loss=torch.mean(pre)
        loss+=current_loss
    return loss/bsize




###############相关函数##################################
def get_datasetLoader(X,Y_id,y_eid,node_mask):
    torch_dataset = data.TensorDataset(X,Y_id,y_eid,node_mask)
    # 把dataset放入DataLoader
    loader = data.DataLoader(
        dataset=torch_dataset,
        batch_size=20,             # 每批提取的数量
        shuffle=True,             # 要不要打乱数据（打乱比较好）
        num_workers=32             # 多少线程来读取数据
    )
    return loader

def get_nodemask_next5(dec_i):
    # print(dec_i.shape)
    mask = []
    for i  in dec_i:
        temp = []
        for j in i:
            if(j not in node_mask.keys()):
                temp.append([0,0,0,0,0,0,0,0])
            else:
                temp.append(node_mask[j])
        mask.append(temp)
    # print(np.array(mask).shape)
    return mask

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

def get_onehot_neighbor(lis,num):
    onehot_encoded = []
    for value in lis:
        letter = [0 for _ in range(num)]
        letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded

# 计算编辑距离
def get_DE(s1,s2):
    de = 0
    for i in range(len(s1)):   
        de += Levenshtein.distance(str(s1[i]),str(s2[i]))

    return de/len(s1)


#######################测试函数################################################
def test_model(model):
    model = model.eval()
    for j in range(epochs):
        all_num = 0

        y_list_test = []
        pre_list_test = []
        for step, (X_batch,y_batch_id,y_eid,node_mask) in enumerate(test_loader):
            with torch.no_grad():

                X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
                y_batch_id = y_batch_id.to("cuda")
                node_mask = node_mask.to("cuda")
                # 实例化模型
                y_pred = model(X_batch)
                y_pred = model.predictor(y_pred).transpose(0,1)

                y_pred = F.softmax(y_pred,dim=2)
                # y_pred = _logsoftmax_self(y_pred,node_mask)

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

                y_list_test = y_list_test + y_eid.cpu().numpy().flatten().tolist()
                pre_list_test = pre_list_test + e.flatten().tolist()
        
        # 计算MRK
        mrk_test = all_num/len(y_list_test)
        # 计算正确率
        # rate = rightNum / all_num  # 训练集
        rate = accuracy_score(y_list_test,pre_list_test)
        de = get_DE(y_list_test,pre_list_test)

        return rate,mrk_test,de
        

#######################训练函数################################################
def train_model(model,train_loader,optimizer,loss_function):
    for j in range(epochs):
        all_num = 0
        # rate = 0

        y_list_train = []
        pre_list_train = []
        for step, (X_batch,y_batch_id,y_eid,node_mask) in enumerate(train_loader):
            model.train()
            
            # 初始化隐藏层数据

            X_batch = torch.tensor(X_batch, dtype=torch.float).to("cuda")
            y_batch_id = y_batch_id.to("cuda")
            node_mask = node_mask.to("cuda")
            # 实例化模型
            y_pred = model(X_batch)
            y_pred = model.predictor(y_pred).transpose(0,1)
            # print(f"y_pred:{y_pred}")  # torch.Size([20, 5, 8])

            # 计算损失，反向传播梯度以及更新模型参数
            # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值
            y_pred = y_pred.transpose(2,1)
            # print(f"y_pred:{y_pred}")  # y_pred:torch.Size([20, 5, 8])
            # print(f"y_batch_id:{y_batch_id}")  # y_batch_id:torch.Size([20, 5])

            # single_loss = my_celoss(y_pred,y_batch_id,node_mask)
            single_loss = loss_function(y_pred,y_batch_id)
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            y_pred = y_pred.transpose(2,1)

            y_pred = F.softmax(y_pred,dim=2)
            # y_pred = _logsoftmax_self(y_pred,node_mask)

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
        de_train = get_DE(y_list_train,pre_list_train)
        acc,mrk_test,de_test = test_model(model)  # 测试集

        print('训练集 epoch：{},loss:{},train_accrate：{},mrk_train:{},de_train:{},test_accrate:{},mrk_test:{},de_test:{}'.format(j,single_loss.item(),rate, mrk_train,de_train,acc,mrk_test,de_test))
        # print(f'训练集 epoch：{j} 的(train)正确率：{rate},loss : {single_loss.item()},train_mrk_1:{mrk_train},test_acc:{acc},test_mrk:{mrk_test},de:{0}')



if __name__ == '__main__':
    # 获取数据集
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

    K = 8
    Node_num = 37486
    Taxi_num = 10356

    output_size = 8
    epochs = 200
    batch_size = 20
    device = "cuda"

    M = 256
    D = 256  # M 是node节点嵌入维度  D 方向嵌入维度
    hidden_size = 512
    B = 512

    edge_dir_emb = np.load("./1114/edge_dir_emb.npy")

    # 图 G
    A = np.load("./1114/edge_info.npy")
    G = nx.Graph()
    G.add_edges_from(A)


    # 定义模型
    model = Nettarj_Transformer().to(device)

    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_model(model,train_loader,optimizer,loss_function)