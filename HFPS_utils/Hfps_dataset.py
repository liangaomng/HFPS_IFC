
import torch
from torch.utils.data import Dataset
import numpy as np

def class_label(condition,sample_t):
    '''
        # angle,radius,vel,T_min 
        return "velocity,angle,size,how long ago?"
    '''
    #angle,radius,vel,T_min

    angle = -condition[0] # 加个负号
    radius = condition[1]
    vel = condition[2]
    T_min = condition [3] #like 201 ,we need to convert the physic time
    How_long_ago = (T_min + sample_t)* 0.05 # 0.05 is second
  
    if vel <= 0.14:
        label_1 = 0
    elif vel> 0.14 and vel<= 0.22:
        label_1 = 1
    elif vel >0.22: 
        label_1 = 2
        
    if angle<=10: #position--augment rotate
        label_2 = 0
    elif angle>10 and angle<=20:
        label_2 = 1
    elif angle>20:
        label_2 = 2
        
    if radius<=14:
        label_3 =0
    elif radius >14 and radius<=22:
        label_3 =1
    elif radius>22:
        label_3 =2
    
    if How_long_ago <= 11.35: #how long ago
        label_4 = 0
    elif How_long_ago>11.35 and How_long_ago<=12.7: 
        label_4 = 1
    elif How_long_ago> 12.7 :
        label_4 = 2
        
    return [label_1,label_2,label_3,label_4]
def labels_to_onehot(labels, num_classes=3):
    '''
        indeed, we use the cross-entropy loss,we do not use this one hot
    '''
    one_hot_labels = np.zeros((len(labels), num_classes))
    # 在对应的类别位置上设置 1
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1
    return one_hot_labels  
class HFPS_Dataset(Dataset):
    def __init__(self, data_numpy, class_list:list,N_steps=10,slice_time=1,T_prime_max=60,sparse=1):
        """
        Args:
            data_numpy:[cases,t,:,;]
            class_list:list of cases classes like[re,angle,radius,time=100]
        """
        self.data_numpy = torch.from_numpy(data_numpy).float()
        print("raw_data_numpy",self.data_numpy.shape)
        self.condition_tensor = torch.tensor(class_list)
        self.N = N_steps
        self.slice_time = slice_time# 时间步
        self.T_prime_max = T_prime_max #物理时间要乘0.05+10.05
        self.sparse = sparse

    def __len__(self):
        """
        返回数据集中的样本总数。
        """
        return self.data_numpy.shape[0]

    def __getitem__(self, idx):
        """
        根据给定的索引 idx 返回一个样本。
        Args:
            idx (int): 样本的索引
            03_01: every time step = 0.05s, the dataset have 80 steps [b,80,128,512]
        Returns:
            tensor: 转换后的样本
        """
    
        t_sample = np.random.randint(0, self.T_prime_max)  # represent the t‘ --3s,[0,T_prime_max)
      
        data = self.data_numpy[idx,t_sample:t_sample+(self.N*self.slice_time):self.slice_time,::self.sparse,::self.sparse]#
        # #归一化
        # # 首先对您感兴趣的第一个维度计算最小值，然后是第二个维度。
        # min_val = data.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        # max_val = data.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        # 对整个 sample 做归一化，而不是每一帧--不破坏时间序列的物理差异--0505
        min_val = data.min()
        max_val = data.max()
        data = (data - min_val) / (max_val - min_val + 1e-8)
        conditions = self.condition_tensor[idx] #list 
        class_labels = class_label(conditions,t_sample)

        labels = torch.tensor(class_labels, dtype=torch.long)  # 创建一个标签
       
        return data,conditions,labels
      
if __name__ =="__main__":
   data_npz = "/root/autodl-tmp/HSPS/IFC/Dataset/216_cases/03_01Data.npz"
  # 加载npz文件
   data = np.load(data_npz)
   # 访问保存的数组
   tensor = data['tensor'].float()
   condition = data['Condition']
   # 使用完数据后，确保关闭文件
   data.close()

   dataset = HFPS_Dataset(data_numpy=tensor,class_list = condition) 
