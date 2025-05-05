import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np
import yaml
import argparse

import sys
import os
import pickle
sys.path.append("/root/autodl-tmp/HSPS/IFC")
from HFPS_utils import Persis
from HFPS_utils.Hfps_dataset import HFPS_Dataset

@Persis.persistent_class
class TrainManager():
    def __init__(self, config_yaml: dict):
        self.config = config_yaml["IFC"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience_limit = self.config["train"]["e_stops"] #early stop
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.prepare_dataset(train_path= self.config["Data_In"]["train_path"],test_path= self.config["Data_In"]["test_path"])
        self.prepare_model()
       
        # 设置随机种子以确保代码的可重复性
        torch.manual_seed(42)  # 为CPU设置种子
        torch.cuda.manual_seed(42)  # 为当前CUDA设备设置种子
        np.random.seed(42)  
        
    def npz_load(self,npz_path):
        '''
            read npz and return two tensors
        '''
        data = np.load(npz_path)
        data_tensor = data['tensor'].astype(float)
        condition = data['Condition']
        data.close()
        return data_tensor,condition
    def prepare_dataset(self,train_path,test_path,test_batch=48):
        
        batch_size = self.config["train"]["batch_size"]
        # 加载npz文件
        train_data_tensor,train_condition = self.npz_load(train_path)
        test_data_tensor, test_condition = self.npz_load(test_path)
        # 假设 self.config 是已定义的配置字典
        slice_time = self.config["Data_In"].get("slice_time", 1)
        T_prime_max = self.config["Data_In"].get("T_prime_max", 1)
        sparse = self.config["Data_In"].get("sparse", 1)

        #N_steps,T_prime_max decide the time
        train_dataset = HFPS_Dataset(data_numpy= train_data_tensor,
                               class_list = train_condition,
                               N_steps= self.config["Data_In"]["seq_len"],slice_time= slice_time,T_prime_max=T_prime_max,sparse=sparse)
        dataset_length = len(train_dataset)
        train_size = int(dataset_length * 0.7)  # 例如：90% 作为训练集
        val_size = int(dataset_length * 0.2) 
        test_size = dataset_length - train_size - val_size  # 剩余的作为验证集

        train_dataset, val_dataset, test_dataset = random_split(train_dataset, [train_size, val_size,test_size])
        
        # test 单独
        test_dataset = HFPS_Dataset(data_numpy= test_data_tensor,
                               class_list = test_condition,
                               N_steps= self.config["Data_In"]["seq_len"],slice_time= slice_time,T_prime_max=T_prime_max,sparse=sparse) 
 
        valid_dataset = val_dataset
        # DataLoader
        self.train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset= test_dataset, batch_size = test_batch, shuffle=False, num_workers=4)

    def prepare_model(self):
        '''
        prepare encoder(nn.module) and classifer(nn.modulelist)
        '''
        from Baseline.IFC import IFC_model
        from Baseline.Classifer import Classifer_mlp
        encoder,classifers = None,None
        
        seq_len =  self.config["Data_In"]["seq_len"]
        spatial_x = self.config["Data_In"]["spatial_x"]
        spatial_y = self.config["Data_In"]["spatial_y"]
        latent_dims = self.config["encoder"]["paras"]["latent"]
        self.num_classifers = self.config["classfiers"]["numbers"]
        if self.config["encoder"]["type"]=="transformer":
            from Baseline.Encoders.Transformer_encoder import TransformerEncoderModel
            heads = self.config["encoder"]["paras"]["nheads"]
            layers = self.config["encoder"]["paras"]["num_layers"]
            encoder = TransformerEncoderModel(seq_len = seq_len,feature_size=spatial_x*spatial_y,latent_dim=latent_dims,nheads=heads,num_layers=layers)
            classifers = nn.ModuleList([Classifer_mlp(latent_dims) for _ in range(self.num_classifers)])
            self.model = IFC_model(encoder,classifers,ifc_type="Encoder_based").to(self.device)
        elif self.config["encoder"]["type"]=="operator":
            from Baseline.Encoders.Operator_encoder import OperatorEncoder
            encoder = OperatorEncoder(seq_len=seq_len,latent_dim=latent_dims,x=spatial_x,y=spatial_y)
            classifers = nn.ModuleList([Classifer_mlp(latent_dims) for _ in range(self.num_classifers)])
            self.model = IFC_model(encoder,classifers,ifc_type="Encoder_based").to(self.device)
        elif self.config["encoder"]["type"] == "cnn":
            from Baseline.Encoders.CNN_encoder import CNNEncoder
            encoder = CNNEncoder(seq_len=seq_len,latent_dim=latent_dims)
            classifers = nn.ModuleList([Classifer_mlp(latent_dims) for _ in range(self.num_classifers)])
            self.model = IFC_model(encoder,classifers,ifc_type="Encoder_based").to(self.device)
        elif self.config["encoder"]["type"] == "vit":
            '''
              actually vit has the classifer only, classifer has the encoder in it
            '''
            encoder = None
            from Baseline.Vit import Vit_classifer
            classifers = nn.ModuleList([Vit_classifer(num_frames=seq_len) for _ in range(self.num_classifers)])
            self.model = IFC_model(encoder,classifers,ifc_type="Vit_based").to(self.device)
        
        
        print("numbers of classifers",self.num_classifers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['train']["lr"])
        self.criterion = nn.CrossEntropyLoss()

    

    def train(self):
         # 初始化每个分类器的准确率统计
        self.correct_train = {f"Classifer_{i}_correct": 0 for i in range(self.num_classifers)}
        self.total_train = {f"Classifer_{i}_total":0  for i in range(self.num_classifers)}
        self.train_loss =[]
        self.valid_loss = []
        self.test_acc = 0
        for epoch in range(self.config["train"]['num_epochs']):
            self.model.train()
            
            for data, target,labels in self.train_loader:
                
                data, target,labels = data.to(self.device), target.to(self.device),labels.to(self.device)
                
                self.optimizer.zero_grad()
            
                outputs = self.model(data)
               
                total_loss = 0.0
                for i in range(self.num_classifers):  #self.num_classifers
                    
                    loss = self.criterion(outputs[:, i, :], labels[:, i])
          
                    total_loss += loss
                    # 计算准确率
                    _, predicted = torch.max(outputs[:, i, :], 1)  # 获取最大值的索引作为预测类别
                    self.correct_train[f"Classifer_{i}_correct"] = (predicted == labels[:, i]).sum().item()  # 更新对应分类器的正确预测数量
                    self.total_train[f"Classifer_{i}_total"] = labels.size(0)  # 更新对应分类器的样本数量

                # 反向传播
                total_loss.backward()
                self.optimizer.step()

            
            train_loss = total_loss / len(self.train_loader)
            self.train_loss.append(train_loss.item())
            
            valid_loss,valid_avg_acc = self.test(self.valid_loader)
            
            self.valid_loss.append(valid_loss.item())
            # 打印每个分类器的准确率
            for i in range(self.num_classifers):
                accuracy_train = 100 * self.correct_train[f"Classifer_{i}_correct"] / self.total_train[f"Classifer_{i}_total"]
                accuracy_valid = 100 * self.correct_valid[f"Classifer_{i}_correct"] / self.total_valid[f"Classifer_{i}_total"]
               
                print(f"train Classifier {i}: Accuracy = {accuracy_train:.2f}%")
                print(f"valid Classifier {i}: Accuracy = {accuracy_valid:.2f}%")

            print(f'Epoch {epoch+1}, Training Loss: {train_loss:.4f}, Valid_Avg_Acc{valid_avg_acc:.4f}')
            
            # Early stopping logic
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_model = self.model

                self.patience_counter = 0
                print("Validation loss decreased, resetting patience.")
            else:
                self.model = self.best_model # ensure best 
                self.patience_counter += 1
                print(f"Validation loss did not decrease, patience counter: {self.patience_counter}/{self.patience_limit}")
            
            if self.patience_counter >= self.patience_limit:
                print("Early stopping triggered.")
                test_loss,test_avg_acc = self.test(self.test_loader)
                self.test_avg_acc = test_avg_acc
                print(f"Test loss:{test_loss:4f} Test acc :{test_avg_acc:.4f}")
                break  # 跳出训练循环

    def test(self,loader):
        '''
            correct_valid the final [-1] is the test
        '''
        self.model.eval()
        total_loss = 0
        self.correct_valid = {f"Classifer_{i}_correct": 0 for i in range(self.num_classifers)}
        self.total_valid = {f"Classifer_{i}_total": 0 for i in range(self.num_classifers)}
        with torch.no_grad():
            for data, target,labels in loader:
                
                data, target,labels = data.to(self.device), target.to(self.device),labels.to(self.device)
                outputs = self.model(data)
                for i in range(self.num_classifers):#self.num_classifers
                    loss = self.criterion(outputs[:, i, :], labels[:, i])
                    total_loss += loss
                    # acc 
                    _, predicted = torch.max(outputs[:, i, :], 1)  # 获取最大值的索引作为预测类别
                    self.correct_valid[f"Classifer_{i}_correct"] = (predicted == labels[:, i]).sum().item()  # 更新对应分类器的正确预测数量
                    self.total_valid[f"Classifer_{i}_total"] = labels.size(0)  # 更新对应分类器的总样本数量

        average_loss = total_loss / len(loader)
        average_accuracy = 0
        for i in range(self.num_classifers):
            accuracy = 100 * self.correct_valid[f"Classifer_{i}_correct"] / self.total_valid[f"Classifer_{i}_total"] if self.total_valid[f"Classifer_{i}_total"] != 0 else 0
            average_accuracy += accuracy
            print(f"Classifier {i}: Accuracy = {accuracy:.2f}%")

        # averge of classifers
        average_accuracy /= self.num_classifers
        print(f"Average Accuracy test: {average_accuracy:.2f}%")
        
        return average_loss,average_accuracy

    def save(self):
        path = self.config["Save"]["path"]+".pkl"
        directory = os.path.dirname(path)
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.model =None # need to reload
        # dir
        if not os.path.exists(directory):
            os.makedirs(directory)
        # save
        with open(path, 'wb') as outp:
            pickle.dump(self, outp)
            
    def pipeline(self):
        self.train()
        self.save()
        print("done")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a YAML configuration file')
    parser.add_argument('--config_yaml', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    # 读取 YAML 配置文件
    with open(args.config_yaml, 'r') as file:
        config = yaml.safe_load(file)
        print(config)
    t_m = TrainManager(config)
    t_m.pipeline()