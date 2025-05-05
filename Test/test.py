### this file is to get the test results

import pickle
import torch
import numpy as np
import yaml
import os
import sys
sys.path.append("/root/autodl-tmp/HSPS/IFC/")
class Test_method():
   
   def __init__(self,checkpoint_path:str,task:list):
      
      '''
         checkpoint_pth: to read
         task: 任务的分类器--not done
      '''
      self.classifers = task
      # load
      with open(checkpoint_path, 'rb') as file:
         loaded_object = pickle.load(file)
      self.model = loaded_object.best_model
      loaded_object.prepare_dataset(train_path = loaded_object.config["Data_In"]["train_path"],
                                                  test_path= loaded_object.config["Data_In"]["test_path"],worker=0)
      self.loader = loaded_object.test_loader
      self.checkpoint = loaded_object
      self.results_pth = "/root/autodl-tmp/HSPS/IFC/" + loaded_object.config["Save"]["path"] + "_results.yaml"
      print("init")
      
   @property
   def results_pth(self):
      # 如果已经手动设置过，则返回手动设置的值，否则计算默认值
    
      return self._results_pth

   @results_pth.setter
   def results_pth(self, pth):
        # 重新设置
      self._results_pth = pth
      
   def MC_test(self,Uncer_MC=50,T_prime_mc=50):
      MC_test = Uncer_MC # 50  big for uncertanity 
      MC_test_DICT= {"C1":[],"C2":[],"C3":[],"C4":[]}
      T_prime_mc = T_prime_mc # for the T prime mc
      dataloader = self.loader
      model = self.model
      model = model.train()
      for Mc in range (MC_test):
         
         predictions = [[] for _ in range(4)]  # 为每个分类器预先创建列表
         true_labels_list = [[] for _ in range(4)]  # 修改变量名以避免冲突
         with torch.no_grad():
            for mc in range(T_prime_mc): # T_prime
         
               for data, conditions, label in dataloader:
                  data = data.cuda()
                  outputs = model(data)
                  #latent = model.latent  # [b, 512]

                  for i in self.classifers:  # calssifer -4 
                     _, predicted = torch.max(outputs[:, i, :], 1)  # 获取最大值的索引作为预测类别
                     predictions[i].extend(predicted.cpu().numpy())
                     true_labels_list[i].extend(label[:, i].cpu().numpy())

            # 绘制每个分类器的预测结果的柱状图
            num_classifiers = len(predictions)

            for i in self.classifers :

               #将列表转换为NumPy数组以便更容易地进行运算
               preds = np.array(predictions[i])
               labels = np.array(true_labels_list[i])
               
               # 计算准确率
               accuracy = np.mean(preds == labels) * 100
            
               MC_test_DICT[f"C{i+1}"].append(accuracy)
               print(f'In MC {Mc}Classifier {i+1} Predicted acc is {accuracy:.2f}')
      
      results = {"Uncer_MC":Uncer_MC,"T_prime_MC":T_prime_mc}
      for i in self.classifers :  # 对于选定的分类器
         mean = np.mean(MC_test_DICT[f"C{i+1}"])  # 计算均值
         std = np.std(MC_test_DICT[f"C{i+1}"])     # 计算标准差
         results[f"C{i+1}"] = {"mean": float(mean), "std": float(std)}
         print(f'Classifier {i+1} Accuracy: {mean:.4f} std: {std:.4f}')

      # 将结果保存到 YAML 文件中
      with open(f"{self.results_pth}", "w", encoding="utf-8") as file:
         yaml.dump(results, file, default_flow_style=False, allow_unicode=True)
         print("save_done")
 