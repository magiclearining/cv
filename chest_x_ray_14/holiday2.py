
# coding: utf-8

# 目前思路是先读取csv文件，将图片路径和label对应为pandas的df，划分前10个压缩包的为训练集，后2个为测试集，解压训练集放到images_train,训练集images_test,手动查看每个集有多少图片
# 然后由于我太菜了，代码部分借鉴自lyz大神的MURA作业, 谢谢大佬，还有部分询问自欢荣师兄，谢谢师兄

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# 产生数据的dataframe
info = pd.read_csv('Data_Entry_2017.csv')

label = np.zeros([info.size,14])
label_str = ['Atelectasis' ,'Cardiomegaly','Effusion','Infiltration','Mass', 'Nodule', 'Pneumonia',
             'Pneumothorax', 'Consolidation','Edema', 'Emphysema', 'Fibrosis','Pleural_Thickening','Hernia']
label = pd.DataFrame(label, columns=label_str)

info = pd.concat([info, label], axis=1, join='inner')
for i in range(14):
    info.loc[info['Finding Labels'].str.contains(label_str[i]),label_str[i]] = 1 


# In[3]:


num_trains = 94999
num_tests = 17121
data = {}
data['train'] = info[:num_trains]
data['test'] = info[num_trains:].reset_index()


# In[4]:


import torch
import os
from torch.nn import DataParallel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from torch import nn
import torchvision.models as models

batch_size = 32
cuda = True
test_every = 1000
epochs = 20
para_path = 'params.pkl'
reload_para = False
save_para = True


# In[5]:


class X_Ray_Dataset(Dataset):
    
    def __init__(self, df, process_type, transform=None):
        self.df = df
        self.transform = transform
        self.process_type = process_type
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        im_path = self.df['Image Index'][idx]
        im_path = os.path.join(os.getcwd(),'images_'+self.process_type, im_path)
        label = [self.df[label_str[i]][idx] for i in range(14)]
        label= torch.from_numpy(np.array(label))
        image = self.transform(pil_loader(im_path))
        sample = {'image': image, 'label': label}
        return sample


# In[6]:


def get_dataloaders(data, batch_size=batch_size):

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((1024,1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: X_Ray_Dataset(data[x], x, transform=data_transforms[x])
                     for x in {'train', 'test'}}   
    train_dataloader = DataLoader(dataset=image_datasets['train'], batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=image_datasets['test'], batch_size=1, shuffle=False)
    
    return train_dataloader, test_dataloader


#------------------net def----------------------------------------------

model = models.densenet121(pretrained=False)
outlayer = nn.Sequential(nn.Linear(in_features=1024, out_features=14, bias=True), nn.Sigmoid())
model.classifier = outlayer

#------------------net def end-----------------------------------------------

device_for_data = torch.device('cuda:0' if cuda else 'cpu')
device_for_model = torch.device('cuda' if cuda else 'cpu')

if reload_para:
    model.load_state_dict(torch.load(para_path))
model = DataParallel(model)
model.to(device_for_model)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[9]:


train_dataloader, test_dataloader = get_dataloaders(data, batch_size=16)
for epoch in range(epochs):
    for i, train_data in enumerate(train_dataloader):
        model.train()
    
        inputs = train_data['image'].to(device_for_data) #16,1,h,w
        labels = train_data['label'].to(device_for_data) #16,1,14
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.float(), labels.float())
        loss.backward()
        optimizer.step()
    
        if i % test_every == 0:
            model.eval()
            acc = 0.0
            for j, test_data in enumerate(test_dataloader):
                inputs_test = test_data['image'].to(device_for_data)
                labels_test = test_data['label'].to(device_for_data)
                labels_prediction = model(inputs_test)
                labels_prediction = torch.round(labels_prediction, out=None)
                if torch.equal(labels_test.double(), labels_prediction.double()):
                    acc = acc + 1.0
            acc = acc / float(num_tests)
            print('epoch{}   batch{}  Loss: {:.4f} Acc: {:.4f}'.format(epoch, i, loss, acc))


if save_para:
    torch.save(model.state_dict(), para_path)

