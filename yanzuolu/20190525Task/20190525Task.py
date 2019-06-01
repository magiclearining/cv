import copy
import os
import time
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets.folder import pil_loader
from tqdm import tqdm

data_cat = ['train', 'valid']

def get_study_level_data(study_type):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        BASE_DIR = 'MURA-v1.1/%s/%s/' % (phase, study_type)
        patients = list(os.walk(BASE_DIR))[0][1]
        study_data[phase] = pd.DataFrame(columns=['path','Count','Label'])
        i = 0
        for patient in tqdm(patients):
            for study in os.listdir(BASE_DIR + patient):
                label = study_label[study.split('_')[1]]
                path = BASE_DIR + patient + '/' + study + '/'
                study_data[phase].loc[i] = [path, len(os.listdir(path)), label]
                i+=1
    return study_data

class MURA_Dataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample

def get_dataloaders(data, batch_size=8, study_level=False):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: MURA_Dataset(data[x], transform=data_transforms[x])
                     for x in data_cat}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size,
                                shuffle=True, num_workers=4) for x in data_cat}
    return dataloaders

def train_model(model, criterion, optimizer, dataloaders, scheduler, 
                dataset_sizes, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                print(i, end='\r')
                inputs = data['images'][0]
                labels = data['label'].type(torch.FloatTensor)
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                outputs = torch.mean(outputs)
                loss = criterion(outputs, labels, phase)
                running_loss += loss.data[0]
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    # plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc

def plot_training(costs, accs):
    '''
    Plots curve of Cost vs epochs and Accuracy vs epochs for 'train' and 'valid' sets during training
    '''
    train_acc = accs['train']
    valid_acc = accs['valid']
    train_cost = costs['train']
    valid_cost = costs['valid']
    epochs = range(len(train_acc))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1,)
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_cost)
    plt.plot(epochs, valid_cost)
    plt.legend(['train', 'valid'], loc='upper left')
    plt.title('Cost')
    
    plt.show()

def n_p(x):
    '''convert numpy float to Variable tensor float'''    
    return Variable(torch.cuda.FloatTensor([x]), requires_grad=False)

def get_count(df, cat):
    '''
    Returns number of images in a study type dataframe which are of abnormal or normal
    Args:
    df -- dataframe
    cat -- category, "positive" for abnormal and "negative" for normal
    '''
    return df[df['path'].str.contains(cat)]['Count'].sum()

class Loss(torch.nn.modules.Module):
    def __init__(self, Wt1, Wt0):
        super(Loss, self).__init__()
        self.Wt1 = Wt1
        self.Wt0 = Wt0
        
    def forward(self, inputs, targets, phase):
        loss = - (self.Wt1[phase] * targets * inputs.log() + self.Wt0[phase] * (1 - targets) * (1 - inputs).log())
        return loss

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # 3 * 224 * 224 - 16 * 112 * 112
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ) # 16 * 112 * 112 - 32 * 56 * 56
        self.out = nn.Sequential(
            nn.Linear(32 * 56 * 56, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        output = self.out(x)
        return output

acc = {}

for study_type in ['XR_ELBOW','XR_FINGER','XR_FOREARM',
'XR_HAND','XR_HUMERUS','XR_SHOULDER','XR_WRIST']:

    print(study_type, ": ")
    study_data = get_study_level_data(study_type)
    dataloaders = get_dataloaders(study_data, batch_size=1)
    dataset_sizes = {x: len(study_data[x]) for x in data_cat}

    # #### Build model
    # tai = total abnormal images, tni = total normal images
    tai = {x: get_count(study_data[x], 'positive') for x in data_cat}
    tni = {x: get_count(study_data[x], 'negative') for x in data_cat}
    Wt1 = {x: n_p(tni[x] / (tni[x] + tai[x])) for x in data_cat}
    Wt0 = {x: n_p(tai[x] / (tni[x] + tai[x])) for x in data_cat}

    print('tai:', tai)
    print('tni:', tni, '\n')
    print('Wt0 train:', Wt0['train'])
    print('Wt0 valid:', Wt0['valid'])
    print('Wt1 train:', Wt1['train'])
    print('Wt1 valid:', Wt1['valid'])

    # model = models.densenet169(pretrained=False)
    # model.classifier = nn.Sequential(OrderedDict([
    #     ('fc', nn.Linear(in_features=1664, out_features=1, bias=True)),
    #     ('sigmoid', nn.Sigmoid())
    # ]))
    # model = model.cuda()

    model = CNN()
    model = model.cuda()

    criterion = Loss(Wt1, Wt0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, verbose=True)

    model, acc[study_type] = train_model(model, criterion, optimizer, dataloaders, scheduler, dataset_sizes, num_epochs=5)
    print('-' * 10)
    print()

print(acc)
# {'XR_ELBOW': tensor(0.6139), 
# 'XR_FINGER': tensor(0.7143), 
# 'XR_FOREARM': tensor(0.5865), 
# 'XR_HAND': tensor(0.6168), 
# 'XR_HUMERUS': tensor(0.6222), 
# 'XR_SHOULDER': tensor(0.7268),
#  'XR_WRIST': tensor(0.7131)}