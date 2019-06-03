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

from torchnet import meter

data_cat = ['train', 'valid']

def get_study_level_data(study_type):

    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    for phase in data_cat:
        columns = ['img_path']
        BASE_DIR = 'MURA-v1.1/%s_image_paths.csv' % (phase)
        data = pd.read_csv(BASE_DIR,names=columns)
        study_data[phase] = pd.DataFrame(columns=['path','Label'])
        i = 0
        for img in tqdm(data['img_path']):
            path_type = img.split('/')[2]
            if path_type == study_type:
                label = study_label[img.split('_')[2].split('/')[0]]
                path = img
                study_data[phase].loc[i] = [path, label]
                i += 1
    return study_data

class MURA_Dataset(Dataset):
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        images = self.transform(pil_loader(study_path))
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
    best_kappa = 0.0
    costs = {x:[] for x in data_cat} # for storing costs per epoch
    accs = {x:[] for x in data_cat} # for storing accuracies per epoch
    print('Train batches:', len(dataloaders['train']))
    print('Valid batches:', len(dataloaders['valid']), '\n')
    for epoch in range(num_epochs):
        confusion_matrix = {x: meter.ConfusionMeter(2, normalized=False) 
                             for x in data_cat}
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in data_cat:
            if phase == 'train':
                torch.set_grad_enabled(True)
            if phase == 'valid':
                torch.set_grad_enabled(False)
            model.train(phase=='train')
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for i, data in enumerate(dataloaders[phase]):
                # get the inputs
                inputs = data['images']
                labels = data['label']
                # wrap them in Variable
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                preds = torch.max(outputs, 1)[1]
                # print(outputs)
                # print(preds)
                # print(labels)
                running_corrects += torch.sum(preds == labels.data)
                confusion_matrix[phase].add(preds, labels.data)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.type(torch.FloatTensor) / dataset_sizes[phase]
            costs[phase].append(epoch_loss)
            accs[phase].append(epoch_acc)
            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            mat = confusion_matrix[phase].value()
            a1 = mat[0][0] + mat[1][0]
            b1 = mat[0][0] + mat[0][1]
            a2 = mat[0][1] + mat[1][1]
            b2 = mat[1][0] + mat[1][1]
            s = a1 + a2
            p0 = (mat[0][0] + mat[1][1]) / s
            pe = (a1 * b1 + a2 * b2) / (s * s)
            k = (p0 - pe) / (1 - pe)
            print('Confusion Meter:\n', mat)
            print("kappa: ", k)
            print()

            # deep copy the model
            if phase == 'valid':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_kappa = k
                    best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('Time elapsed: {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))
    print('Best valid kappa: {:4f}'.format(best_kappa))
    # plot_training(costs, accs)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_kappa