import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
#import save_gif
from PIL import Image
from models import Model, Model_v2, Model_v3
from visdom import Visdom
from dataloader.dataloader import Loader

def train():
    
    device = torch.device('cuda:0')
    model = Model_v3().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    visdom_server = Visdom(port=3387)

    train_set = Loader(mode = 'train')
    val_set = Loader(mode = 'val')
    show_set = Loader(mode = 'show')
    val_show_set = Loader(mode = 'val_show')

    data_loader = torch.utils.data.DataLoader(train_set , batch_size = 1 , shuffle = False , num_workers = 1)
    val_data_loader = torch.utils.data.DataLoader(val_set , batch_size = 1 , shuffle = False , num_workers = 1)
    show_loader = torch.utils.data.DataLoader(show_set , batch_size = 1 , shuffle = False , num_workers = 1)
    val_show_loader = torch.utils.data.DataLoader(val_show_set , batch_size = 1 , shuffle = False , num_workers = 1)
    
    print('The number of train dataloader:' , len(data_loader))
    print('The number of val dataloader:' , len(val_data_loader))
    print('The number of gif for training every 500 epochs' , len(show_loader))
    print('The number of gif for validation every 500 epochs:' , len(val_show_loader))

    for epoch in range(5000):
        #====================training
        
        total_loss = 0
        for i , (image , target) in enumerate(data_loader):
            image = image.to(device)
            target = target.to(device)
            predict, loss = model(image, target)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(data_loader)

        if total_loss<=5000:
          visdom_server.line([total_loss], [epoch], win='loss', env='version2',  update='append')
        else:
          visdom_server.line([5000], [epoch], win='loss', env='version2',  update='append')
        
        #=====================validation
        
        total_loss = 0         
        for i , (image , target) in enumerate(val_data_loader):
            image = image.to(device)
            target = target.to(device)
            predict , loss  =  model(image, target) 
            total_loss += loss.item()
        total_loss /= len(val_data_loader)

        if total_loss<=5000:
          visdom_server.line([total_loss], [epoch], win='val', env='version2',  update='append')
        else:
          visdom_server.line([5000], [epoch], win='val', env='version2',  update='append')
        
        #======================save model and gif for training and validation
        '''
        if epoch%500==0:
            path = f'./pretrained/{epoch}_pretrained.pth'
            torch.save(model.state_dict() , path)
            for idx , (image , target) in enumerate(show_loader):
                image = image.to(device)
                target = target.to(device)
                predict , _ = model(image , target)
                predict = torch.squeeze(predict) 
                predict = predict.cpu().detach().numpy()
                save_gif(predict , epoch , idx , mode = 'train')
           
            for idx , (image , target) in enumerate(val_show_loader):
                image = image.to(device)
                target = target.to(device)
                predict , _ = model(image , target)
                predict = torch.squeeze(predict)
                predict = predict.cpu().detach().numpy()
                save_gif(predict , epoch , idx , mode = 'val')

         '''    

if __name__ == '__main__':
    train()
