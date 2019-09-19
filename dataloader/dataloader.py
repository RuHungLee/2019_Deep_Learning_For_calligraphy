import  sys
from PIL import  Image
import  numpy as np
import  pandas as pd
import  os
import  torch
from torchvision.transforms.transforms import ToTensor
from torch.utils.data import  Dataset

path = os.path.join(os.getcwd())
np.set_printoptions(threshold=sys.maxsize)

def loader(path , mode='image'):
  if mode == 'image':
    image = Image.open(path).convert('RGB')
    image = np.array(image)
    image = ToTensor()(image)

    return image
  elif mode == 'cor':
    cor = pd.read_csv(path)
    cor = cor.to_numpy()
    cor = np.pad(cor , [(0, 1024-cor.shape[0]) , (0  , 0)] , mode='constant' , constant_values = 0)
    cor = torch.from_numpy(cor)
    cor = cor.type(torch.float32)

    return cor

def readfile(pair_file):
  image_cor_paths = []
  with open(pair_file , 'r') as f:
    lines = f.readlines()
    for line in lines:
        image_path = os.path.join(path , 'skeleton' , line.split()[1])
        cor_path = os.path.join(path , 'cor' , line.split()[0])
        image_cor_paths.append((image_path , cor_path))
    return image_cor_paths

  
class Loader(Dataset):
  def __init__(self , mode = 'train' , loader = loader):
    self.image_cor_path = None
    self.mode = mode
    self.loader = loader
    if self.mode == 'train':
        self.image_cor_path = readfile(os.path.join(path,'pair','train_pair.txt'))
    elif self.mode == 'val':
        self.image_cor_path = readfile(os.path.join(path,'pair','val_pair.txt'))
    elif self.mode == 'show':
        self.image_cor_path = readfile(os.path.join(path,'pair','show_pair.txt'))
    elif self.mode == 'val_show':
        self.image_cor_path = readfile(os.path.join(path,'pair','val_show_pair.txt'))

  def __len__(self):
    return len(self.image_cor_path)

  def __getitem__(self , idx):
    image_path , cor_path  = self.image_cor_path[idx]
    image = self.loader(image_path , mode = 'image')
    cor = self.loader(cor_path , mode = 'cor')
    return image , cor    
    

if __name__ == "__main__":
    train_set = Loader(mode = 'val')
    data_loader = torch.utils.data.DataLoader(train_set , batch_size = 2  , shuffle = False , num_workers = 1)
    print('dataset num is: ',len(data_loader))
    for i  , (image , cor) in enumerate(data_loader):
        print(cor)
        print(image.shape)
        print(cor.shape)
