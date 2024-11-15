from torch.utils.data import Dataset
import os 
import pandas as pd 
import numpy as np 
from torchvision.io import read_image
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
class Image_Classification_Dataset(Dataset):
    def __init__(self,train_data_dir,task="Binary",val=False,val_labels = None,test=False):
        super().__init__()
        self.test = test
        if(val==True):
            val_tsv_path = os.path.join(train_data_dir,"val_images.tsv")
            val_df = pd.read_csv(val_tsv_path,sep="\t",header=None)
            self.img_dirs = np.array(val_df.iloc[:,0].apply(lambda x: x.replace("image_class/",train_data_dir)))
            self.img_labels = val_labels 
        elif(test==True):
            test_tsv_path = os.path.join(train_data_dir,"test_images.tsv")
            test_df = pd.read_csv(test_tsv_path,sep="\t",header=None)
            self.img_dirs = np.array(test_df.iloc[:,0].apply(lambda x: x.replace("image_class/",train_data_dir)))
            self.img_labels = None
        else:
            train_tsv_path = os.path.join(train_data_dir,"image_labels.tsv")
            train_df = pd.read_csv(train_tsv_path,sep="\t",header=None)
            self.img_dirs = np.array(train_df.iloc[:,0].apply(lambda x: x.replace("image_class/",train_data_dir)))
            if(task=="Binary"):
                self.img_labels = np.array(train_df.iloc[:,1])
            elif(task=="Multiclass"):
                self.img_labels = np.array(train_df.iloc[:,2])
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self,idx):
        image_path = self.img_dirs[idx]
        image = read_image(image_path)
        image = image_processor(images=image,return_tensors="pt")['pixel_values'][0]
        if(self.test==False):
            label = self.img_labels[idx]
            return image,label
        else:
            return image





