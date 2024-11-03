from torch.utils.data import Dataset
import os 
import pandas as pd 
import numpy as np 
from torchvision.io import read_image
from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
class Image_Classification_Dataset(Dataset):
    def __init__(self,train_data_dir,task="Binary"):
        super().__init__()
        train_tsv_path = os.path.join(train_data_dir,"image_labels.tsv")
        train_df = pd.read_csv(train_tsv_path,sep="\t")
        self.img_dirs = np.array(train_df.iloc[:,0])
        if(task=="Binary"):
            self.img_labels = np.array(train_df.iloc[:,1])
        elif(task=="Multiclass"):
            self.img_labels = np.array(train_df.iloc[:,2])
    def __len__(self):
        return len(self.img_labels)
    def __get__item(self,idx):
        image_path = self.img_dirs[idx]
        label = self.img_labels[idx]
        image = read_image(image_path)
        image = image_processor(images=image,return_tensors="pt")['pixel_values'][0]





