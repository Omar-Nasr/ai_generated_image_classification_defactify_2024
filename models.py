from training_loop import train_model
from data_handler import Image_Classification_Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torch
def train_swin(train_data_dir,checkpoint_path,num_epochs=10,batch_sz=16,task="binary"):
    model = models.swin_v2_b(pretrained=True)
    from data_handler import Image_Classification_Dataset
    from torch.utils.data import DataLoader
    train_data_dir = "/kaggle/input/d/omarnasr/ai-image-classification-2/" 
    img_dataset = Image_Classification_Dataset(train_data_dir)
    train_dataloader = DataLoader(img_dataset,batch_sz,num_workers=4)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    if(task=="binary"):
        classifier = torch.nn.Linear(1000,2)
    else:
        classifier = torch.nn.Linear(1000,6)
    model_trained = train_model(model,criterion,optimizer,scheduler,train_dataloader,classifier,num_epochs,checkpoint_path,task)
    return model_trained
