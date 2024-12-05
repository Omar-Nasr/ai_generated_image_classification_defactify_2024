from training_loop import train_model
from data_handler import Image_Classification_Dataset
from torch.utils.data import DataLoader
from training_loop import train_classical_classifier
from torchvision import models
import torch
from torch import nn
from adopt import ADOPT
def train_classifier(train_data_dir,checkpoint_path,num_epochs=10,val_data_dir=None,val_labels=None,val=False,batch_sz=16,task="Binary",model_name="swin",optimizer_name="adam",use_fourier=False,lr=1e-7,lr2=1e-4,fine_tune=False,trial=None,dropout_rate=0.18,num_classes=6,freeze_number=0,classical_ml=False):
    if(model_name=="swin"):
        model = models.swin_v2_b(pretrained=True)
    elif(model_name=="vit"):
        model = models.vit_l_32(pretrained=True)
    elif(model_name=="convnext"):
        model = models.convnext_large(pretrained=True)
    else:
        model = models.vgg16_bn(pretrained=True)
    img_dataset = Image_Classification_Dataset(train_data_dir,task=task)

    train_dataloader = DataLoader(img_dataset,batch_sz,num_workers=4,shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    if(task=="Binary"):
        classifier = torch.nn.Linear(1000,2)
    else:
        classifier = nn.Sequential(nn.Dropout(dropout_rate),nn.Linear(1000,1000),nn.GELU(),nn.Dropout(dropout_rate),nn.Linear(1000,num_classes))
    i = 0 
    for param in model.parameters():
        if(i==freeze_number):
            break
        param.requires_grad=False
  
    if(optimizer_name=="adam"):
        optimizer = torch.optim.Adam(params=model.parameters(),lr=lr)
        optimizer2 = torch.optim.Adam(params=classifier.parameters(),lr=lr2)
    else:
        optimizer = ADOPT(params=model.parameters(),lr=lr)
        optimizer2 = ADOPT(params=classifier.parameters(),lr=lr*100)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    # scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min') 
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.001,steps_per_epoch=len(train_dataloader),epochs=num_epochs)
    # scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.1,steps_per_epoch=len(train_dataloader),epochs=num_epochs)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer2)
    if(val==True):
        val_dataset = Image_Classification_Dataset(val_data_dir,task=task,val=True,val_labels=val_labels)
        val_dataloader = DataLoader(val_dataset,batch_sz,num_workers=4)
        if(classical_ml==True):
            model_trained=train_classical_classifier(model,train_dataloader,val_dataloader,batch_sz,num_epochs)
        else:
            model_trained = train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourier=use_fourier,model_name=model_name,val_dataloader=val_dataloader,batch_sz=batch_sz,fine_tune=fine_tune,trial=trial)
    else:
        model_trained = train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourier=use_fourier,model_name=model_name,batch_sz=batch_sz,fine_tune=fine_tune)
    return model_trained
