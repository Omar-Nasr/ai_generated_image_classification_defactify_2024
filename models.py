from training_loop import train_model
from data_handler import Image_Classification_Dataset
from torch.utils.data import DataLoader
from torchvision import models
import torch
from adopt import ADOPT
def train_classifier(train_data_dir,checkpoint_path,num_epochs=10,val_data_dir=None,val_labels=None,val=False,batch_sz=16,task="binary",model_name="swin",optimizer_name="adam",use_fourier=False):
    if(model_name=="swin"):
        model = models.swin_v2_b(pretrained=True)
    elif(model_name=="vit"):
        model = models.vit_l_32(pretrained=True)
    elif(model_name=="convnext"):
        model = models.convnext_large(pretrained=True)
    else:
        model = models.vgg16_bn(pretrained=True)
    from data_handler import Image_Classification_Dataset
    from torch.utils.data import DataLoader
    img_dataset = Image_Classification_Dataset(train_data_dir)

    train_dataloader = DataLoader(img_dataset,batch_sz,num_workers=4)
    if(optimizer_name=="adam"):
        optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-5)
    else:
        optimizer = ADOPT(params=model.parameters(),lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    criterion = torch.nn.CrossEntropyLoss()
    if(task=="binary"):
        classifier = torch.nn.Linear(1000,2)
    else:
        classifier = torch.nn.Linear(1000,6)
    if(val==True):
        val_dataset = Image_Classification_Dataset(val_data_dir,task=task,val=True,val_labels=val_labels)
        val_dataloader = DataLoader(val_dataset,batch_sz,num_workers=4)
        model_trained = train_model(model,criterion,optimizer,scheduler,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourrier=use_fourier,model_name=model_name,val_dataloader=val_dataloader,batch_sz=batch_sz)
    else:
        model_trained = train_model(model,criterion,optimizer,scheduler,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourrier=use_fourier,model_name=model_name,batch_sz=batch_sz)
    return model_trained
