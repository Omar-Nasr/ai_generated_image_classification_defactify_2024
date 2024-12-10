from training_loop import train_model
from data_handler import Image_Classification_Dataset
from torch.utils.data import DataLoader
from training_loop import train_classical_classifier
from torchvision import models
import torch
from torch import nn
from adopt import ADOPT
from torchmetrics import F1Score
import os
from sklearn.model_selection import KFold
import numpy as np
def train_classifier(train_data_dir,checkpoint_path,num_epochs=10,val_data_dir=None,val_labels=None,val=False,batch_sz=16,task="Binary",model_name="swin",optimizer_name="adam",use_fourier=False,lr=1e-7,lr2=1e-4,fine_tune=False,trial=None,dropout_rate=0.18,num_classes=6,freeze_number=0,classical_ml=False,k_fold=False,passed_model=None,passed_classifier=None,test=False):
    if(passed_model!=None):
        model=passed_model
    elif(model_name=="swin"):
        model = models.swin_v2_b(pretrained=True)
    elif(model_name=="vit"):
        model = models.vit_l_32(pretrained=True)
    elif(model_name=="convnext"):
        model = models.convnext_large(pretrained=True)
    else:
        model = models.vgg16_bn(pretrained=True)
    img_dataset = Image_Classification_Dataset(train_data_dir,task=task)

    train_dataloader = DataLoader(img_dataset,batch_sz,num_workers=4,shuffle=True)
       

    # Initialize the model and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if(passed_classifier!=None):
        classifier=passed_classifier
    elif(task=="Binary"):
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
        optimizer2 = ADOPT(params=classifier.parameters(),lr=lr2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min') 
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.001,steps_per_epoch=len(train_dataloader),epochs=num_epochs)
    # # scheduler2 = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.1,steps_per_epoch=len(train_dataloader),epochs=num_epochs)
    # scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    # scheduler2 = torch.optim.lr_scheduler.ConstantLR(optimizer2)
    if(val==True):
        val_dataset = Image_Classification_Dataset(val_data_dir,task=task,val=True,val_labels=val_labels)
        val_dataloader = DataLoader(val_dataset,batch_sz,num_workers=4)
        if(classical_ml==True):
            pass
            # model.eval()
            # model_trained=train_classical_classifier(model,train_dataloader,val_dataloader,batch_sz,num_epochs)
        elif(k_fold!=True):
            model_trained = train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourier=use_fourier,model_name=model_name,val_dataloader=val_dataloader,batch_sz=batch_sz,fine_tune=fine_tune,test=test)
        else:
            kf = KFold(n_splits=3, shuffle=True)
            best_val_f1=0
            Calc_F1 = F1Score(task="multiclass",num_classes=6)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            for fold, (train_idx, test_idx) in enumerate(kf.split(img_dataset)):
                print(f"Fold {fold + 1}")
                print("-------")

                # Define the data loaders for the current fold
                train_dataloader = DataLoader(
                    img_dataset,
                    batch_sz,
                    sampler=torch.utils.data.SubsetRandomSampler(train_idx),
                )
                test_dataloader = DataLoader(
                    img_dataset,
                    batch_sz,
                    sampler=torch.utils.data.SubsetRandomSampler(test_idx),
                )
                model,classifier =  train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourier=use_fourier,model_name=model_name,val_dataloader=test_dataloader,batch_sz=batch_sz,fine_tune=fine_tune,test=test)
                model_trained=model,classifier
                val_preds = []
                val_labels = []
                running_loss=0
                for inputs,labels in val_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if(use_fourier==True):
                        first_dim = inputs.shape[0]
                        inputs = inputs.reshape(first_dim,224,224,3)
                        inputs = torch.fft.fftn(inputs,3)
                        inputs = inputs.abs()
                        inputs = inputs.reshape(first_dim,3,224,224)
                    features = model(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs,labels)
                    _,preds = torch.max(outputs,1)

                    running_loss += loss.item() * inputs.size(0)
                    val_preds.append(preds.cpu())
                    val_labels.append(labels.cpu())
                running_loss = running_loss/len(val_dataloader)
                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels)
                val_preds = torch.from_numpy(val_preds)
                val_labels = torch.from_numpy(val_labels)
                curr_f1 = Calc_F1(val_preds,val_labels)

                with open("logs","a") as f:
                    f.write(f'Fold ${fold} Test F1_Score {curr_f1} Best F1: {best_val_f1}')
                    print(f'Fold ${fold} Val F1_Score {curr_f1} Best F1: {best_val_f1}')


                if(curr_f1>best_val_f1):
                    best_f1 = curr_f1
                    best_val_f1 = curr_f1
                    model_path = os.path.join(checkpoint_path, model_name+".pt")
                    classifier_path = os.path.join(checkpoint_path, "classifier.pt")
                    torch.save(model.state_dict(),model_path)
                    torch.save(classifier.state_dict(),classifier_path)
                    model_trained=model,classifier


    else:
        model_trained = train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task,use_fourier=use_fourier,model_name=model_name,batch_sz=batch_sz,fine_tune=fine_tune)
    return model_trained
