import time 
from torchmetrics import F1Score
import torch
import numpy as np
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model,criterion,optimizer,scheduler,train_dataloader,classifier,num_epochs,checkpoint_path,task="binary",use_fourrier=False,model_name = "test",val_dataloader=None):
    since = time.time()
    if(task=="binary"):
        Calc_F1 = F1Score(task="binary")
    else:
        Calc_F1 = F1Score(task="multiclass",num_classes=6)

    best_f1 = 0
    best_val_f1=0
    checkpoint_path = os.path.join(checkpoint_path, model_name+".pt")
    with open("logs","a") as f:
        for epoch in range(num_epochs):
            running_loss = 0
            f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
            print(f'Epoch {epoch}/{num_epochs - 1}\n')
            f.write('-' * 10 + "\n")
            print('-' * 10 + "\n")
            model.train()
            model.to(device)
            classifier.to(device)
            full_preds = []
            full_labels = []
            # k=1
            for inputs,labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):

                    if use_fourrier==True:
                        print(inputs.shape)
                        inputs = torch.fft.fftn(inputs,3)
                        inputs = inputs.abs()
                        print(inputs.shape)
                    features = model(inputs)
                    outputs = classifier(features)
                    _,preds = torch.max(outputs,1)
                    full_preds.append(preds.cpu())
                    full_labels.append(labels.cpu())
                    loss = criterion(outputs,labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                # print(f"Batch {k} loss: {running_loss}" )
                # k+=1
            full_preds = np.concatenate(full_preds)
            full_preds = torch.from_numpy(full_preds)
            full_labels = np.concatenate(full_labels)
            full_labels = torch.from_numpy(full_labels)
            curr_f1 = Calc_F1(full_preds,full_labels)
            if(curr_f1>best_f1):
                best_f1 = curr_f1
                torch.save(model.state_dict(),checkpoint_path)

            f.write(f'Epoch ${epoch} Training Loss: ${running_loss} \n')
            print(f'Epoch ${epoch} Training Loss: ${running_loss} \n')

            f.write(f'Epoch ${epoch} Training F1_Score {curr_f1} Best F1: {best_f1}')
            print(f'Epoch ${epoch} Training F1_Score {curr_f1} Best F1: {best_f1}')

            running_loss=0
            val_preds = []
            val_labels = []
            if(val_dataloader!=None):
                for inputs,labels in val_dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    if(use_fourrier==True):
                        inputs = torch.fft.fftn(inputs,3)
                    features = model(inputs)
                    outputs = classifier(features)
                    loss = criterion(outputs,labels)
                    _,preds = torch.max(outputs,1)

                    running_loss += loss.item() * inputs.size(0)
                    val_preds.append(preds.cpu())
                    val_labels.append(labels.cpu())
                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels)
                val_preds = torch.from_numpy(val_preds)
                val_labels = torch.from_numpy(val_labels)
                curr_f1 = Calc_F1(val_preds,val_labels)
                if(curr_f1>best_val_f1):
                    best_f1 = curr_f1
                    best_val_f1 = curr_f1
                    torch.save(model.state_dict(),checkpoint_path)
                f.write(f'Epoch ${epoch} Val Loss: ${running_loss} \n')
                print(f'Epoch ${epoch} Val Loss: ${running_loss} \n')

                f.write(f'Epoch ${epoch} Val F1_Score {curr_f1} Best F1: {best_val_f1}')
                print(f'Epoch ${epoch} Val F1_Score {curr_f1} Best F1: {best_val_f1}')


        time_elapsed = time.time() - since
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')

        f.write(f'Best F1: {best_f1} after {num_epochs} Epochs')
        print(f'Best F1: {best_f1} after {num_epochs} Epochs')      
        model.load_state_dict(torch.load(checkpoint_path))
        return model,classifier


        
