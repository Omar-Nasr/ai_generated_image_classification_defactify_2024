import time 
from torchmetrics import F1Score
import xgboost as xgb
import torch
import numpy as np
import os 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model,criterion,optimizer,optimizer2,scheduler,scheduler2,train_dataloader,classifier,num_epochs,checkpoint_path,task="Binary",use_fourier=False,model_name = "test",val_dataloader=None,batch_sz=16,fine_tune=False,trial=None,test=False):
    since = time.time()
    if(test==True):
        batch_sz=1
        num_epochs=1
    if(task=="Binary"):
        Calc_F1 = F1Score(task="binary")
    else:
        Calc_F1 = F1Score(task="multiclass",num_classes=6)

    best_f1 = 0
    best_val_f1=0
    model_path = os.path.join(checkpoint_path, model_name+"best.pt")
    classifier_path = os.path.join(checkpoint_path, "best_classifier.pt")
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

                if(fine_tune!=True):
                    optimizer.zero_grad()
                else:
                    for param in model.parameters():
                        param.requires_grad = False
                optimizer2.zero_grad()
                if use_fourier==True:
                    first_dim = inputs.shape[0]
                    inputs = inputs.reshape(first_dim,224,224,3)
                    inputs = torch.fft.fftn(inputs,3)
                    inputs = inputs.abs()
                    inputs = inputs.reshape(first_dim,3,224,224)
                features = model(inputs)
                outputs = classifier(features)
                _,preds = torch.max(outputs,1)
                full_preds.append(preds.cpu())
                full_labels.append(labels.cpu())
                loss = criterion(outputs,labels)
                loss.backward()
                if(fine_tune!=True):
                    optimizer.step()
                optimizer2.step()
                # if(fine_tune!=True):
                #     scheduler.step()
                # scheduler2.step()
                running_loss += loss.item() * inputs.size(0)
                # print(f"Batch {k} loss: {running_loss}" )
                # k+=1
                if(test==True):
                    break
            running_loss = running_loss/len(train_dataloader)
            full_preds = np.concatenate(full_preds)
            full_preds = torch.from_numpy(full_preds)
            full_labels = np.concatenate(full_labels)
            full_labels = torch.from_numpy(full_labels)
            curr_f1 = Calc_F1(full_preds,full_labels)
            if(curr_f1>best_f1):
                best_f1 = curr_f1
                torch.save(model.state_dict(),model_path)
                torch.save(classifier.state_dict(),classifier_path)

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
                    if(test==True):
                        break
                scheduler.step(running_loss)
                scheduler2.step(running_loss)
                running_loss = running_loss/len(val_dataloader)
                val_preds = np.concatenate(val_preds)
                val_labels = np.concatenate(val_labels)
                val_preds = torch.from_numpy(val_preds)
                val_labels = torch.from_numpy(val_labels)
                curr_f1 = Calc_F1(val_preds,val_labels)
                if(trial!=None):
                    trial.report(curr_f1, epoch)
                if(curr_f1>best_val_f1):
                    best_f1 = curr_f1
                    best_val_f1 = curr_f1
                    torch.save(model.state_dict(),model_path)
                    torch.save(classifier.state_dict(),classifier_path)
                f.write(f'Epoch ${epoch} Val Loss: ${running_loss} \n')
                print(f'Epoch ${epoch} Val Loss: ${running_loss} \n')

                f.write(f'Epoch ${epoch} Val F1_Score {curr_f1} Best F1: {best_val_f1}')
                print(f'Epoch ${epoch} Val F1_Score {curr_f1} Best F1: {best_val_f1}')


        time_elapsed = time.time() - since
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')

        f.write(f'Best F1: {best_f1} after {num_epochs} Epochs')
        print(f'Best F1: {best_f1} after {num_epochs} Epochs')      
        model.load_state_dict(torch.load(model_path))
        classifier.load_state_dict(torch.load(classifier_path))
        return model,classifier


def train_classical_classifier(backbone,train_dataloader,val_dataloader,batch_sz,num_epochs,test_dataloader):        
    features_list = np.array([])
    labels_list = np.array([])
    idx=0
    for inputs,labels in train_dataloader:

        features = backbone(inputs)
        features = features.detach().numpy()
        if(idx==0):
            features_list = np.array(features)
        else:
            features_list = np.append(features_list,features)
        labels.detach().numpy()
        if(idx==0):
            labels_list = np.array(labels)
        else:
            labels_list = np.append(labels_list,labels)
        idx+=1
    X_Train=features_list 
    Y_Train=labels_list
    idx=0
    for inputs,labels in val_dataloader:
        features = backbone(inputs)
        features = features.detach().numpy()
        if(idx==0):
            features_list = np.array(features)
        else:
            features_list = np.append(features_list,features)
        labels.detach().numpy()
        if(idx==0):
            labels_list = np.array(labels)
        else:
            labels_list = np.append(labels_list,labels)
        idx+=1
    X_Val = features_list
    Y_Val = labels_list
    clf = xgb.XGBClassifier(tree_method="hist", early_stopping_rounds=2)
    clf.fit(X_Train, Y_Train, eval_set=[(X_Val, Y_Val)])
    clf.save_model("clf.json")
    return 0


