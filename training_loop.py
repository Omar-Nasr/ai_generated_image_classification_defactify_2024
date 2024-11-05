import time 
from torchmetrics import F1Score
import torch
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_model(model,criterion,optimizer,scheduler,train_dataloader,classifier,num_epochs,checkpoint_path,task="binary"):
    since = time.time()
    if(task=="binary"):
        Calc_F1 = F1Score(task="binary")
    else:
        Calc_F1 = F1Score(task="multiclass",num_classes=6)

    running_loss = 0
    best_f1 = 0
    with open("logs","a") as f:
        for epoch in range(num_epochs):
            f.write(f'Epoch {epoch}/{num_epochs - 1}\n')
            print(f'Epoch {epoch}/{num_epochs - 1}\n')
            f.write('-' * 10 + "\n")
            print('-' * 10 + "\n")
            model.train()
            full_preds = []
            full_labels = []
            for inputs,labels in train_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(True):
                    features = model(inputs)
                    outputs = classifier(features)
                    _,preds = torch.max(outputs,1)
                    full_preds.append(preds)
                    full_labels.append(labels)
                    loss = criterion(labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
            full_preds = np.concatenate(full_preds)
            full_labels = np.concatenate(full_labels)
            curr_f1 = Calc_F1(full_preds,full_labels)
            if(curr_f1>best_f1):
                best_f1 = curr_f1
                torch.save(model.state_dict(),checkpoint_path)

            f.write(f'Epoch ${epoch} Training Loss: ${running_loss} \n')
            print(f'Epoch ${epoch} Training Loss: ${running_loss} \n')

            f.write(f'Epoch ${epoch} Training F1_Score {curr_f1} Best F1: {best_f1}')
            print(f'Epoch ${epoch} Training F1_Score {curr_f1} Best F1: {best_f1}')
        time_elapsed = time.time() - since
        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')

        f.write(f'Best F1: {best_f1} after {num_epochs} Epochs')
        print(f'Best F1: {best_f1} after {num_epochs} Epochs')      
        model.load_state_dict(torch.load(checkpoint_path))
        return model,classifier


        
