from models import train_classifier
def objective(trial,train_data_dir,checkpoint_path,num_epochs,val_data_dir,val_labels,val,batch_sz,task,optimizer_name,output_dict):
    lr = trial.suggest_float("lr",1e-7,1e-5,log=True)
    lr2 = trial.suggest_float("lr2",1e-6,1e-3,log=True)
    model,classifier,val_f1 = train_classifier(train_data_dir,checkpoint_path,num_epochs,val_data_dir,val_labels,val,batch_sz,task,optimizer_name=optimizer_name,lr=lr,lr2=lr2,trial=trial)
    output_dict["model"]=model
    output_dict["classifier"]=classifier
    return val_f1
