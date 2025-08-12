
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import cv2
import sys
import os
from dataset import brain_ImageData
from ConvolutionalNeuralNetwork import CNN
import torch.nn as nn
import torch.nn.functional as F



image_Data = brain_ImageData()
image_Data.normalize()
image_Data.data_split()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
networkModel = CNN().to(device)



learning_rate = .0001            
epoch = 100                        
optimizer = torch.optim.Adam(networkModel.parameters(), lr=learning_rate)

#Creating a DataLoader
image_Data.mode = 'train'
train_dataLoader = DataLoader(image_Data,batch_size=32, shuffle=True)


image_Data.mode = 'test'
test_dataLoader = DataLoader(image_Data,batch_size=40, shuffle=False)
image_Data.mode = 'train'

epoch_train_loss = []
epoch_test_loss = []

# tracking
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)
best_auc = -1.0
train_acc_hist, val_acc_hist, val_auc_hist = [], [], []

for epoch in range(1,epoch):
    train_losses = []
    correct = 0
    total = 0
    networkModel.train()         
    image_Data.mode = 'train'
    for sample in train_dataLoader:
            optimizer.zero_grad()          #Resets the optimizer before every Epcoch
            img = sample['images'].to(device)            
            label = sample['labels'].to(device)
            img = img.reshape((img.shape[0],img.shape[3],img.shape[1],img.shape[2]))        #Reshapes the data so it aligns with the format [#number of data points, 3 RGB Chanels, 128, 128]
       
            y_hat = networkModel(img)                        #Output Data in tensor form
            
          

            error = nn.BCELoss()                                     
            loss= torch.sum( error(y_hat.squeeze(),label))          
            loss.backward()                                        
            optimizer.step()                                      
            train_losses.append(loss.item())                              

            preds = (y_hat.detach().squeeze() >= 0.5).long().cpu().numpy()
            correct += (preds == label.cpu().numpy().astype(int)).sum()
            total += label.size(0)

    epoch_train_loss.append(np.mean(train_losses))
    train_acc_hist.append(correct/total if total>0 else 0.0)


    networkModel.eval()
    image_Data.mode = 'test'
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for sample in test_dataLoader:
            img = sample['images'].to(device)
            label = sample['labels'].to(device)
            img = img.reshape((img.shape[0],img.shape[3],img.shape[1],img.shape[2]))
            y_hat = networkModel(img)
            all_probs.append(y_hat.cpu().numpy().squeeze())
            all_labels.append(label.cpu().numpy())
    if len(all_probs) > 0:
        probs = np.concatenate(all_probs).reshape(-1)
        labels = np.concatenate(all_labels).reshape(-1).astype(int)
        preds = (probs >= 0.5).astype(int)
        val_acc = accuracy_score(labels, preds)
        try:
            val_auc = roc_auc_score(labels, probs)
        except ValueError:
            val_auc = float("nan")
        val_acc_hist.append(val_acc)
        val_auc_hist.append(val_auc)

        # save best weights by AUC
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save(networkModel.state_dict(), os.path.join(out_dir, "best_model.pth"))

    image_Data.mode = 'train'

    if (epoch+1) % 10 ==0:
        print( 'Train Epoch: {}\tLoss: {:.6f}'.format(epoch+1, np.mean(train_losses)))                     #Prints epoch and the loss every 10th Epoch, takes an average of all the losses



torch.save(networkModel.state_dict(), os.path.join(out_dir, "last_model.pth"))


def threshold(values, threshold = .50, min =0, max = 1):
     functionOutput = np.array(list(values))
     functionOutput[functionOutput >= threshold] = max
     functionOutput[functionOutput < threshold] = min
     return functionOutput


# after training, evaluate once and plot/save figures
networkModel.eval()
outputs = []          #Stores Output values out of the Network
actualValues = []       #Stores actual correct values 
image_Data.mode = 'test'
with torch.no_grad():
        for sample in test_dataLoader:
            img = sample['images'].to(device)             #Sets storage device to the same as the CNN
            label = sample['labels'].to(device)
            img = img.reshape((img.shape[0],img.shape[3],img.shape[1],img.shape[2]))        #Reshapes the data so it aligns with the format [#number of data points, 3 RGB Chanels, 128, 128]
            y_hat = networkModel(img)                        #Output Data in tensor form
            outputs.append(y_hat.cpu().detach().numpy())                   #Changes Output data to numpy form 
            actualValues.append(label.cpu().detach().numpy())

outputs = np.concatenate(outputs, axis=0).squeeze()                        #Adds all outputs and actualValues to the array and squeeze function removes the extra dimension
actualValues = np.concatenate(actualValues, axis=0).squeeze()

# Calculates accuracy of the neural network 
accuracy = accuracy_score(actualValues,threshold(outputs))
print(accuracy)        

# save arrays
np.save(os.path.join(out_dir, "final_outputs.npy"), outputs)
np.save(os.path.join(out_dir, "final_labels.npy"), actualValues)

# plots
x_values = range(1,len(outputs)+1)

plt.figure(figsize=(16,9))
plt.plot(x_values,outputs,'b-', linewidth=1.5)
plt.xlabel('Data Points Index')
plt.ylabel('Output Value')
plt.title('Convolutional Neural Network Testing Outputs')
plt.grid(True)
plt.savefig(os.path.join(out_dir, "outputs_curve.png"), bbox_inches="tight")
plt.show()

plt.figure(figsize=(16,9))
plt.plot(epoch_train_loss, c='b', label='Train Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Data')
plt.grid(True)
plt.savefig(os.path.join(out_dir, "training_loss.png"), bbox_inches="tight")
plt.show()

# optional accuracy/auc curves if collected
if len(train_acc_hist)>0 and len(val_acc_hist)>0:
    plt.figure(figsize=(12,8))
    plt.plot(train_acc_hist, label="train_acc")
    plt.plot(val_acc_hist, label="val_acc")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_curves.png"), bbox_inches="tight")
    plt.show()

if len(val_auc_hist)>0:
    plt.figure(figsize=(12,8))
    plt.plot(val_auc_hist)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.title('Validation AUC')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "val_auc_curve.png"), bbox_inches="tight")
    plt.show()


try:
    fpr, tpr, _ = roc_curve(actualValues, outputs)
    auc_val = roc_auc_score(actualValues, outputs)
    print(auc_val)
    plt.figure(figsize=(10,8))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), bbox_inches="tight")
    plt.show()
except Exception as e:
    print("AUC/ROC could not be computed:", e)

cm = confusion_matrix(actualValues, threshold(outputs))
plt.figure(figsize=(8,6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
plt.show()

print("Saved weights and figures to:", out_dir)
