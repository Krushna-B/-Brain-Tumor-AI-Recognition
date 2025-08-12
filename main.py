
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



def plot_random(healthy, tumor, num=5):
    healthy_Imgs = healthy[np.random.choice(healthy.shape[0],num,replace=False)]
    tumor_Imgs = tumor[np.random.choice(tumor.shape[0],num,replace=False)]

    plt.figure(figsize = (16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('healthy')
        plt.imshow(healthy_Imgs[i])

    plt.figure(figsize = (16,9))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.title('tumor')
        plt.imshow(tumor_Imgs[i])













image_Data = brain_ImageData()
image_Data.normalize()

#Creates a device and then creates a CNN Instance and sets it to cuda cores if possible and cpu if not possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
networkModel = CNN().to(device)



dataLoader = DataLoader(image_Data,batch_size=32, shuffle=False)



networkModel.eval()
outputs = []          
actualValues = []       

with torch.no_grad():
  
        for sample in dataLoader:
            img = sample['images'].to(device)            
            label = sample['labels'].to(device)
            img = img.reshape((img.shape[0],img.shape[3],img.shape[1],img.shape[2]))        #Reshapes the data so it aligns with the format [#number of data points, 3 RGB Chanels, 128, 128]
       
            y_hat = networkModel(img)                        #Output Data in tensor form
            outputs.append(y_hat.cpu().detach().numpy())                   #Changes Output data to numpy form 
            actualValues.append(label.cpu().detach().numpy())

outputs = np.concatenate(outputs, axis=0).squeeze()                        
actualValues = np.concatenate(actualValues, axis=0).squeeze()


#Threshold fuction inputs the sigmoid output between 0 and 1 and returns a value of 0 if <.5 and 1 if >= .5
def threshold(values, threshold = .50, min =0, max = 1):
     functionOutput = np.array(list(values))
     functionOutput[functionOutput >= threshold] = max
     functionOutput[functionOutput < threshold] = min
     return functionOutput




# Calculates accuracy of the neural network 
accuracy = accuracy_score(actualValues,threshold(outputs))

print(accuracy)        

x_values = range(1,len(outputs)+1)


out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(16,9))
plt.plot(x_values,outputs,'b-', linewidth=1.5)

plt.xlabel('Data Points Index')
plt.ylabel('Output Value')
plt.title('Convolutional Neural Network Outputs')
plt.grid(True)
plt.savefig(os.path.join(out_dir, "outputs_curve.png"), bbox_inches="tight")
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


np.save(os.path.join(out_dir, "outputs.npy"), outputs)
np.save(os.path.join(out_dir, "labels.npy"), actualValues)
torch.save(networkModel.state_dict(), os.path.join(out_dir, "model_weights.pth"))
print("Saved figures, arrays, and weights to:", out_dir)

     