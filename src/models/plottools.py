import matplotlib.pyplot as plt
import numpy as np

def loss_plot(train,filename='loss.png',path='reports/figures/',title='Log Likelihood Loss'):
    plt.figure()
    plt.plot(train,label="train")
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.savefig(path+filename)

def accuracy_plot(train,filename='accuracy.png',path='reports/figures/',title='Accuracy',label="train"):
    plt.figure()
    plt.plot(train,label=label)
    plt.legend(loc="upper right")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.savefig(path+filename)

def train_plots(accuracy,loss):
    loss_plot(loss,filename='train_accuracy.png',title="Training Loss")
    accuracy_plot(accuracy,filename='train_loss.png',title="Training Accuracy")

    
