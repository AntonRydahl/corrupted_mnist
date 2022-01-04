import argparse
import sys

import torch
import numpy as np

from model import MyAwesomeModel
from data import get_train_loader
from plottools import train_plots

def train():
    print("Training day and night")
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print("Arguments:")
    print(args)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = get_train_loader()

    # Training Loop
    model.train()
    epochs = 5

    loss_arr = []
    acc_arr = []

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = torch.nn.NLLLoss()
    for epoch in range(epochs):
        train_loss = 0
        train_accuracy = 0
        train_count = 0
        for idx, batch in enumerate(train_set):
            images, labels = batch
            labels = labels.type(torch.LongTensor)
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(torch.FloatTensor(log_ps), labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            train_accuracy += torch.mean(equals.type(torch.FloatTensor))
            train_count += 1
        train_accuracy /= train_count

        loss_arr.append(train_loss)
        acc_arr.append(train_accuracy)
        print(
            f'Training iteration {epoch}:\n\t loss: {train_loss/train_count}\n\tAccuracy: {train_accuracy.item()*100} %')
    filename = 'trained_model.pt'
    torch.save(model, './models/'+filename)
    print(f'saved model {filename}')
    train_plots(acc_arr,loss_arr)

if __name__ == '__main__':
    print("Training Feed Forward Neural Network")
    train()