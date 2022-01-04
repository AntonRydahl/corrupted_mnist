import argparse
import sys

import torch
import numpy as np

from model import MyAwesomeModel
from data import get_test_loader

def predict():
    print("Evaluating Model")

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    model = torch.load('models/trained_model.pt')
    test_set = get_test_loader()

    criterion = torch.nn.NLLLoss()

    # Validation
    model.eval()
    validation_accuracy = 0
    validation_count = 0
    val_loss = 0
    with torch.no_grad():
        # validation pass here
        for idx, batch in enumerate(test_set):
            images, labels = batch
            labels = labels.type(torch.LongTensor)
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            val_loss += loss.item()
            _, top_class = log_ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
            validation_count += 1
    validation_accuracy /= validation_count
    print(
        f'Validation:\n\t loss: {val_loss/validation_count}\n\tAccuracy: {validation_accuracy.item()*100} %')

if __name__ == '__main__':
    print("Evaluating Feed Forward Neural Network")
    predict()
