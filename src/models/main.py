import argparse
import sys

import torch
import numpy as np

from data import mnist
from model import MyAwesomeModel

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print("Arguments:")
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        ## Training Loop
        model.train()
        epochs = 20

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        criterion = torch.nn.NLLLoss()
        for epoch in range(epochs):
            train_loss = 0
            train_accuracy = 0
            train_count = 0
            for idx,batch in enumerate(train_set):
                images,labels = batch
                labels = labels.type(torch.LongTensor)
                optimizer.zero_grad()
                
                log_ps = model(images)
                loss = criterion(torch.FloatTensor(log_ps), labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, top_class  = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_accuracy += torch.mean(equals.type(torch.FloatTensor))
                train_count +=1
            train_accuracy /= train_count
            print(f'Training iteration {epoch}:\n\t loss: {train_loss/train_count}\n\tAccuracy: {train_accuracy.item()*100}%)')
        filename = 'trained_model.pt'
        torch.save(model, filename)
        print(f'saved model {filename}')
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        model = torch.load(args.load_model_from)
        _, test_set = mnist()

        criterion = torch.nn.NLLLoss()

        ## Validation
        model.eval()
        validation_accuracy = 0
        validation_count = 0
        val_loss = 0
        with torch.no_grad():
            # validation pass here
            for idx,batch in enumerate(test_set):
                images, labels = batch
                labels = labels.type(torch.LongTensor)
                log_ps = model(images)
                loss = criterion(log_ps, labels)
                val_loss += loss.item()
                _, top_class  = log_ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                validation_accuracy += torch.mean(equals.type(torch.FloatTensor))
                validation_count +=1
        validation_accuracy /= validation_count
        print(f'Validation:\n\t loss: {val_loss/validation_count}\n\tAccuracy: {validation_accuracy.item()*100}%)')

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    