# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv

import torch
from torchvision import transforms
from numpy import load, concatenate,float

class DataParser():
    def __init__(self,file_names,path='/mnt/c/Users/rydah/Documents/02476/data/corruptmnist/',verbose=False):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
        for i in range(len(file_names)):
            if (i==0):
                data = load(path+"/"+file_names[0])
                if verbose:
                    for f in data.files:
                        print(f)
                        print(type(data[f]))
                self.images = data['images']
                self.labels = data['labels']
            else:
                data = load(path+"/"+file_names[i])
                if verbose:
                    for f in data.files:
                        print(f)
                        print(type(data[f]))
                self.images = concatenate([self.images,data['images']])
                self.labels = concatenate([self.labels,data['labels']])
        self.images = self.images.astype(float)
        self.labels = self.labels.astype(float)
    def __len__(self):
        return len(self.labels)
    
    def save_to_tensor(self,filename,path=''):
        images = [self.transform(img) for img in self.images]
        m = {'images': images, 'labels': self.labels}
        torch.save(m, path+"/"+filename)

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train = DataParser(['train_0.npz','train_1.npz','train_2.npz','train_3.npz','train_4.npz'],path=input_filepath)
    test = DataParser(['test.npz'])
    train.save_to_tensor('train.pt',path=output_filepath)
    test.save_to_tensor('test.pt',path=output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
