#!/usr/bin/env python
# coding: utf-8

# PROGRAMMER: Dan Eruteya
# DATE CREATED:     15/08/2022                             

import argparse

import torch
import numpy as np
import json
from train import check_gpu
from torchvision import models

def get_input_args():
    # Define a parser
    parser = argparse.ArgumentParser()

    # Point towards image for prediction
    parser.add_argument('--image', 
                        type=str, 
                        help='Point to impage file for prediction.',
                        required=True)

    # Load checkpoint created by train.py
    parser.add_argument('--checkpoint', 
                        type=str, 
                        help='Point to checkpoint file as str.',
                        required=True)
    
    # Specify top-k
    parser.add_argument('--top_k', 
                        type=int, 
                        help='Choose top K matches as int.')
    
    # Import category names
    parser.add_argument('--category_names', 
                        type=str, 
                        help='Mapping from categories to real names.')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')

    # Parse args
    in_arg = parser.parse_args()
    return in_arg    
    


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint():
    """
    Loads deep learning model checkpoint.
    """
    
    # Load the saved file
    checkpoint = torch.load("'my_checkpoint.pth")
    
    # Download pretrained model
    model = models.densenet121(pretrained=True);
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model


# ## Image Preprocessing
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(image_path) as image:  
        image = test_transforms(image).numpy()
        
    return image


# ## Class Prediction
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to("cpu")
    input_img = process_image(image_path)
    torch_input_img = torch.from_numpy(input_img).float()
    torch_input_img.unsqueeze_(0) # Add a dimension at first position (will represent the batch size, here 1)
    
    model.eval()
    # Turn off gradients for validation --> saves memory and computations
    with torch.no_grad():
        output = model.forward(torch_input_img)
        probs, indices = output.topk(topk)
        # Transform tensors to numpy arrays and take the first (and single) element
        probs = np.exp(probs.numpy()[0]) # Do not forget to get back the exponential value as output is log-softmax !
        indices = indices.numpy()[0]
        # Revert the dict 'class to indice' to get 'indice to class'
        idx_classes = {v: k for k, v in train_data.class_to_idx.items()}
        classes = [v for k, v in idx_classes.items() if k in indices]
    
    return probs, classes


def main():
    """
    Executing relevant functions
    """

    in_arg = get_input_args()

    # Load categories to names json file
    with open(in_arg.category_names, 'r') as f:
        	cat_to_name = json.load(f)

    # Load model trained with train.py
    model = load_checkpoint(in_arg.checkpoint)
    
    # Process Image
    image_tensor = process_image(in_arg.image)
    
    # Check for GPU
    device = check_gpu(gpu_arg=in_arg.gpu);
    
    # Use `processed_image` to predict the top K most likely classes
    top_probs, top_labels, top_flowers = predict(image_tensor, model, 
                                                 device, cat_to_name,
                                                 in_arg.top_k)
    
if __name__ == '__main__':
    main()
    