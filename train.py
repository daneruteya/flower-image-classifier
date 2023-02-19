#!/usr/bin/env python
# coding: utf-8

# PROGRAMMER: Dan Eruteya
# DATE CREATED:     15/08/2022


# Imports
from workspace_utils import active_session

import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

with active_session():
    # do long-running work here


    # Define parser
    def get_input_args():
        parser = argparse.ArgumentParser()

        # Add architecture selection to parser
        parser.add_argument('--arch',
                            type=str,
                            help='CNN model architecture')

        # Add checkpoint directory to parser
        parser.add_argument('--save_dir',
                            type=str,
                            help='directory to save checkpoints.')

        # Add hyperparameter tuning to parser
        parser.add_argument('--learning_rate',
                            type=float,
                            help='Learning rate')
        parser.add_argument('--hidden_units',
                            type=int,
                            help='Number of neurons in hidden layer')
        parser.add_argument('--epochs',
                            type=int,
                            help='Number of epochs for training')

        # Add GPU Option to parser
        parser.add_argument('--gpu',
                            action="store_true",
                            help='GPU to be used for training')

        # Parse args
        in_arg = parser.parse_args()
        return in_arg



    # Function train_transform(train_dir) performs training transforms on a dataset
    def train_transform(train_dir):
       # Define transformation
       train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
       # Load the Data
       train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
       return train_data

    # Function test_transform(test_dir) performs test transforms on a dataset
    def test_transform(test_dir):
        # Define transformation
        test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
        # Load the Data
        test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
        return test_data

    # Function valid_transform(valid_dir) performs valid transforms on a dataset
    def valid_transform(valid_dir):
       # Define transformation
       valid_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
       # Load the Data
       valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
       return valid_data



    # Function data_loader(data, train=True) creates a dataloader from dataset imported
    def data_loader(data, train=True):
        if train:
            loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(data, batch_size=64)
        return loader


    # Function check_gpu(gpu_arg) make decision on using CUDA with GPU or CPU
    def check_gpu(gpu_arg):
       # If gpu_arg is false then simply return the cpu device
        if not gpu_arg:
            return torch.device("cpu")

        # If gpu_arg then make sure to check for CUDA before assigning it
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Print result
        if device == "cpu":
            print("CUDA was not found on device, using CPU instead.")
        return device

    # Function [model_architecture(architecture="densenet121")] that downloads model from torchvision
    def model_architecture(architecture="densenet121"):
        # Load Defaults if none specified
        if type(architecture) == type(None):
            model = models.densenet121(pretrained=True)
            model.name = "densenet121"
            print("Architecture used: densenet121.")

        model = models.densenet121(pretrained=True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        return model

    # Define Classifier
    def classifier(model, hidden_units):
        # Check that hidden layers has been input
        if type(hidden_units) == type(None):
            hidden_units = 256 #hyperparamters
            print("Number of Hidden Layers specificed as 256.")


        # Define Classifier
        model.classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 500)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(500, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        return model.classifier





    # Function network_trainer represents the training of the network model
    def network_trainer(model, trainloaders, testloaders, device,
                      criterion, optimizer, epochs, print_every, steps):


        # Train the classifier layers using backpropogation using the pre-trained network to get features
        epochs = 3
        steps = 0
        running_loss = 0
        print_every = 5

        print("Start..")

        for epoch in range(epochs):
            for inputs, labels in trainloaders:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in testloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Test loss: {test_loss/len(testloaders):.3f}.. "
                          f"Test accuracy: {accuracy/len(testloaders):.3f}")
                    running_loss = 0
                    model.train()

        return model


    #Function validation(model, testloaders, device) validate the above model on test data images
    def validation(model, testloaders, device):
       # Do validation on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            model.eval()
            for data in testloaders:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

    # Function checkpoint(model, Save_Dir, train_data) saves the model at a defined checkpoint
    def checkpoint(model, save_dir, train_data):

        # Save model at checkpoint
        if type(save_dir) == type(None):
            print("Model checkpoint directory not specified, model will not be saved.")
        else:
            if isdir(save_dir):
                # Create `class_to_idx` attribute in model
                model.class_to_idx = train_data.class_to_idx
                # Create checkpoint dictionary
                checkpoint = {'architecture': 'densenet121',
                              'classifier': model.classifier,
                              'class_to_idx': model.class_to_idx,
                              'state_dict': model.state_dict()}

                # Save checkpoint
                torch.save(checkpoint, 'my_checkpoint.pth')

            else:
                print("Directory not found, model will not be saved.")


    # Main program function defined below
    def main():


        # Get Keyword in_arg for Training
        in_arg = get_input_args()


        # Set directory for training
        data_dir = 'flowers'
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        # Pass transforms in, then create trainloader
        train_data = train_transform(train_dir)
        valid_data = valid_transform(valid_dir)
        test_data = test_transform(test_dir)

        trainloaders = data_loader(train_data)
        validloaders = data_loader(valid_data, train=False)
        testloaders = data_loader(test_data, train=False)

        # Load Model
        model = model_architecture(architecture=in_arg.arch)

        # Build Classifier
        model.classifier = classifier(model,
                                hidden_units=in_arg.hidden_units)



        # Check for GPU

        device = check_gpu(gpu_arg=in_arg.gpu);

        # Send model to device
        model.to(device);

        # Check for learnrate in_arg
        if type(in_arg.learning_rate) == type(None):
            learning_rate = 0.003
            print("Learning rate specificed as 0.003")
        else: learning_rate = in_arg.learning_rate

        # Define loss and optimizer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)




        # Train the classifier layers using backpropogation
        trained_model = network_trainer(model, trainloaders, validloaders,
                                        device, criterion, optimizer, in_arg.epochs,
                                        print_every, steps)

        print("\nFinished Training!!")

        # Validate the model
        validation(trained_model, testloaders, device)

        # Save the model
        checkpoint(trained_model, in_arg.save_dir, train_data)


    # Call to main function to run the program
    if __name__ == '__main__':
        main()
