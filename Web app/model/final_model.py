# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# Pillow
from PIL import Image
import pickle

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        """
        This function initializes the parameters for a conv2D layer

        Parameters
        ------------
        in_channels : int
        Number of channels in the input image

        out_channels : int
        Number of channels produced by the convolution

        kernel_size : int or tuple
        Size of the convolving kernel 

        stride : int or tuple
        Stride of the convolution. Default: 1

        padding: int, tuple or str
        Padding added to all four sides of the input. Default: 'same'
        """
        super(Conv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(
                    (self.out_channels, self.in_channels, *self.kernel_size),
                    requires_grad=True
                )
            )
        )
        self.bias = nn.Parameter(
            torch.zeros((self.out_channels,), requires_grad=True)
        )

    def forward(self, x):
        """
        This function performs convolution operation on the input
        Parameters
        ------------
        x : tensor, float32
        Input image to the convolution layer

        Returns
        ------------
        x : tensor, float32
        feature map output from the last layer
        """
        x = F.conv2d(x,self.weight,self.bias,padding=self.padding,stride=self.stride)
        x = F.relu(x)
        return x



class MaxPool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        """
        This function initializes the parameters for a maxpool layer

        Parameters
        ------------
        kernel_size : int
        window height and width for the maxpooling window

        stride : int
        the stride of the window. Default value is kernel_size

        padding: int
        implicit zero padding to be added on both sides
        """
        super(MaxPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding


    def forward(self, x): 
        """
        This function performs max-pool operation on the input

        Parameters
        ------------
        x : tensor, float32
        Input image to the convolution layer

        Returns
        ------------
        x : tensor, float32
        max-pooled output from the last layer
        """
        x = F.max_pool2d(x,kernel_size=self.kernel_size,padding=self.padding,
                         stride=self.stride)
        return x


class Dense(nn.Module):
    def __init__(self, in_features, out_features):
        """
        This function initializes the parameters for a dense layer
        Parameters
        ------------- 
        in_features : int
        shape of the input to the dense layer

        out_features : int
        number of units in the dense layer
        """
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features =  out_features
        
        self.weight = nn.Parameter(
            torch.nn.init.xavier_uniform_(
                torch.empty(
                    (self.in_features, self.out_features),
                    requires_grad=True,
                )
            )
        )
        print(self.weight.shape)
        self.bias = nn.Parameter(
            torch.zeros((self.out_features,), requires_grad=True)
        )
        print(self.bias.shape)

    def forward(self, x):
        """
        This function performs dense operation on the input
        Parameters
        ------------
        x : tensor, float32
        Input flattened image to the convolution layer

        Returns
        ------------
        x : tensor, float32
        linear operation output from the last layer
        """
        x = torch.mm(x, self.weight) + self.bias
        return x



class CNNModel(nn.Module):
    def __init__(self):
        """
        This function initializes the layers for the CNN model
        """
        super(CNNModel, self).__init__()

        # parameters for conv layers
        filter_dim = 3
        in_channels = [3, 32, 64, 64, 64]
        out_channels = [32, 64, 64, 64,64]

        # parameters for dense layers
        dense_in_features = [64*8*8, 1024]
        dense_out_features = [1024, 7]

        # initializing all the layers
        self.c1 = Conv2D(in_channels[0], out_channels[0], filter_dim)
        self.m1 = MaxPool(2)
        self.dropout = nn.Dropout(p=0.25)
        self.c2 = Conv2D(in_channels[1], out_channels[1], filter_dim)
        self.m2 = MaxPool(2)
        self.dropout = nn.Dropout(p=0.25)
        
        self.c3 = Conv2D(in_channels[2], out_channels[2], filter_dim)
        self.m3 = MaxPool(2)
        self.dropout = nn.Dropout(p=0.25)
        self.c4 = Conv2D(in_channels[3],out_channels[3],filter_dim)
        self.m4 = MaxPool(2)
        self.dropout = nn.Dropout(p=0.25)
        
        self.c5 = Conv2D(in_channels[4], out_channels[4], filter_dim)
        self.m5 = MaxPool(2)
        self.dropout = nn.Dropout(p=0.25)
        
        self.d1 = Dense(dense_in_features[0],dense_out_features[0])

        self.d2 = Dense(dense_in_features[1], dense_out_features[1])
        
    def forward(self,x):
        """
        This function performs convolutions, relu, max_pooling, dropout, 
        reshape and dense operations on the input to the model.

        Parameters
        ------------
        x : tensor, float32
        Input image to the model

        Returns
        ------------
        x : tensor, float32
        output from the last layer

        """
        x = self.c1(x)
        x = self.m1(x)
        x = self.dropout(x)
        
        x = self.c2(x)
        x = self.m2(x)
        x = self.dropout(x)
        
        x = self.c3(x)
        x = self.m3(x)
        x = self.dropout(x)
        
        x = self.c4(x)
        x = self.m5(x)
        x = self.dropout(x)
        
        x = self.c5(x)
        x = self.m3(x)
        x = self.dropout(x)
        
        #x = torch.flatten(x,-1)
        x = x.view(x.shape[0],-1)
        x = self.d1(x)
        x = F.relu(x)

        x = self.d2(x)

        return x
    
def loss(target_y, predicted_y):
    """
    Cross entropy loss between target and predicted value
      
    Parameters
    ----------
    target_y: tensor, float32
    Target labels
    predicted_y: tensor, float32
    Prediction of the classes made by model
      
    Returns
    -------
    cost: tensor, float32
    The average cross-entropy cost of the mini-batch of inputs
      
    """
    cost = F.cross_entropy(predicted_y,target_y)
    return cost

def train(model: nn.Module, inputs, outputs, optimizer: torch.optim.Optimizer):
    # Set the model to training mode
    model.train()

    # Zero the gradients
    optimizer.zero_grad()

    # Compute the model's predictions
    predictions = model(inputs)

    # Compute the loss
    loss = nn.functional.cross_entropy(predictions, outputs)

    # Compute the gradients
    loss.backward()

    # Update the weights
    optimizer.step()

    return loss, predictions


num_epochs = 10

def train_model(model, train_loader, validation_loader, optimizer, loss, num_epochs, device,version):
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    
    print("------------------------Training--------------------------")
    
    for i in range(num_epochs):
        t0 = datetime.now()
        train_loss = []
        train_acc = []
        model.train()
        for num, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            losses, pred = train(model, x_batch, y_batch, optimizer) 
            train_loss.append(losses.item())
            train_acc.append((y_batch == pred.argmax(dim=-1)).float().mean().item())

        test_loss = []
        test_acc = []
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in validation_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                test_pred = model(x_batch)
                test_loss.append(loss(y_batch, test_pred).item())
                test_acc.append((y_batch == test_pred.argmax(dim=-1)).float().mean().item())
        dt = datetime.now() - t0
        print('\nEpoch: {}\t\tTrain Loss: {}\tTrain Accuracy: {}\nDuration: {}\tTest Loss: \
        {}\tTest Accuracy: {}\n'.format(
             i+1, np.mean(train_loss), np.mean(train_acc),dt,np.mean(test_loss), 
             np.mean(test_acc)
        ))

        # tracking accuracy and lossin each epoch for plot
        history['train_loss'].append(np.mean(train_loss))
        history['train_acc'].append(np.mean(train_acc))
        history['test_loss'].append(np.mean(test_loss))
        history['test_acc'].append(np.mean(test_acc))



    return history


if __name__ == "__main__":

    data_path = "C:/Users/Saurab/Desktop/Plant Disease Final/Dataset/Plant"
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    indices = list(range(len(dataset)))
    split = int(np.floor(0.90 * len(dataset)))  # train_size
    validation = int(np.floor(0.70 * split))   # validation
    np.random.shuffle(indices)

    train_indices, validation_indices, test_indices = (
        indices[:validation],
        indices[validation:split],
        indices[split:],
    )
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=validation_sampler
    )


    LEARNING_RATE = 0.001
    
    model = CNNModel()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    num_epochs = 25


    criterion = nn.CrossEntropyLoss()  # this include softmax + cross entropy loss
    #optimizer = torch.optim.Adam(model.parameters())


    history = train_model(model,train_loader, validation_loader, optimizer, loss, num_epochs, device,version=2)

    torch.save(model.state_dict(),'TomatoFinal/model/final_model.pth')
