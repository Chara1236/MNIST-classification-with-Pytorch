# MNIST Classifcation with PyTorch

Using a simple definition of a Convolutional Neural Network model, this program trains the model to recognize/classify hand written digits from 0-9.

## Installation
```
pip install torchvision
pip install torch
pip install matplotlib
```

## Training data

Training/test data are taken from torchvision.datsets:
```
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)

test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)
```

## Sample test data
Dimensions: 28x28 px

![image](https://github.com/Chara1236/MNIST-classification-with-Pytorch/assets/53840675/94db4b03-c80d-45bd-b535-a0b5a0a4b073)

## CNN model definition
The CNN model is defined using the neural network module of torch (torch.nn):
```
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return F.softmax(x)
```

## Results
#### Accuracy 58069/60000 (97)%



Over 10 epochs:

Epoch 1: average loss: 0.0154, Accuracy 55303/60000 (92)%

Epoch 5: average loss: 0.0150, Accuracy 57553/60000 (96)%

Epoch 10: average loss: 0.0149, Accuracy 58069/60000 (97)%

Correct prediction:

![samplePred](https://github.com/Chara1236/MNIST-classification-with-Pytorch/assets/53840675/c606ce7e-be2f-4ab6-a011-693ab59cb585)

Incorrect prediction:

![image](https://github.com/Chara1236/MNIST-classification-with-Pytorch/assets/53840675/0275cfcb-2fb6-46b9-b20a-ccccac12ecb2)



