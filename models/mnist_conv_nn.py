import torchvision
import torch.nn as nn
from torchsummary import summary

class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters, kernel_size, linear_width):
        super().__init__()
        conv_out_width = 28 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        self.seq = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc1_indim, linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(linear_width, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.seq(x)
class CIFARConvNet(nn.Module):
    #implement resnet18
    def __init__(self):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10) 

if __name__ == "__main__":
    # Test MNISTConvNet
    model = MNISTConvNet(3, 5, 64)
    model.to("cuda")
    # print summary
    summary(model,(1,28,28))
    

    # Test CIFARConvNet
    # model = CIFARConvNet()
    # model.to("cuda")
    # # print summary
    # summary(model,(3,224,224))
    