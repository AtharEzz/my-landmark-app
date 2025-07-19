import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        
  
        
        #Convolutional layers ( feature extractor)
        self.features= nn.Sequential(
        
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (32, 112, 112)
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (64, 56, 56) 
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=2, stride=2) , #Output: (128, 28, 28) 
            
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=2, stride=2), #Output: (256, 14, 14) 
            
            )
            
        #The Classifier ( Fully Connected Layers) 
        self.classifier= nn.Sequential(
        
            nn.Flatten(), 
            nn.Linear(256*14*14, 512), 
            nn.ReLU(), 
            nn.Dropout(p=dropout), 
            nn.Linear(512, num_classes)
        
        )

           
            
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x= self.features(x)
        x= self.classifier(x)
        return x
        


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2, num_workers=0)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
