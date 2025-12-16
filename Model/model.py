import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super(SmallCNN, self).__init__()

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            
            # image size : 48,48, 3 -> 48,48,32
            # parameters :  weights =1*3*3*32 = 288 , bias =32 = 320
            
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), #image size : 48,48,32 ->24,24,32

            nn.Conv2d(32, 64, 3, padding=1),
            # image size : 24,24, 32 -> 24,24,64
            # parameters :  weights =32*3*3*64 = 18,432 , bias =64 = 18,496
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),#image size : 24,24,64 ->12,12,64
            nn.Conv2d(64, 128, 3, padding=1),
            # image size : 12,12, 64 -> 12,12,128
            # parameters :  weights =64*3*3*128 = 73,728 , bias =128 = 73,856
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),#image size : 12,12,128 ->6,6,128
        
        # parameters= 320 + 18,496+ 73,856 = 92,672
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),
            # parameters :  weights =128*6*6*128 = 589,952 , bias =128 = 590,080
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes) # parameters :  weights =128*5 =640  , bias =5 = 645
        # parameters= 590,080 + 645 = 590,725
        
        # all parameters= 590,725 + 92,672 =683,397
        # optimum parameters = 10*dataset_length < parameters < 10*dataset_length 
        # dataset size 50k so parmeters range should be from half mil to mil
        )

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.classifier(x)
        return x
#

##