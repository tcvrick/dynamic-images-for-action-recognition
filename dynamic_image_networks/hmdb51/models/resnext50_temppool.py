import torchsummary
import torchvision
import torch.nn as nn


def get_model(num_classes):
    return FusedResNextTempPool(num_classes)


class FusedResNextTempPool(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        # Define the ResNet.
        resnet = torchvision.models.resnext50_32x4d(pretrained=True)
        resnet.fc = nn.Sequential()

        # Define the classifier.
        self.features = resnet
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(inplace=True),
                                nn.Dropout(p=0.5),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        batch_size, segment_size, c, h, w = x.shape
        num_fc_input_features = self.fc[0].in_features

        # Time distribute the inputs.
        x = x.view(batch_size * segment_size, c, h, w)
        x = self.features(x)

        # Re-structure the data and then temporal max-pool.
        x = x.view(batch_size, segment_size, num_fc_input_features)
        x = x.max(dim=1).values

        # FC.
        x = self.fc(x)
        return x


def main():
    _model = get_model(num_classes=51)
    torchsummary.summary(_model, input_size=(10, 3, 224, 224), device='cpu')


if __name__ == '__main__':
    main()
