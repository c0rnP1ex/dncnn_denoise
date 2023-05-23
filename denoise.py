import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.init as init

class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

class denoise():

    def __init__(self, model, device, name_noisy_img) -> None:
        self.device = device
        self.model = model
        self.img = cv2.imread(name_noisy_img, 0)
        pass
    
    def main(self):
        data = []
        data.append(self.img)
        data = np.stack(data, axis=0)
        data = np.expand_dims(data, axis=3)
        data = data.astype(np.float32)/255.0
        tensor = torch.from_numpy(data.transpose((0, 3, 1, 2))).to(device)
        output = self.model(tensor)
        output = output.cpu().detach().numpy().transpose((0, 2, 3, 1))
        return output


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DnCNN().to(device)
    model = torch.load('model.pth')
    model.eval()
    name_noisy_img = 'img_noisy.png'
    denoise_t = denoise(model, device, name_noisy_img)
    cv2.imwrite('img_noisy_denoise.png', denoise_t.main()[0, :, :, 0]*255.0)