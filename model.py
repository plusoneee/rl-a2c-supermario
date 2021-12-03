import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class ConvDQN(nn.Module):

    def __init__(self, img_h, img_w, output_dim):
        super(ConvDQN, self).__init__()

        def conv2d_out_size(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        self.output_channel = 64
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, self.output_channel, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        convw = conv2d_out_size(conv2d_out_size(conv2d_out_size(img_h)))
        convh = conv2d_out_size(conv2d_out_size(conv2d_out_size(img_w)))
        
        linear_input_size = convw * convh * self.output_channel
        self.dense = nn.Linear(linear_input_size, output_dim)

    def forward(self, x):
        output = self.features(x)
        output = output.view(output.size(0), -1)
        return self.dense(output)

if __name__ == '__main__':
    
    # create screen image
    screen_shape = (240, 256, 3)
    # Transpose it into torch order (CHW).
    random_state = np.random.random(screen_shape).transpose((2, 0, 1))
    screen = torch.Tensor(random_state)
    screen = torch.unsqueeze(screen, 0)
    model = ConvDQN(img_h=240, img_w=256, output_dim=256)
    
    