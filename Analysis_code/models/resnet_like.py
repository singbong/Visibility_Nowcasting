import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
# ResNet-like Implementation
class ResNetLike(nn.Module):
    def __init__(self, input_dim, d_main=128, d_hidden=64, n_blocks=4, dropout_first=0.25, dropout_second=0.0, num_classes=2):
        super(ResNetLike, self).__init__()

        self.num_classes = num_classes  # 클래스 개수 저장

        # Input layer
        self.input_layer = nn.Linear(input_dim, d_main)

        # Residual blocks
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_main, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout_first),
                nn.Linear(d_hidden, d_main),
                nn.Dropout(dropout_second)
            ) for _ in range(n_blocks)
        ])
        
        if num_classes == 2:
            self.output_layer = nn.Linear(d_main, 1)  # Binary classification
        elif num_classes > 2:
            self.output_layer = nn.Linear(d_main, num_classes)  # Multi classification

    def forward(self, x_num, x_cat):  # 두 개의 입력을 받음
        x = torch.cat([x_num, x_cat], dim=1)  # 두 개를 하나로 합쳐서 모델에 전달
        x = F.relu(self.input_layer(x))
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        x = self.output_layer(x)
        
        return x

