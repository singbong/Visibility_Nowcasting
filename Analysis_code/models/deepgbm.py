import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')
class DeepGBM(nn.Module):
    def __init__(self, num_features, cat_features, d_main=128, d_hidden=64, n_blocks=4, dropout=0.2, num_classes=2):
        super(DeepGBM, self).__init__()

        self.num_classes = num_classes
        
        # 연속형 변수 처리 (Linear)
        self.num_linear = nn.Linear(num_features, d_main)

        # 범주형 변수 처리 (Embedding)
        self.cat_embedding = nn.ModuleList([
            nn.Embedding(cat_size, d_main) for cat_size in cat_features
        ])

        # ResNet-like 블록
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_main, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_main),
                nn.ReLU()
            ) for _ in range(n_blocks)
        ])

        if num_classes == 2:
            self.output_layer = nn.Linear(d_main, 1)  # Binary classification
        elif num_classes > 2:
            self.output_layer = nn.Linear(d_main, num_classes)  # Multi classification

    def forward(self, x_num, x_cat):  # 두 개의 입력을 받음
        x_num = self.num_linear(x_num)

        # 범주형 변수를 임베딩 후 합산
        x_cat = [embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embedding)]
        x_cat = torch.stack(x_cat, dim=1).sum(dim=1)  
        x = x_num + x_cat  # 연속형 + 범주형 결합
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        x = self.output_layer(x)
        
        return x

