import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
# FT-Transformer Implementation
class FTTransformer(nn.Module):
    def __init__(self, num_features, cat_cardinalities, d_token=192, n_blocks=6, n_heads=8, attention_dropout=0.2, ffn_dropout=0.2, num_classes=2):
        super(FTTransformer, self).__init__()

        self.num_classes = num_classes  # 클래스 개수 저장

        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(num_categories, d_token) for num_categories in cat_cardinalities
        ])

        # Linear layer for numerical features
        self.num_linear = nn.Linear(num_features, d_token)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_token,
                nhead=n_heads,
                dim_feedforward=4 * d_token,
                dropout=attention_dropout,
                activation='gelu'
            ) for _ in range(n_blocks)
        ])

        self.ffn_dropout = nn.Dropout(ffn_dropout)
        if num_classes == 2:
            self.output_layer = nn.Linear(d_token, 1)  # Binary classification
        elif num_classes > 2:
            self.output_layer = nn.Linear(d_token, num_classes)  # Multi classification

    def forward(self, x_num, x_cat):
        # Numerical feature embedding
        x_num = self.num_linear(x_num)

        # Categorical feature embedding
        x_cat = [embed(x_cat[:, i]) for i, embed in enumerate(self.cat_embeddings)]
        x_cat = torch.stack(x_cat, dim=1)

        # Combine numerical and categorical embeddings
        x = x_num.unsqueeze(1) + x_cat

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Pooling and output
        x = x.mean(dim=1)
        x = self.ffn_dropout(x)
        x = self.output_layer(x)

        return x

