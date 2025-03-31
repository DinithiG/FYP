class LCNN(nn.Module):
    def __init__(self, input_dim=124 + 7 + 6 + 128 + 384, num_classes=1):  # Removed 12 for Chroma
        super(LCNN, self).__init__()
        
        self.features = nn.Sequential(
            ResidualBlock(input_dim, 256),  # Increased initial channels
            AttentionBlock(256),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),  # Increased dropout
            
            ResidualBlock(256, 512),
            AttentionBlock(512),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.4),
            
            ResidualBlock(512, 768),
            AttentionBlock(768),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.5),
            
            ResidualBlock(768, 1024),
            AttentionBlock(1024),
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.6)  # Increased dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        return x
