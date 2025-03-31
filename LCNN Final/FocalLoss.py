class FocalLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.5, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, inputs, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        F_loss = alpha_factor * (1 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
