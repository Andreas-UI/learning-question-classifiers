import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, num_filters, kernels, num_classes
    ):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings + 1, embedding_dim=embedding_dim
        )
        self.conv1 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernels[0],
            padding="same",
        )
        self.conv2 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernels[1],
            padding="same",
        )
        self.conv3 = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=num_filters,
            kernel_size=kernels[2],
            padding="same",
        )
        self.fc = nn.Linear(in_features=3 * num_filters, out_features=num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        h1 = F.relu(self.conv1(x))
        h1 = F.max_pool1d(h1, kernel_size=h1.size(2)).squeeze(2)
        h2 = F.relu(self.conv2(x))
        h2 = F.max_pool1d(h2, kernel_size=h2.size(2)).squeeze(2)
        h3 = F.relu(self.conv3(x))
        h3 = F.max_pool1d(h3, kernel_size=h3.size(2)).squeeze(2)
        h = torch.cat([h1, h2, h3], dim=1)
        return self.fc(h)
