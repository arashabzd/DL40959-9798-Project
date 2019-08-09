import torch
from torch import nn
from torch.nn import functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # N * 1 * 60 * 400
        self.conv1 = nn.Conv2d(1, 64, (3, 3), (1, 1), (0, 0))
        self.mp1 = nn.MaxPool2d((2, 2), (2, 2), (1, 1))
        # N * 64 * 30 * 200
        self.conv2 = nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1))
        self.mp2 = nn.MaxPool2d((2, 2), (2, 2), (1, 0))
        # N * 128 * 16 * 100
        self.conv3 = nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1))
        self.bn1 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.mp3 = nn.MaxPool2d((2, 1), (2, 1), (0, 0))
        # N * 256 * 8 * 100
        self.conv5 = nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1))
        self.bn2 = nn.BatchNorm2d(512)
        self.mp4 = nn.MaxPool2d((1, 2), (1, 2), (0, 0))
        # N * 512 * 8 * 50
        self.conv6 = nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1))
        self.bn3 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.mp1(F.relu(self.conv1(x)))
        x = self.mp2(F.relu(self.conv2(x)))
        x = self.mp3(F.relu(self.conv4(F.relu(self.bn1(self.conv3(x))))))
        x = self.mp4(F.relu(self.bn2(self.conv5(x))))
        x = self.bn3(self.conv6(x))
        return x


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()

        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True,
                          bidirectional=True)
        # self.h = nn.Parameter(torch.randn(num_layers * 2, 1, hidden_dim,
        #                                   requires_grad=True))

    def forward(self, x):
        # h = self.h.repeat(1, x.shape[0], 1)
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = torch.transpose(x, 0, 1)
        # x = torch.stack([self.gru(xi, h)[0] for xi in x], dim=1)
        x = torch.stack([self.gru(xi)[0] for xi in x], dim=1)
        return x


class Attention(nn.Module):

    def __init__(self, hidden_dim, context_dim):
        super(Attention, self).__init__()

        self.e = nn.Sequential(nn.Linear(in_features=hidden_dim + context_dim,
                                         out_features=hidden_dim),
                               nn.Tanh(),
                               nn.Linear(in_features=hidden_dim,
                                         out_features=1))

    def forward(self, h, v):
        N, H, W, C = v.shape
        v = v.view(N, -1, C)
        h = torch.unsqueeze(h, 1)
        h = h.repeat(1, H * W, 1)
        c = self.e(torch.cat([v, h], dim=2))
        c = F.softmax(c, dim=1)
        c = c * v
        c = c.sum(dim=1)
        return c


class Decoder(nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=input_dim,
                                      embedding_dim=embedding_dim)
        self.gru = nn.GRUCell(input_size=embedding_dim,
                              hidden_size=hidden_dim)
        # self.h = nn.Parameter(torch.randn(1, hidden_dim, requires_grad=True))
        self.attention = Attention(hidden_dim, 2*hidden_dim)
        self.o = nn.Sequential(nn.Linear(in_features=hidden_dim + 2*hidden_dim,
                                         out_features=hidden_dim),
                               nn.Tanh())
        self.y_hat = nn.Linear(in_features=hidden_dim,
                               out_features=input_dim)

    def forward(self, v, y):

        # h = self.h.repeat(v.shape[0], 1)
        h = None
        y_hat = []

        if self.training:
            y = self.embedding(y)
            y = torch.transpose(y, 0, 1)
            for yt in y:
                h = self.gru(yt, h)
                c = self.attention(h, v)
                o = self.o(torch.cat([h, c], dim=1))
                y_hat.append(self.y_hat(o))

        else:
            for t in range(200):
                # if y.item() == 0: break
                y = self.embedding(y)
                h = self.gru(y, h)
                c = self.attention(h, v)
                o = self.o(torch.cat([h, c], dim=1))
                y = self.y_hat(o)
                y_hat.append(y)
                y = y.argmax(dim=1)

        return torch.stack(y_hat, dim=1)


class Model(nn.Module):

    def __init__(self, encoder_input_dim, decoder_input_dim,
                 hidden_dim, embedding_dim):
        super(Model, self).__init__()

        self.cnn = CNN()
        self.encoder = Encoder(input_dim=encoder_input_dim,
                               hidden_dim=hidden_dim)

        self.decoder = Decoder(input_dim=decoder_input_dim,
                               embedding_dim=embedding_dim,
                               hidden_dim=hidden_dim)

    def forward(self, x, y=None):
        x = self.cnn(x)
        x = self.encoder(x)
        return self.decoder(x, y)

    def save(self, path):
        torch.save(self.state_dict(), path + 'model.chkpt')

    def load(self, path):
        self.load_state_dict(torch.load(path + 'model.chkpt'))
