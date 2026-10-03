"""Neural components of the TS-augmentation pipeline (32x32 RGB, pixel space [0,1]).

- TeacherCNN : small CNN classifier; scores candidates by class probability.
- StudentNet : CNN encoder + latent-conditioned decoder (generator) + a small
               classification head. The head is trained on the student's own
               generations, so each student can also *score* candidates
               (peer scoring inside the ensemble).
- ProtoNet   : CNN feature extractor used for the final downstream evaluation
               (prototypes = per-class mean features, cosine similarity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_blocks(channels, with_pool_last=False):
    """3 conv blocks (32->16->8, optional ->4) with BN+ReLU."""
    c1, c2, c3 = channels
    layers = [
        nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c1),
        nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c1), nn.MaxPool2d(2),
        nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c2),
        nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c2), nn.MaxPool2d(2),
        nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c3),
        nn.Conv2d(c3, c3, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(c3),
    ]
    if with_pool_last:
        layers.append(nn.MaxPool2d(2))
    return layers


class TeacherCNN(nn.Module):
    """32x32 -> 16 -> 8 -> 4; features (c3*4*4) -> 256 -> logits."""

    def __init__(self, n_classes, channels=(32, 64, 128), feat_dim=256, dropout=0.2):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(*_conv_blocks(channels, with_pool_last=True))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c3 * 4 * 4, feat_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(feat_dim, n_classes))
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.net(x))


class StudentNet(nn.Module):
    """Encoder-decoder generator with latent code z and a peer-classification head.

    forward path:  x --Enc--> f (256)  ;  (f, z) --Dec--> x' in [0,1]
    scoring:       x --Enc--> f --cls--> logits (peer vote on candidates)
    """

    def __init__(self, n_classes, channels=64, latent_dim=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        c1, c2 = channels, channels * 2
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(*_conv_blocks((3, c1, c2), with_pool_last=False))
        self.enc_fc = nn.Linear(c2 * 8 * 8, 256)
        # decoder: Linear -> (c2,4,4) -> ConvT x3 -> (3,32,32)
        self.dec_fc = nn.Linear(256 + latent_dim, c2 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(c1),
            nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1), nn.ReLU(), nn.BatchNorm2d(c1),
            nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1), nn.Sigmoid())
        self.cls = nn.Linear(256, n_classes)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02, generator=g)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02, generator=g)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        return F.relu(self.enc_fc(self.encoder(x).flatten(1)))

    def generate(self, x, z):
        """x: [B,3,32,32] source images; z: [B,latent_dim] latents -> [B,3,32,32]."""
        v = self.dec_fc(torch.cat([self.encode(x), z], dim=1))
        b = v.shape[0]
        v = v.view(b, -1, 4, 4)
        for layer in self.decoder:
            v = layer(v)
        return v

    def classify(self, x):
        return self.cls(self.encode(x))


class ProtoNet(nn.Module):
    """Feature extractor for prototype-based few-shot evaluation.

    `forward` returns unit-usable features; `logits` adds a trainable linear
    head used only during feature-extractor training (CE).
    """

    def __init__(self, n_classes, channels=(32, 64, 128), feat_dim=128):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(*_conv_blocks(channels, with_pool_last=False))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c3 * 8 * 8, feat_dim), nn.ReLU())
        self.classifier = nn.Linear(feat_dim, n_classes)
        self.feat_dim = feat_dim
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.net(x))

    def logits(self, x):
        return self.classifier(self.forward(x))
