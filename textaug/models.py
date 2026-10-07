"""Neural components of the TEXT TS-augmentation pipeline.

- TextEncoder : Embedding -> BiLSTM -> final hidden states (2*hidden) as features.
- TeacherText : TextEncoder + linear head; scores candidates by class probability.
- StudentText : encoder + latent-conditioned LSTM decoder (sequence generator)
                + classification head (peer scoring).
- ProtoText   : feature extractor for prototype-based few-shot evaluation.

Token ids: 0 = <UNK>, 1 = <PAD>, 2 = <BOS> (see textaug/data.py).
Sequences are right-padded with PAD and encoded via packed sequences, so
padding never leaks into the features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

PAD, UNK, BOS = 1, 0, 2


class TextEncoder(nn.Module):
    """Bidirectional LSTM encoder (1-2 layers); feature = concatenated final
    hidden states of the last layer."""

    def __init__(self, vocab_size, embed_dim=64, hidden=64, n_layers=1,
                 dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.n_layers = max(1, int(n_layers))
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        # inter-layer dropout is only meaningful (and valid) with >1 layer
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=self.n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if self.n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden * 2, hidden * 2)
        self.act = nn.ReLU()

    def forward(self, x):
        # x: [B, L] long tensor, right-padded with PAD
        lens = (x != PAD).sum(1).clamp(min=1).cpu()
        emb = self.embed(x)
        packed = pack_padded_sequence(emb, lens, batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        f = torch.cat([h[0], h[1]], dim=1)        # [B, 2*hidden]
        return self.act(self.fc(f))


class TeacherText(nn.Module):
    """Class scorer. Trained on the current dataset D_t; scores candidates by
    the softmax probability of the claimed class. Never generates data.

    The teacher's width (hidden) and depth (n_layers) are decoupled from the
    students': a wider/deeper scorer is allowed to be more expensive than the
    generators, which are the runtime bottleneck."""

    def __init__(self, vocab_size, n_classes, embed_dim=64, hidden=64,
                 dropout=0.2, n_layers=1):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden, n_layers,
                                   dropout)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(hidden * 2, n_classes))
        self.n_classes = n_classes

    def forward(self, x):
        return self.head(self.encoder(x))


class ProtoText(nn.Module):
    """Prototypical-network evaluator (downstream classifier)."""

    def __init__(self, vocab_size, n_classes, embed_dim=64, hidden=64,
                 n_layers=1, dropout=0.0):
        super().__init__()
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden, n_layers,
                                   dropout)
        self.classifier = nn.Linear(hidden * 2, n_classes)
        self.feat_dim = hidden * 2
        self.n_classes = n_classes

    def forward(self, x):
        return self.encoder(x)

    def logits(self, x):
        return self.classifier(self.forward(x))


class StudentText(nn.Module):
    """Latent-conditioned sequence generator + peer scorer.

    - encoder: TextEncoder (same architecture, independent parameters)
    - decoder: LSTM, initial state from (source feature, latent z);
      autoregressive token sampling, shares the encoder's embedding table
    - cls head: scores its own generations (peer scoring in the ensemble)
    """

    def __init__(self, vocab_size, n_classes, embed_dim=64, hidden=64,
                 latent_dim=8, seed=0, n_layers=1, dropout=0.0):
        super().__init__()
        self.hidden = hidden
        self.latent_dim = latent_dim
        self.encoder = TextEncoder(vocab_size, embed_dim, hidden, n_layers,
                                   dropout)
        self.init_h = nn.Linear(hidden * 2 + latent_dim, hidden)
        self.decoder = nn.LSTM(embed_dim, hidden, batch_first=True)
        self.tok_head = nn.Linear(hidden, vocab_size)
        self.cls = nn.Sequential(nn.Dropout(dropout),
                                 nn.Linear(hidden * 2, n_classes))

    def encode(self, x):
        return self.encoder(x)

    def _init_state(self, f, z):
        h0 = torch.tanh(self.init_h(torch.cat([f, z], dim=1)))  # [B, H]
        # batched (3-D) LSTM input requires a 3-D initial state: [layers, B, H]
        h0 = h0.unsqueeze(0)
        return h0, h0.clone()

    def teacher_forced_logits(self, x_in, z, target):
        """Decode `target` (teacher forcing): input BOS + target[:, :-1]."""
        f = self.encode(x_in)
        state = self._init_state(f, z)
        bos = torch.full((x_in.shape[0], 1), BOS, dtype=torch.long,
                         device=x_in.device)
        inp = torch.cat([bos, target[:, :-1]], dim=1)
        out, _ = self.decoder(self.encoder.embed(inp), state)
        return self.tok_head(out)                       # [B, L, V]

    @torch.no_grad()
    def generate(self, x_in, z, max_len, temperature=1.0):
        """Autoregressive sampling; returns [B, max_len] (PAD right padding)."""
        f = self.encode(x_in)
        state = self._init_state(f, z)
        B, dev = x_in.shape[0], x_in.device
        tokens = torch.full((B, 1), BOS, dtype=torch.long, device=dev)
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_len):
            out, state = self.decoder(self.encoder.embed(tokens), state)
            logits = self.tok_head(out[:, -1])
            prob = torch.softmax(logits / max(temperature, 1e-3), dim=1)
            prob[:, PAD] = 0.0
            nxt = torch.multinomial(prob, 1).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, PAD), nxt)
            finished |= (nxt == PAD)
            tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        return tokens[:, 1:]                             # drop BOS

    def classify(self, x):
        return self.cls(self.encode(x))

