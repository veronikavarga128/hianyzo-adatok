import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, TransformerMixin


class _DAE(nn.Module):
    def __init__(self, n_in, n_out, hidden, latent):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class DenoisingAutoencoderImputer(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        sizes,
        hidden=64,
        latent=16,
        epochs=150,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-5,
        mask_rate=0.25,
        random_state=0,
        writer=None,
    ):
        self.sizes = sizes
        self.hidden = hidden
        self.latent = latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.mask_rate = mask_rate
        self.random_state = random_state
        self.writer = writer

    # bemenet/kimenet

    def _build_input(self, X, known):
        n = X.shape[0]
        parts = []
        for j, size in enumerate(self.sizes):
            oh = np.zeros((n, size), dtype=np.float32)
            rows = np.where(known[:, j])[0]
            oh[rows, X[rows, j].astype(int)] = 1.0
            parts.append(oh)
        num = (X[:, self.n_cat_:] - self.mean_) / self.std_
        num = np.where(known[:, self.n_cat_:], num, 0.0).astype(np.float32)
        parts.append(num)
        parts.append(known.astype(np.float32))
        return np.concatenate(parts, axis=1)

    # sklearn API

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        known = ~np.isnan(X)
        self.n_cat_ = len(self.sizes)
        n_num = X.shape[1] - self.n_cat_
        num = X[:, self.n_cat_:]
        with np.errstate(all="ignore"):
            mean = np.nanmean(num, axis=0)
            std = np.nanstd(num, axis=0)
        self.mean_ = np.where(np.isnan(mean), 0.0, mean)
        self.std_ = np.where(np.isnan(std) | (std < 1e-8), 1.0, std)

        gen = torch.Generator().manual_seed(self.random_state)
        torch.manual_seed(self.random_state)
        n_in = sum(self.sizes) + n_num + X.shape[1]
        n_out = sum(self.sizes) + n_num
        self.model_ = _DAE(n_in, n_out, self.hidden, self.latent)
        opt = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        Xz = np.where(known, X, 0.0)
        known_t = torch.tensor(known)
        X_t = torch.tensor(Xz, dtype=torch.float32)
        n = X.shape[0]
        offsets = np.cumsum([0] + list(self.sizes))

        self.model_.train()
        step = 0
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            total, count = 0.0, 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb, kb = X_t[idx], known_t[idx]
                drop = (torch.rand(kb.shape, generator=gen) < self.mask_rate) & kb
                if not drop.any():
                    continue
                kin = kb & ~drop
                inp = torch.tensor(
                    self._build_input(xb.numpy(), kin.numpy()), dtype=torch.float32
                )
                out = self.model_(inp)
                loss = self._loss(out, xb, drop, offsets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                count += 1
                step += 1
            if self.writer is not None and count:
                self.writer.add_scalar("loss/train", total / count, epoch)
        self.model_.eval()
        return self

    def _loss(self, out, xb, drop, offsets):
        losses = []
        for j, size in enumerate(self.sizes):
            m = drop[:, j]
            if m.any():
                logits = out[m][:, offsets[j] : offsets[j + 1]]
                target = xb[m, j].long()
                losses.append(nn.functional.cross_entropy(logits, target))
        n_cat_out = offsets[-1]
        for k in range(self.n_cat_, xb.shape[1]):
            m = drop[:, k]
            if m.any():
                pred = out[m, n_cat_out + (k - self.n_cat_)]
                target = (xb[m, k] - float(self.mean_[k - self.n_cat_])) / float(
                    self.std_[k - self.n_cat_]
                )
                losses.append(nn.functional.mse_loss(pred, target))
        return torch.stack(losses).mean()

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        known = ~np.isnan(X)
        inp = torch.tensor(self._build_input(X, known), dtype=torch.float32)
        with torch.no_grad():
            out = self.model_(inp).numpy()
        result = X.copy()
        offsets = np.cumsum([0] + list(self.sizes))
        for j in range(len(self.sizes)):
            miss = ~known[:, j]
            if miss.any():
                logits = out[miss][:, offsets[j] : offsets[j + 1]]
                result[miss, j] = logits.argmax(axis=1)
        for k in range(self.n_cat_, X.shape[1]):
            miss = ~known[:, k]
            if miss.any():
                col = out[miss, offsets[-1] + (k - self.n_cat_)]
                result[miss, k] = col * self.std_[k - self.n_cat_] + self.mean_[k - self.n_cat_]
        return result