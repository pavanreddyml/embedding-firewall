# file: embfirewall/detectors/unsupervised.py
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.svm import OneClassSVM
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import Detector


class CentroidDistance(Detector):
    def __init__(self, name: str = "centroid") -> None:
        super().__init__(name=name)
        self.centroid_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "CentroidDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"CentroidDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        self.centroid_ = np.mean(X_train, axis=0)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.centroid_ is None:
            raise RuntimeError("CentroidDistance not fitted")
        diff = X - self.centroid_[None, :]
        return np.linalg.norm(diff, axis=1)


class KNNDistance(Detector):
    def __init__(self, k: int = 10, name: str = "knn") -> None:
        super().__init__(name=name)
        self.k = int(k)
        self.nn_: Optional[NearestNeighbors] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "KNNDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"KNNDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        k = max(1, int(self.k))
        self.nn_ = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self.nn_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.nn_ is None:
            raise RuntimeError("KNNDistance not fitted")
        dists, _ = self.nn_.kneighbors(X, return_distance=True)
        return np.mean(dists, axis=1)


class OneClassSVMDetector(Detector):
    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: Union[str, float] = "scale",
        name: str = "ocsvm",
    ) -> None:
        super().__init__(name=name)
        self.nu = float(nu)
        self.kernel = str(kernel)
        self.gamma = gamma
        self.model_: Optional[OneClassSVM] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        self.model_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("OneClassSVMDetector not fitted")
        # decision_function: + for inliers, - for outliers => negate
        return -self.model_.decision_function(X).reshape(-1)


class IsolationForestDetector(Detector):
    def __init__(
        self,
        n_estimators: int = 200,
        max_samples: Union[str, int, float] = "auto",
        contamination: Union[str, float] = "auto",
        random_state: Optional[int] = 0,
        n_jobs: int = -1,
        name: str = "iforest",
        **_ignored: object,
    ) -> None:
        # **_ignored keeps backward-compat if configs pass extra keys (won't crash)
        super().__init__(name=name)
        self.n_estimators = int(n_estimators)
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.model_: Optional[IsolationForest] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("IsolationForestDetector not fitted")
        # score_samples: higher => more normal, so negate
        return -self.model_.score_samples(X).reshape(-1)


class MahalanobisDistance(Detector):
    def __init__(self, name: str = "mahalanobis") -> None:
        super().__init__(name=name)
        self.mean_: Optional[np.ndarray] = None
        self.precision_: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "MahalanobisDistance":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(
                f"MahalanobisDistance.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}"
            )
        lw = LedoitWolf().fit(X_train)
        self.mean_ = lw.location_
        self.precision_ = lw.precision_
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisDistance not fitted")
        diff = X - self.mean_[None, :]
        # squared Mahalanobis distance
        m = np.einsum("ij,jk,ik->i", diff, self.precision_, diff)
        return m.reshape(-1)


class LocalOutlierFactorDetector(Detector):
    def __init__(
        self,
        n_neighbors: int = 35,
        leaf_size: int = 30,
        name: str = "lof",
        n_jobs: int = -1,
    ) -> None:
        super().__init__(name=name)
        self.n_neighbors = int(n_neighbors)
        self.leaf_size = int(leaf_size)
        self.n_jobs = int(n_jobs)
        self.model_: Optional[LocalOutlierFactor] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "LocalOutlierFactorDetector":
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            leaf_size=self.leaf_size,
            metric="euclidean",
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("LocalOutlierFactorDetector not fitted")
        # decision_function: + inlier, - outlier => negate
        return -self.model_.decision_function(X).reshape(-1)


class PCAReconstructionError(Detector):
    def __init__(
        self,
        n_components: int = 64,
        whiten: bool = False,
        random_state: Optional[int] = 0,
        name: str = "pca",
    ) -> None:
        super().__init__(name=name)
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.random_state = random_state
        self.pca_: Optional[PCA] = None

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "PCAReconstructionError":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"PCAReconstructionError.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        n = max(1, min(int(self.n_components), int(X_train.shape[1])))
        self.pca_ = PCA(n_components=n, whiten=self.whiten, random_state=self.random_state)
        self.pca_.fit(X_train)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAReconstructionError not fitted")
        Z = self.pca_.transform(X)
        X_hat = self.pca_.inverse_transform(Z)
        err = X - X_hat
        return np.sum(err * err, axis=1).reshape(-1)


def _activation(name: str) -> nn.Module:
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "gelu":
        return nn.GELU()
    if n in ("elu", "elu+"):
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_mlp(dims: Sequence[int], activation: nn.Module, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class _AutoencoderNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        latent_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        act = _activation(activation)
        enc_dims = [input_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]
        self.encoder = _build_mlp(enc_dims, act, dropout)
        self.decoder = _build_mlp(dec_dims, act, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AutoencoderDetector(Detector):
    """
    Simple MLP autoencoder trained with reconstruction loss.
    Higher reconstruction error => more anomalous.
    """

    def __init__(
        self,
        hidden_dims: Iterable[int] = (256, 128),
        latent_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 30,
        device: Optional[str] = None,
        seed: Optional[int] = 0,
        name: str = "autoencoder",
    ) -> None:
        super().__init__(name=name)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.device = device
        self.seed = seed
        self.model_: Optional[_AutoencoderNet] = None
        self.device_: Optional[torch.device] = None

    def _setup(self, input_dim: int) -> None:
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_ = dev
        self.model_ = _AutoencoderNet(input_dim, self.hidden_dims, self.latent_dim, self.activation, self.dropout).to(dev)

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "AutoencoderDetector":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"AutoencoderDetector.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        self._setup(X_train.shape[1])
        assert self.model_ is not None and self.device_ is not None
        dataset = TensorDataset(_to_tensor(X_train, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, _ = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("AutoencoderDetector not fitted")
        self.model_.eval()
        dataset = TensorDataset(_to_tensor(X, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        scores: List[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                recon, _ = self.model_(batch)
                err = torch.sum((batch - recon) ** 2, dim=1)
                scores.append(err.cpu().numpy())
        return np.concatenate(scores, axis=0)


class _VAENet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        latent_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        act = _activation(activation)
        enc_dims = [input_dim, *hidden_dims]
        self.encoder = _build_mlp(enc_dims, act, dropout)
        enc_out = enc_dims[-1]
        self.mu = nn.Linear(enc_out, latent_dim)
        self.logvar = nn.Linear(enc_out, latent_dim)
        dec_dims = [latent_dim, *reversed(hidden_dims), input_dim]
        self.decoder = _build_mlp(dec_dims, act, dropout)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


class VariationalAutoencoderDetector(Detector):
    """
    VAE detector: reconstruction + KL divergence acts as anomaly score.
    """

    def __init__(
        self,
        hidden_dims: Iterable[int] = (256, 128),
        latent_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 40,
        beta: float = 1.0,
        device: Optional[str] = None,
        seed: Optional[int] = 0,
        name: str = "vae",
    ) -> None:
        super().__init__(name=name)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.latent_dim = int(latent_dim)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.beta = float(beta)
        self.device = device
        self.seed = seed
        self.model_: Optional[_VAENet] = None
        self.device_: Optional[torch.device] = None

    def _setup(self, input_dim: int) -> None:
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_ = dev
        self.model_ = _VAENet(input_dim, self.hidden_dims, self.latent_dim, self.activation, self.dropout).to(dev)

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "VariationalAutoencoderDetector":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(
                f"VariationalAutoencoderDetector.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}"
            )
        self._setup(X_train.shape[1])
        assert self.model_ is not None and self.device_ is not None
        dataset = TensorDataset(_to_tensor(X_train, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model_.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                optimizer.zero_grad()
                recon, mu, logvar = self.model_(batch)
                recon_loss = torch.sum((batch - recon) ** 2, dim=1).mean()
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + self.beta * kl
                loss.backward()
                optimizer.step()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None or self.device_ is None:
            raise RuntimeError("VariationalAutoencoderDetector not fitted")
        self.model_.eval()
        dataset = TensorDataset(_to_tensor(X, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        scores: List[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                recon, mu, logvar = self.model_(batch)
                recon_err = torch.sum((batch - recon) ** 2, dim=1)
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
                scores.append((recon_err + self.beta * kl).cpu().numpy())
        return np.concatenate(scores, axis=0)


class _Discriminator(nn.Module):
    def __init__(self, dims: Sequence[int], activation: str, dropout: float) -> None:
        super().__init__()
        act = _activation(activation)
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act.__class__())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*layers)
        self.out = nn.Linear(dims[-2], dims[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_extractor(x)
        return self.out(h)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class _Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: Iterable[int], activation: str, dropout: float) -> None:
        super().__init__()
        act = _activation(activation)
        dims = [noise_dim, *hidden_dims, output_dim]
        self.net = _build_mlp(dims, act, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GANDiscriminatorDetector(Detector):
    """Train a small GAN and use the discriminator confidence as the anomaly score."""

    def __init__(
        self,
        hidden_dims: Iterable[int] = (256, 128),
        noise_dim: int = 64,
        activation: str = "gelu",
        dropout: float = 0.0,
        lr_d: float = 2e-4,
        lr_g: float = 2e-4,
        batch_size: int = 256,
        epochs: int = 40,
        feature_match_weight: float = 10.0,
        device: Optional[str] = None,
        seed: Optional[int] = 0,
        name: str = "gan",
    ) -> None:
        super().__init__(name=name)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.noise_dim = int(noise_dim)
        self.activation = activation
        self.dropout = float(dropout)
        self.lr_d = float(lr_d)
        self.lr_g = float(lr_g)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.feature_match_weight = float(feature_match_weight)
        self.device = device
        self.seed = seed
        self.disc_: Optional[_Discriminator] = None
        self.gen_: Optional[_Generator] = None
        self.device_: Optional[torch.device] = None

    def _setup(self, input_dim: int) -> None:
        if self.seed is not None:
            torch.manual_seed(int(self.seed))
        dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_ = dev
        disc_dims = [input_dim, *self.hidden_dims, 1]
        self.disc_ = _Discriminator(disc_dims, self.activation, self.dropout).to(dev)
        self.gen_ = _Generator(self.noise_dim, input_dim, self.hidden_dims, self.activation, self.dropout).to(dev)

    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> "GANDiscriminatorDetector":
        if not isinstance(X_train, np.ndarray) or X_train.ndim != 2 or X_train.shape[0] == 0:
            raise ValueError(f"GANDiscriminatorDetector.fit expects 2D non-empty array, got {getattr(X_train, 'shape', None)}")
        self._setup(X_train.shape[1])
        assert self.disc_ is not None and self.gen_ is not None and self.device_ is not None
        dataset = TensorDataset(_to_tensor(X_train, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        opt_d = torch.optim.Adam(self.disc_.parameters(), lr=self.lr_d, betas=(0.5, 0.999))
        opt_g = torch.optim.Adam(self.gen_.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            for (real,) in loader:
                # Train discriminator
                opt_d.zero_grad()
                noise = torch.randn(real.size(0), self.noise_dim, device=self.device_)
                fake = self.gen_(noise).detach()
                logits_real = self.disc_(real)
                logits_fake = self.disc_(fake)
                loss_d = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
                loss_d.backward()
                opt_d.step()

                # Train generator
                opt_g.zero_grad()
                noise = torch.randn(real.size(0), self.noise_dim, device=self.device_)
                fake = self.gen_(noise)
                logits_fake = self.disc_(fake)
                loss_g = bce(logits_fake, torch.ones_like(logits_fake))
                if self.feature_match_weight > 0:
                    with torch.no_grad():
                        real_feat = self.disc_.features(real).detach()
                    fake_feat = self.disc_.features(fake)
                    fm = torch.mean(torch.abs(fake_feat.mean(dim=0) - real_feat.mean(dim=0)))
                    loss_g = loss_g + self.feature_match_weight * fm
                loss_g.backward()
                opt_g.step()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.disc_ is None or self.device_ is None:
            raise RuntimeError("GANDiscriminatorDetector not fitted")
        self.disc_.eval()
        dataset = TensorDataset(_to_tensor(X, self.device_))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        scores: List[np.ndarray] = []
        with torch.no_grad():
            for (batch,) in loader:
                logits = self.disc_(batch)
                prob_real = torch.sigmoid(logits).squeeze(dim=1)
                scores.append((1.0 - prob_real).cpu().numpy())
        return np.concatenate(scores, axis=0)
