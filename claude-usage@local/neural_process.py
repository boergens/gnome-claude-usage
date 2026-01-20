#!/usr/bin/env python3
"""
Neural Process for Claude Usage Prediction

Predicts when Claude usage will be depleted based on historical consumption patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader


class NeuralProcess(nn.Module):
    """
    Neural Process for curve completion.
    Given context points (t_context, y_context), predicts values at target times t_target.
    """

    def __init__(
        self,
        x_dim: int = 1,
        y_dim: int = 1,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_encoder_layers: int = 2,
        n_decoder_layers: int = 2,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        encoder_layers.append(nn.Linear(x_dim + y_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        for _ in range(n_encoder_layers - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(x_dim + latent_dim, hidden_dim))
        decoder_layers.append(nn.ReLU())
        for _ in range(n_decoder_layers - 1):
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        self.fc_out_mu = nn.Linear(hidden_dim, y_dim)
        self.fc_out_logvar = nn.Linear(hidden_dim, y_dim)

    def encode(self, x_context, y_context):
        xy = torch.cat([x_context, y_context], dim=-1)
        hidden = self.encoder(xy)
        aggregated = hidden.mean(dim=1)
        mu = self.fc_mu(aggregated)
        logvar = self.fc_logvar(aggregated)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x_target, z):
        batch_size, n_target, _ = x_target.shape
        z_expanded = z.unsqueeze(1).expand(-1, n_target, -1)
        xz = torch.cat([x_target, z_expanded], dim=-1)
        hidden = self.decoder(xz)
        mu = self.fc_out_mu(hidden)
        logvar = self.fc_out_logvar(hidden)
        return mu, logvar

    def forward(self, x_context, y_context, x_target, y_target=None):
        mu_z, logvar_z = self.encode(x_context, y_context)
        z = self.reparameterize(mu_z, logvar_z)
        mu_y, logvar_y = self.decode(x_target, z)

        result = {
            'mu_y': mu_y,
            'logvar_y': logvar_y,
            'mu_z': mu_z,
            'logvar_z': logvar_z,
        }

        if y_target is not None:
            var = torch.exp(logvar_y)
            recon_loss = 0.5 * (logvar_y + (y_target - mu_y).pow(2) / var).sum(dim=[-1, -2])
            kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1)

            result['recon_loss'] = recon_loss.mean()
            result['kl_loss'] = kl_loss.mean()
            result['loss'] = result['recon_loss'] + 0.1 * result['kl_loss']

        return result

    def predict(self, x_context, y_context, x_target, n_samples=50):
        if x_context.dim() == 2:
            x_context = x_context.unsqueeze(0)
            y_context = y_context.unsqueeze(0)
            x_target = x_target.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        with torch.no_grad():
            mu_z, logvar_z = self.encode(x_context, y_context)

            all_samples = []
            for _ in range(n_samples):
                z = self.reparameterize(mu_z, logvar_z)
                mu_y, logvar_y = self.decode(x_target, z)
                std_y = torch.exp(0.5 * logvar_y)
                sample = mu_y + std_y * torch.randn_like(mu_y)
                all_samples.append(sample)

            samples = torch.stack(all_samples)
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)

            if squeeze_output:
                mean = mean.squeeze(0)
                std = std.squeeze(0)
                samples = samples.squeeze(1)

            return mean, std, samples


class UsageDataset(Dataset):
    """Dataset of Claude usage curves."""

    def __init__(self, curves: np.ndarray, context_range: tuple = (3, 30), augment: bool = True):
        self.curves = torch.tensor(curves, dtype=torch.float32)
        self.n_curves, self.curve_length = self.curves.shape
        self.context_range = context_range
        self.augment = augment
        self.times = torch.linspace(0, 1, self.curve_length).unsqueeze(-1)

    def __len__(self):
        return self.n_curves * 10

    def __getitem__(self, idx):
        curve_idx = idx % self.n_curves
        curve = self.curves[curve_idx].clone()

        if self.augment:
            curve = curve + torch.randn(1) * 0.02
            curve = curve + torch.randn_like(curve) * 0.01

        curve = curve.clamp(0, 1)

        n_context = torch.randint(self.context_range[0], min(self.context_range[1], self.curve_length), (1,)).item()

        x_context = self.times[:n_context]
        y_context = curve[:n_context].unsqueeze(-1)
        x_target = self.times
        y_target = curve.unsqueeze(-1)

        return x_context, y_context, x_target, y_target


def collate_fn(batch):
    x_contexts, y_contexts, x_targets, y_targets = zip(*batch)
    max_context = max(x.shape[0] for x in x_contexts)

    x_context_padded = []
    y_context_padded = []

    for x, y in zip(x_contexts, y_contexts):
        n = x.shape[0]
        pad_size = max_context - n
        if pad_size > 0:
            x = F.pad(x, (0, 0, 0, pad_size))
            y = F.pad(y, (0, 0, 0, pad_size))
        x_context_padded.append(x)
        y_context_padded.append(y)

    return (
        torch.stack(x_context_padded),
        torch.stack(y_context_padded),
        torch.stack(x_targets),
        torch.stack(y_targets),
    )


class UsagePredictor:
    """High-level interface for usage prediction."""

    def __init__(self, model_path: str = None, data_dir: str = None):
        self.device = 'cpu'
        self.curve_length = 100  # Points per 5-hour session
        self.session_hours = 5.0
        self.times = torch.linspace(0, 1, self.curve_length).unsqueeze(-1)

        if data_dir is None:
            data_dir = os.path.expanduser("~/.claude/usage_history")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if model_path is None:
            model_path = self.data_dir / "np_model.pt"
        self.model_path = Path(model_path)

        self.model = NeuralProcess(hidden_dim=64, latent_dim=16)
        self.load_model()

    def load_model(self):
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        return False

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, self.model_path)

    def record_observation(self, session_pct_used: float, weekly_pct_used: float):
        """Record a usage observation."""
        history_file = self.data_dir / "observations.jsonl"

        observation = {
            "timestamp": datetime.now().isoformat(),
            "session_pct_used": session_pct_used,
            "weekly_pct_used": weekly_pct_used,
        }

        with open(history_file, "a") as f:
            f.write(json.dumps(observation) + "\n")

    def get_historical_curves(self) -> np.ndarray:
        """Load historical curves from observations."""
        history_file = self.data_dir / "observations.jsonl"

        if not history_file.exists():
            return self._generate_synthetic_curves(10)

        observations = []
        with open(history_file) as f:
            for line in f:
                if line.strip():
                    observations.append(json.loads(line))

        if len(observations) < 10:
            return self._generate_synthetic_curves(10)

        # Group observations into sessions (5-hour windows)
        curves = []
        current_curve = []
        last_session_pct = 0

        for obs in observations:
            session_pct = obs['session_pct_used']

            # Detect session reset (new session started)
            if session_pct < last_session_pct - 10:
                if len(current_curve) >= 5:
                    # Normalize curve to 100 points
                    curve = self._normalize_curve(current_curve)
                    curves.append(curve)
                current_curve = []

            # Store as "remaining" (1 - used)
            current_curve.append(1.0 - session_pct / 100.0)
            last_session_pct = session_pct

        if len(curves) < 5:
            return self._generate_synthetic_curves(10)

        return np.array(curves)

    def _normalize_curve(self, curve: list) -> np.ndarray:
        """Normalize a curve to fixed length."""
        curve = np.array(curve)
        if len(curve) == self.curve_length:
            return curve

        x_old = np.linspace(0, 1, len(curve))
        x_new = np.linspace(0, 1, self.curve_length)
        return np.interp(x_new, x_old, curve)

    def _generate_synthetic_curves(self, n: int) -> np.ndarray:
        """Generate synthetic training curves."""
        curves = []
        t = np.linspace(0, 1, self.curve_length)

        for _ in range(n):
            # Different consumption patterns
            pattern = np.random.choice(['linear', 'heavy_early', 'heavy_late', 'moderate'])

            if pattern == 'linear':
                rate = 0.8 + 0.4 * np.random.rand()
                curve = 1 - t ** rate
            elif pattern == 'heavy_early':
                rate = 1.5 + 0.5 * np.random.rand()
                curve = 1 - t ** (1/rate)
            elif pattern == 'heavy_late':
                rate = 0.5 + 0.3 * np.random.rand()
                curve = 1 - t ** rate
            else:  # moderate
                curve = 1 - t

            curve += np.random.randn(self.curve_length) * 0.02
            curve = np.clip(curve, 0, 1)

            # Ensure monotonically decreasing
            for j in range(1, len(curve)):
                if curve[j] > curve[j-1]:
                    curve[j] = curve[j-1] - 0.001

            curves.append(curve)

        return np.array(curves)

    def train(self, epochs: int = 200, verbose: bool = False):
        """Train or update the model."""
        curves = self.get_historical_curves()

        if verbose:
            print(f"Training on {len(curves)} curves...")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        dataset = UsageDataset(curves, augment=True)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            n_batches = 0

            for x_ctx, y_ctx, x_tgt, y_tgt in dataloader:
                optimizer.zero_grad()
                result = self.model(x_ctx, y_ctx, x_tgt, y_tgt)
                loss = result['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/n_batches:.4f}")

        self.model.eval()
        self.save_model()

    def predict_depletion(self, current_session_pct_used: float, n_samples: int = 50) -> dict:
        """
        Predict when session usage will be depleted.

        Args:
            current_session_pct_used: Current session usage percentage (0-100)
            n_samples: Number of samples for uncertainty

        Returns:
            Dictionary with predictions
        """
        # Convert to remaining (normalized 0-1)
        current_remaining = 1.0 - current_session_pct_used / 100.0

        # Estimate how far into the session we are based on current usage
        # Assume roughly linear consumption to estimate position
        estimated_position = int(current_session_pct_used / 100.0 * self.curve_length)
        estimated_position = max(3, min(estimated_position, self.curve_length - 10))

        # Create context: observed points from start to current position
        n_context = estimated_position
        x_context = self.times[:n_context]

        # Linear interpolation for observed values
        observed_values = torch.linspace(1.0, current_remaining, n_context).unsqueeze(-1)

        x_target = self.times

        # Predict
        mean, std, samples = self.model.predict(x_context, observed_values, x_target, n_samples)

        mean = mean.numpy().squeeze()
        std = std.numpy().squeeze()
        samples_np = samples.numpy().squeeze()

        # Find depletion times
        depletion_threshold = 0.05  # 5% remaining = effectively empty
        depletion_indices = []

        for sample in samples_np:
            below = np.where(sample < depletion_threshold)[0]
            if len(below) > 0:
                depletion_indices.append(below[0])
            else:
                depletion_indices.append(self.curve_length)

        depletion_indices = np.array(depletion_indices)

        # Convert to hours remaining
        current_position_hours = estimated_position / self.curve_length * self.session_hours
        depletion_hours = depletion_indices / self.curve_length * self.session_hours
        time_remaining = depletion_hours - current_position_hours

        return {
            'time_remaining_hours_mean': float(np.mean(time_remaining)),
            'time_remaining_hours_std': float(np.std(time_remaining)),
            'time_remaining_hours_min': float(np.min(time_remaining)),
            'time_remaining_hours_max': float(np.max(time_remaining)),
            'confidence': float(1.0 - np.std(time_remaining) / self.session_hours),
            'current_position_pct': current_session_pct_used,
            'predicted_curve_mean': mean.tolist(),
            'predicted_curve_std': std.tolist(),
        }


def main():
    """CLI interface for the predictor."""
    import argparse

    parser = argparse.ArgumentParser(description='Claude Usage Predictor')
    parser.add_argument('--train', action='store_true', help='Train/update the model')
    parser.add_argument('--predict', type=float, help='Predict depletion given session %% used')
    parser.add_argument('--record', nargs=2, type=float, metavar=('SESSION', 'WEEKLY'),
                        help='Record observation (session_pct, weekly_pct)')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()

    predictor = UsagePredictor()

    if args.train:
        predictor.train(epochs=200, verbose=args.verbose)
        print("Model trained and saved.")

    if args.record:
        predictor.record_observation(args.record[0], args.record[1])
        if args.verbose:
            print(f"Recorded: session={args.record[0]}%, weekly={args.record[1]}%")

    if args.predict is not None:
        result = predictor.predict_depletion(args.predict)
        print(f"TIME_REMAINING_HOURS={result['time_remaining_hours_mean']:.2f}")
        print(f"TIME_REMAINING_STD={result['time_remaining_hours_std']:.2f}")
        print(f"CONFIDENCE={result['confidence']:.2f}")

        hours = result['time_remaining_hours_mean']
        if hours >= 1:
            print(f"Predicted: {hours:.1f}h ± {result['time_remaining_hours_std']:.1f}h remaining")
        else:
            mins = hours * 60
            print(f"Predicted: {mins:.0f}min ± {result['time_remaining_hours_std']*60:.0f}min remaining")


if __name__ == "__main__":
    main()
