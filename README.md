# Claude Usage Monitor - GNOME Extension

A GNOME Shell extension that displays your Claude Code usage (session and weekly remaining) in the top panel, with **AI-powered predictions** of when your session will be depleted.

![Panel showing: 🤖 2.5h (64%)](https://img.shields.io/badge/🤖_2.5h_(64%25)-blue)

## Features

- Shows predicted time remaining until session depletion
- Session and weekly usage percentages
- **Neural Process model** learns your usage patterns
- Auto-refreshes every 5 minutes
- Manual refresh button
- Works with Claude Max/Pro subscriptions

## Requirements

- GNOME Shell 42-47
- `tmux` installed
- `python3` with PyTorch (for predictions)
- Claude Code CLI installed and authenticated

## GNOME Version Compatibility

| GNOME Version | Extension File | Notes |
|---------------|----------------|-------|
| 45, 46, 47 | `extension.js` | Uses ESM modules (modern) |
| 42, 43, 44 | `extension_legacy.js` | Uses legacy imports |

The `install.sh` script automatically detects your GNOME version and uses the appropriate file.

## Installation

### Using the install script (recommended)

```bash
git clone https://github.com/boergens/gnome-claude-usage.git
cd gnome-claude-usage
./install.sh
```

Then restart GNOME Shell:
- **X11:** Press `Alt+F2`, type `r`, press Enter
- **Wayland:** Log out and log back in

Enable the extension:
```bash
gnome-extensions enable claude-usage@local
```

### Installing PyTorch for predictions

For the Neural Process predictions to work:
```bash
pip install torch
```

Or if you have a venv, update the `PYTHON_BIN` path in `fetch_usage.sh`.

### Manual Installation

1. Copy the extension to your GNOME extensions directory:
```bash
mkdir -p ~/.local/share/gnome-shell/extensions/
cp -r claude-usage@local ~/.local/share/gnome-shell/extensions/
```

2. For GNOME 42-44, swap to the legacy extension:
```bash
cd ~/.local/share/gnome-shell/extensions/claude-usage@local
mv extension.js extension_modern.js
mv extension_legacy.js extension.js
```

3. Restart GNOME Shell and enable the extension.

## Usage

Once enabled, you'll see in your top panel:
```
🤖 2.5h (64%)
```
- **2.5h:** Predicted time until session depletes
- **64%:** Current session remaining percentage

Click to see:
- Session remaining percentage
- Predicted time left with confidence level
- Weekly remaining percentage (with extra usage if applicable)
- Last update timestamp
- Manual refresh button

## Neural Process Prediction

The extension uses a **Neural Process** model to predict when your usage will be depleted:

- **Learns your patterns:** Records usage observations over time
- **Curve completion:** Given current usage %, predicts the full depletion curve
- **Uncertainty quantification:** Shows confidence in predictions
- **Few-shot learning:** Works with just a few historical sessions

### Training the model

The model trains automatically on synthetic data initially. As you use Claude, it records observations and can be retrained:

```bash
python3 claude-usage@local/neural_process.py --train --verbose
```

### Manual predictions

Test predictions at different usage levels:
```bash
python3 claude-usage@local/neural_process.py --predict 50
# Output: Predicted: 2.5h ± 0.2h remaining
```

## How it Works

1. **Fetch usage:** Starts a `tmux` session, runs `/usage` command, captures output
2. **Parse data:** Extracts session/weekly percentages from terminal output
3. **Record observation:** Stores data point for model training
4. **Predict depletion:** Neural Process predicts time remaining
5. **Display:** Shows time + percentage in GNOME panel

The fetch script takes ~10 seconds to run, so the extension refreshes every 5 minutes by default.

## Configuration

To change the refresh interval, edit `extension.js`:
```javascript
const REFRESH_INTERVAL_SECONDS = 300; // Change to desired seconds
```

## Troubleshooting

### "Error" shown in panel

1. Make sure `claude` CLI is in your PATH
2. Make sure `tmux` is installed: `sudo apt install tmux` or `brew install tmux`
3. Check that you're authenticated with Claude Code
4. Test the fetch script manually:
   ```bash
   ~/.local/share/gnome-shell/extensions/claude-usage@local/fetch_usage.sh
   ```
5. Check GNOME Shell logs: `journalctl -f -o cat /usr/bin/gnome-shell`

### Predictions not working

1. Install PyTorch: `pip install torch`
2. Update the `PYTHON_BIN` path in `fetch_usage.sh` if using a venv
3. Train the model: `python3 neural_process.py --train`

### Extension not appearing

1. Verify the extension is enabled: `gnome-extensions list --enabled`
2. Check for errors: `gnome-extensions info claude-usage@local`

### Wrong GNOME version file

If you see import errors, you may have the wrong extension.js for your GNOME version:
- GNOME 45+: Use `extension.js` (ESM modules)
- GNOME 42-44: Use `extension_legacy.js` (rename to `extension.js`)

## Uninstalling

```bash
gnome-extensions disable claude-usage@local
rm -rf ~/.local/share/gnome-shell/extensions/claude-usage@local
```

## How the Neural Process Works

The Neural Process is a meta-learning model that learns to complete curves from partial observations:

1. **Encoder:** Compresses observed (time, value) points into a latent representation
2. **Latent space:** Captures the "fingerprint" of the consumption pattern
3. **Decoder:** Uses the fingerprint to predict values at any target time
4. **Uncertainty:** Samples multiple predictions to estimate confidence

This is ideal for usage prediction because:
- Works with few historical examples (3-5 sessions)
- Naturally handles "given M points, predict the rest"
- Provides uncertainty estimates
- Learns typical consumption patterns automatically

## License

MIT
