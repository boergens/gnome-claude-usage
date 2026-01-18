# Claude Usage Monitor - GNOME Extension

A GNOME Shell extension that displays your Claude Code usage (session and weekly remaining) in the top panel.

![Panel showing: 🤖 W:89% S:64%](https://img.shields.io/badge/🤖_W:89%25_S:64%25-blue)

## Features

- Shows weekly and session usage remaining in the GNOME panel
- Dropdown with detailed usage info including extra usage
- Auto-refreshes every 5 minutes
- Manual refresh button
- Works with Claude Max/Pro subscriptions

## Requirements

- GNOME Shell 42-47
- `tmux` installed
- `python3` installed
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
🤖 W:89% S:64%
```
- **W:** Weekly usage remaining
- **S:** Session usage remaining

Click to see:
- Session remaining percentage
- Weekly remaining percentage (with extra usage if applicable)
- Last update timestamp
- Manual refresh button

## How it Works

Since Claude Code's `/usage` command only works in interactive mode, this extension uses a clever workaround:

1. Starts a `tmux` session with `claude --dangerously-skip-permissions`
2. Simulates typing `/usage` with proper timing
3. Captures the terminal output
4. Parses the usage percentages
5. Cleans up the tmux session

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

## License

MIT
