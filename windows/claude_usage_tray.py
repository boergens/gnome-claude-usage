#!/usr/bin/env python3
"""
Claude Usage System Tray - Windows taskbar utility
Shows Claude Code usage (session and weekly remaining) with AI-powered predictions.
"""

import subprocess
import os
import sys
import re
import threading
import time
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageDraw, ImageFont
    import pystray
except ImportError:
    print("Please install required packages: pip install pystray pillow")
    sys.exit(1)

# Add parent directory to path for neural_process import
SCRIPT_DIR = Path(__file__).parent.parent / "claude-usage@local"
sys.path.insert(0, str(SCRIPT_DIR))

REFRESH_INTERVAL_SECONDS = 300  # 5 minutes


def create_icon_image(text="🤖", bg_color="#4A90D9"):
    """Create a simple icon with text."""
    size = 64
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw background circle
    draw.ellipse([4, 4, size-4, size-4], fill=bg_color)

    # Try to draw text (fallback to simple shapes if font fails)
    try:
        # Try to use a system font
        font = ImageFont.truetype("arial.ttf", 24)
        bbox = draw.textbbox((0, 0), text[:2], font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (size - text_width) // 2
        y = (size - text_height) // 2 - 4
        draw.text((x, y), text[:2], fill="white", font=font)
    except:
        # Fallback: draw a simple robot face
        draw.rectangle([18, 20, 28, 30], fill='white')  # left eye
        draw.rectangle([36, 20, 46, 30], fill='white')  # right eye
        draw.rectangle([22, 38, 42, 44], fill='white')  # mouth

    return img


class ClaudeUsageTray:
    def __init__(self):
        self.session_remaining = "??"
        self.weekly_remaining = "??"
        self.time_remaining_str = None
        self.confidence = None
        self.last_updated = "Never"
        self.running = True

        # Create initial icon
        self.icon = pystray.Icon(
            "claude_usage",
            create_icon_image(),
            "Claude Usage: Loading...",
            menu=self.create_menu()
        )

        # Start refresh thread
        self.refresh_thread = threading.Thread(target=self.refresh_loop, daemon=True)
        self.refresh_thread.start()

    def create_menu(self):
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: f"Session: {self.session_remaining}% remaining",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                lambda item: self.get_time_remaining_text(),
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Weekly: {self.weekly_remaining}% remaining",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Updated: {self.last_updated}",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh Now", self.refresh_clicked),
            pystray.MenuItem("Quit", self.quit_clicked),
        )

    def get_time_remaining_text(self):
        if self.time_remaining_str:
            text = f"Predicted: ~{self.time_remaining_str}"
            if self.confidence:
                try:
                    conf = round(float(self.confidence) * 100)
                    text += f" ({conf}% conf)"
                except:
                    pass
            return text
        return "Predicted: --"

    def refresh_clicked(self, icon, item):
        threading.Thread(target=self.fetch_usage, daemon=True).start()

    def quit_clicked(self, icon, item):
        self.running = False
        icon.stop()

    def refresh_loop(self):
        while self.running:
            self.fetch_usage()
            for _ in range(REFRESH_INTERVAL_SECONDS):
                if not self.running:
                    break
                time.sleep(1)

    def fetch_usage(self):
        """Fetch Claude usage via subprocess."""
        self.update_tooltip("Claude Usage: Refreshing...")

        try:
            # Try using the fetch script if on WSL or Git Bash with tmux
            fetch_script = SCRIPT_DIR / "fetch_usage.sh"

            if fetch_script.exists() and self.has_tmux():
                result = subprocess.run(
                    ["bash", str(fetch_script)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(SCRIPT_DIR)
                )
                output = result.stdout
                self.parse_fetch_output(output)
            else:
                # Fallback: try direct claude command with expect-like behavior
                self.fetch_usage_direct()

        except Exception as e:
            print(f"Error fetching usage: {e}")
            self.update_tooltip("Claude Usage: Error")

    def has_tmux(self):
        """Check if tmux is available."""
        try:
            result = subprocess.run(["which", "tmux"], capture_output=True)
            return result.returncode == 0
        except:
            try:
                result = subprocess.run(["where", "tmux"], capture_output=True, shell=True)
                return result.returncode == 0
            except:
                return False

    def fetch_usage_direct(self):
        """Try to fetch usage without tmux (Windows native)."""
        try:
            # Use PowerShell to run claude and capture output
            # This is a simplified approach - may need adjustment
            ps_script = '''
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = "claude"
            $psi.Arguments = "--dangerously-skip-permissions"
            $psi.UseShellExecute = $false
            $psi.RedirectStandardInput = $true
            $psi.RedirectStandardOutput = $true
            $psi.CreateNoWindow = $true

            $process = [System.Diagnostics.Process]::Start($psi)
            Start-Sleep -Seconds 2
            $process.StandardInput.WriteLine("")
            Start-Sleep -Seconds 2
            $process.StandardInput.WriteLine("")
            Start-Sleep -Seconds 2
            $process.StandardInput.WriteLine("/usage")
            Start-Sleep -Seconds 3
            $process.StandardInput.WriteLine("/exit")
            Start-Sleep -Seconds 1

            $output = $process.StandardOutput.ReadToEnd()
            $process.WaitForExit(5000)
            Write-Output $output
            '''

            result = subprocess.run(
                ["powershell", "-Command", ps_script],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.stdout:
                self.parse_raw_output(result.stdout)
            else:
                # If PowerShell approach fails, show error
                self.update_tooltip("Claude Usage: Run fetch_usage.sh manually")

        except Exception as e:
            print(f"Direct fetch failed: {e}")
            self.update_tooltip("Claude Usage: Error (install tmux via WSL)")

    def parse_fetch_output(self, output):
        """Parse output from fetch_usage.sh."""
        try:
            data = {}
            for line in output.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

            self.session_remaining = data.get('SESSION_REMAINING', '??')
            self.weekly_remaining = data.get('WEEKLY_REMAINING', '??')
            self.time_remaining_str = data.get('TIME_REMAINING_STR')
            self.confidence = data.get('CONFIDENCE')
            self.last_updated = datetime.now().strftime('%H:%M:%S')

            # Update tooltip
            if self.time_remaining_str:
                self.update_tooltip(f"Claude: {self.time_remaining_str} ({self.session_remaining}%)")
            else:
                self.update_tooltip(f"Claude: S:{self.session_remaining}% W:{self.weekly_remaining}%")

        except Exception as e:
            print(f"Parse error: {e}")
            self.update_tooltip("Claude Usage: Parse error")

    def parse_raw_output(self, output):
        """Parse raw Claude /usage output."""
        try:
            # Find session usage
            session_match = re.search(r"Current session.*?(\d+)%\s*used", output, re.DOTALL)
            session_pct = session_match.group(1) if session_match else None

            # Find weekly usage
            weekly_match = re.search(r"Current week \(all models\).*?(\d+)%\s*used", output, re.DOTALL)
            weekly_pct = weekly_match.group(1) if weekly_match else None

            if session_pct:
                session_used = int(session_pct)
                self.session_remaining = str(100 - session_used)

                # Try predictions
                try:
                    from neural_process import UsagePredictor
                    predictor = UsagePredictor()

                    if weekly_pct:
                        predictor.record_observation(session_used, int(weekly_pct))

                    result = predictor.predict_depletion(session_used)
                    time_remaining = result["time_remaining_hours_mean"]
                    self.confidence = str(result["confidence"])

                    if time_remaining <= 0:
                        self.time_remaining_str = "<5m"
                    elif time_remaining >= 1:
                        self.time_remaining_str = f"{time_remaining:.1f}h"
                    else:
                        self.time_remaining_str = f"{int(time_remaining * 60)}m"
                except Exception as e:
                    print(f"Prediction error: {e}")

            if weekly_pct:
                self.weekly_remaining = str(100 - int(weekly_pct))

            self.last_updated = datetime.now().strftime('%H:%M:%S')

            if self.time_remaining_str:
                self.update_tooltip(f"Claude: {self.time_remaining_str} ({self.session_remaining}%)")
            else:
                self.update_tooltip(f"Claude: S:{self.session_remaining}% W:{self.weekly_remaining}%")

        except Exception as e:
            print(f"Parse error: {e}")

    def update_tooltip(self, text):
        """Update the system tray tooltip."""
        self.icon.title = text

    def run(self):
        """Run the system tray app."""
        self.icon.run()


def main():
    print("Starting Claude Usage System Tray...")
    print("Look for the icon in your system tray (bottom right)")
    app = ClaudeUsageTray()
    app.run()


if __name__ == "__main__":
    main()
