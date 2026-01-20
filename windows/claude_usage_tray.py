#!/usr/bin/env python3
"""
Claude Usage System Tray - Windows taskbar utility
Shows Claude Code usage (session and weekly remaining) with AI-powered predictions.
"""

import subprocess
import os
import sys
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
        self.account_email = None
        self.plan_type = None
        self.session_remaining = "??"
        self.weekly_remaining = "??"
        self.time_remaining_str = None
        self.confidence = None
        self.session_resets = None
        self.weekly_resets = None
        self.exhausts_before_reset = False
        self.last_updated = "Never"
        self.debug_status = "Starting..."
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

    def get_account_text(self):
        if self.account_email:
            text = f"Account: {self.account_email}"
            if self.plan_type:
                text += f" ({self.plan_type})"
            return text
        return "Account: --"

    def get_time_remaining_text(self):
        if self.time_remaining_str and self.time_remaining_str != "??":
            text = f"Depletes in ~{self.time_remaining_str}"
            if self.confidence:
                try:
                    conf = round(float(self.confidence) * 100)
                    text += f" ({conf}% conf)"
                except:
                    pass
            if self.exhausts_before_reset:
                text += " - before reset!"
            return text
        return "Depletes: --"

    def get_session_resets_text(self):
        if self.session_resets:
            return f"Resets at {self.session_resets}"
        return "Resets: --"

    def get_weekly_resets_text(self):
        if self.weekly_resets:
            return f"Resets {self.weekly_resets}"
        return "Resets: --"

    def create_menu(self):
        return pystray.Menu(
            pystray.MenuItem(
                lambda item: self.get_account_text(),
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
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
            pystray.MenuItem(
                lambda item: self.get_session_resets_text(),
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Weekly: {self.weekly_remaining}% remaining",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                lambda item: self.get_weekly_resets_text(),
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                lambda item: f"Updated: {self.last_updated}",
                None,
                enabled=False
            ),
            pystray.MenuItem(
                lambda item: f"Debug: {self.debug_status}",
                None,
                enabled=False
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Refresh Now", self.refresh_clicked),
            pystray.MenuItem("Quit", self.quit_clicked),
        )

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
        """Fetch Claude usage by calling fetch_usage.sh script."""
        self.update_tooltip("Claude Usage: Refreshing...")
        self.debug_status = "Fetching..."

        fetch_script = SCRIPT_DIR / "fetch_usage.sh"

        if not fetch_script.exists():
            self.debug_status = f"Script not found: {fetch_script}"
            self.update_tooltip("Claude Usage: Script not found")
            return

        try:
            # Try bash (Git Bash, WSL, or native)
            result = subprocess.run(
                ["bash", str(fetch_script)],
                capture_output=True,
                text=True,
                timeout=90,
                cwd=str(SCRIPT_DIR)
            )

            if result.returncode != 0:
                self.debug_status = f"Script error: {result.stderr[:40]}"
                self.update_tooltip("Claude Usage: Script error")
                return

            self.parse_script_output(result.stdout)

        except FileNotFoundError:
            self.debug_status = "bash not found - install Git Bash or WSL"
            self.update_tooltip("Claude Usage: Install bash")
        except subprocess.TimeoutExpired:
            self.debug_status = "Timeout (90s)"
            self.update_tooltip("Claude Usage: Timeout")
        except Exception as e:
            self.debug_status = f"Error: {str(e)[:40]}"
            self.update_tooltip("Claude Usage: Error")

    def parse_script_output(self, output):
        """Parse key=value output from fetch_usage.sh."""
        try:
            data = {}
            for line in output.strip().split('\n'):
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

            self.account_email = data.get('ACCOUNT_EMAIL')
            self.plan_type = data.get('PLAN_TYPE')
            self.session_remaining = data.get('SESSION_REMAINING', '??')
            self.weekly_remaining = data.get('WEEKLY_REMAINING', '??')
            self.time_remaining_str = data.get('TIME_REMAINING_STR')
            self.confidence = data.get('CONFIDENCE')
            self.session_resets = data.get('SESSION_RESETS')
            self.weekly_resets = data.get('WEEKLY_RESETS')
            self.exhausts_before_reset = data.get('EXHAUSTS_BEFORE_RESET') == 'true'
            self.last_updated = datetime.now().strftime('%H:%M:%S')
            self.debug_status = f"OK ({len(data)} values)"

            # Update tooltip - show warning if will exhaust before reset
            warning = "[!] " if self.exhausts_before_reset else ""
            if self.time_remaining_str and self.time_remaining_str != "??":
                self.update_tooltip(f"{warning}Claude: {self.time_remaining_str} ({self.session_remaining}%)")
            else:
                self.update_tooltip(f"Claude: S:{self.session_remaining}% W:{self.weekly_remaining}%")

        except Exception as e:
            self.debug_status = f"Parse error: {str(e)[:30]}"
            self.update_tooltip("Claude Usage: Parse error")

    def update_tooltip(self, text):
        """Update the system tray tooltip."""
        self.icon.title = text

    def run(self):
        """Run the system tray app."""
        self.icon.run()


def main():
    print("Starting Claude Usage System Tray...")
    print("Look for the icon in your system tray (bottom right)")
    print("Requires: bash (Git Bash or WSL) and tmux")
    app = ClaudeUsageTray()
    app.run()


if __name__ == "__main__":
    main()
