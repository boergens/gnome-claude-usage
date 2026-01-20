#!/usr/bin/env python3
"""
Claude Usage Menu Bar - macOS status bar utility
Shows Claude Code usage (session and weekly remaining) with AI-powered predictions.
"""

import subprocess
import os
import sys
import re
import threading
import time
from pathlib import Path

# Hide dock icon before importing rumps
try:
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
except ImportError:
    pass  # AppKit not available, dock icon will show

import rumps

# Add parent directory to path for neural_process import
SCRIPT_DIR = Path(__file__).parent.parent / "claude-usage@local"
sys.path.insert(0, str(SCRIPT_DIR))

REFRESH_INTERVAL_SECONDS = 300  # 5 minutes


class ClaudeUsageApp(rumps.App):
    def __init__(self):
        super().__init__(
            "Claude Usage",
            icon=None,
            title="🤖 --",
            quit_button=None
        )

        # Menu items
        self.account_item = rumps.MenuItem("Account: --", callback=None)
        self.account_item.set_callback(None)

        self.session_item = rumps.MenuItem("Session: Loading...", callback=None)
        self.session_item.set_callback(None)

        self.time_remaining_item = rumps.MenuItem("Depletes: --", callback=None)
        self.time_remaining_item.set_callback(None)

        self.session_resets_item = rumps.MenuItem("Resets: --", callback=None)
        self.session_resets_item.set_callback(None)

        self.weekly_item = rumps.MenuItem("Weekly: Loading...", callback=None)
        self.weekly_item.set_callback(None)

        self.weekly_resets_item = rumps.MenuItem("Resets: --", callback=None)
        self.weekly_resets_item.set_callback(None)

        self.last_updated_item = rumps.MenuItem("Last updated: Never", callback=None)
        self.last_updated_item.set_callback(None)

        self.menu = [
            self.account_item,
            None,  # Separator
            self.session_item,
            self.time_remaining_item,
            self.session_resets_item,
            None,  # Separator
            self.weekly_item,
            self.weekly_resets_item,
            None,  # Separator
            self.last_updated_item,
            None,  # Separator
            rumps.MenuItem("Refresh Now", callback=self.refresh_clicked),
            None,  # Separator
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

        # Start refresh timer
        self.timer = rumps.Timer(self.refresh_timer, REFRESH_INTERVAL_SECONDS)
        self.timer.start()

        # Initial fetch (in background)
        threading.Thread(target=self.fetch_usage, daemon=True).start()

    def refresh_clicked(self, _):
        """Manual refresh button clicked."""
        threading.Thread(target=self.fetch_usage, daemon=True).start()

    def refresh_timer(self, _):
        """Periodic refresh timer."""
        threading.Thread(target=self.fetch_usage, daemon=True).start()

    def fetch_usage(self):
        """Fetch Claude usage via tmux automation."""
        # Show loading state
        self.title = "🤖 ⟳"

        session_name = f"claude_usage_fetch_{os.getpid()}"

        try:
            # Kill any existing session
            subprocess.run(
                ["tmux", "kill-session", "-t", session_name],
                capture_output=True,
                timeout=5
            )
        except Exception:
            pass

        try:
            # Start claude in tmux
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session_name, "-x", "120", "-y", "50",
                 "claude", "--dangerously-skip-permissions"],
                check=True,
                timeout=10
            )

            # Wait for startup
            time.sleep(2)

            # Press enter for trust dialog
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Enter"], timeout=5)
            time.sleep(2)

            # Press enter again
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Enter"], timeout=5)
            time.sleep(2)

            # Type /usage character by character
            for char in "/usage":
                subprocess.run(["tmux", "send-keys", "-t", session_name, "-l", char], timeout=5)
                time.sleep(0.2)

            # Press Enter
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Enter"], timeout=5)
            time.sleep(2)

            # Capture usage output
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True,
                text=True,
                timeout=10
            )
            output = result.stdout

            # Navigate left twice to get account info
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Left"], timeout=5)
            time.sleep(0.5)
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Left"], timeout=5)
            time.sleep(1)

            # Capture account info
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"],
                capture_output=True,
                text=True,
                timeout=10
            )
            account_output = result.stdout

            # Clean up tmux session
            subprocess.run(["tmux", "send-keys", "-t", session_name, "Escape"], timeout=5)
            time.sleep(0.5)
            subprocess.run(["tmux", "send-keys", "-t", session_name, "/exit", "Enter"], timeout=5)
            time.sleep(1)
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True, timeout=5)

            # Parse the output
            self.parse_usage(output, account_output)

        except Exception as e:
            print(f"Error fetching usage: {e}")
            self.set_error("Error")
            # Try to clean up
            try:
                subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True, timeout=5)
            except Exception:
                pass

    def parse_usage(self, output, account_output=""):
        """Parse Claude usage output and update UI."""
        try:
            # Find account email from account output
            account_match = re.search(r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", account_output)
            account_email = account_match.group(1) if account_match else None

            # Find plan type
            plan_match = re.search(r"(Max|Pro|Team|Enterprise|Free)", account_output, re.IGNORECASE)
            plan_type = plan_match.group(1) if plan_match else None

            # Find session usage
            session_match = re.search(r"Current session.*?(\d+)%\s*used", output, re.DOTALL)
            session_pct = session_match.group(1) if session_match else None

            # Find weekly usage (all models)
            weekly_match = re.search(r"Current week \(all models\).*?(\d+)%\s*used", output, re.DOTALL)
            weekly_pct = weekly_match.group(1) if weekly_match else None

            # Find extra usage
            extra_match = re.search(r"Extra usage.*?(\d+)%\s*used", output, re.DOTALL)
            extra_pct = extra_match.group(1) if extra_match else None

            # Find reset times
            session_reset_match = re.search(r"(?:session|limit).*?(?:resets?|refreshes?).*?(?:in\s+)?(\d+[hm](?:\s*\d+[hm])?|\d{1,2}:\d{2}(?:\s*[AP]M)?)", output, re.IGNORECASE)
            session_resets = session_reset_match.group(1) if session_reset_match else None

            weekly_reset_match = re.search(r"(?:week|weekly).*?(?:resets?|refreshes?).*?(?:in\s+)?(\d+[dhm](?:\s*\d+[hm])?|\w+day|\d{1,2}:\d{2})", output, re.IGNORECASE)
            weekly_resets = weekly_reset_match.group(1) if weekly_reset_match else None

            # Calculate remaining
            session_used = int(session_pct) if session_pct else None
            session_remaining = 100 - session_used if session_used is not None else "??"

            weekly_used = int(weekly_pct) if weekly_pct else None
            weekly_remaining = 100 - weekly_used if weekly_used is not None else "??"

            # Try to get predictions from neural process
            time_remaining_str = None
            confidence = None

            if session_used is not None:
                try:
                    from neural_process import UsagePredictor
                    predictor = UsagePredictor()

                    # Record observation for future training
                    if weekly_used is not None:
                        predictor.record_observation(
                            session_used, weekly_used,
                            session_resets=session_resets,
                            weekly_resets=weekly_resets
                        )

                    # Get prediction
                    result = predictor.predict_depletion(session_used)
                    time_remaining = result["time_remaining_hours_mean"]
                    confidence = result["confidence"]

                    # Format time remaining
                    if time_remaining <= 0:
                        time_remaining_str = "<5m"
                    elif time_remaining >= 1:
                        time_remaining_str = f"{time_remaining:.1f}h"
                    else:
                        time_remaining_str = f"{int(time_remaining * 60)}m"

                except Exception as e:
                    print(f"Prediction error: {e}")

            # Update UI (must be done on main thread via rumps)
            if time_remaining_str:
                self.title = f"🤖 {time_remaining_str} ({session_remaining}%)"
            else:
                self.title = f"🤖 W:{weekly_remaining}% S:{session_remaining}%"

            # Update account info
            if account_email:
                account_text = f"Account: {account_email}"
                if plan_type:
                    account_text += f" ({plan_type})"
                self.account_item.title = account_text
            else:
                self.account_item.title = "Account: --"

            self.session_item.title = f"Session remaining: {session_remaining}%"

            if time_remaining_str:
                time_text = f"Depletes in ~{time_remaining_str}"
                if confidence:
                    conf = round(confidence * 100)
                    time_text += f" ({conf}% conf)"
                self.time_remaining_item.title = time_text
            else:
                self.time_remaining_item.title = "Depletes: --"

            # Session reset time (from Claude)
            if session_resets:
                self.session_resets_item.title = f"Resets in {session_resets}"
            else:
                self.session_resets_item.title = "Resets: --"

            weekly_text = f"Weekly remaining: {weekly_remaining}%"
            if extra_pct:
                weekly_text += f" (Extra: {extra_pct}% used)"
            self.weekly_item.title = weekly_text

            # Weekly reset time (from Claude)
            if weekly_resets:
                self.weekly_resets_item.title = f"Resets in {weekly_resets}"
            else:
                self.weekly_resets_item.title = "Resets: --"

            from datetime import datetime
            self.last_updated_item.title = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

        except Exception as e:
            print(f"Error parsing usage: {e}")
            self.set_error("Parse error")

    def set_error(self, msg):
        """Set error state in UI."""
        self.title = f"🤖 {msg}"
        self.session_item.title = f"Session: {msg}"
        self.weekly_item.title = f"Weekly: {msg}"


if __name__ == "__main__":
    ClaudeUsageApp().run()
