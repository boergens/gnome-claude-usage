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

        self.debug_item = rumps.MenuItem("Debug: --", callback=None)
        self.debug_item.set_callback(None)

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
            self.debug_item,
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
        """Fetch Claude usage by calling fetch_usage.sh script."""
        # Show loading state
        self.title = "🤖 ⟳"
        self.debug_item.title = "Debug: Fetching..."

        try:
            # Call the fetch_usage.sh script directly
            script_path = SCRIPT_DIR / "fetch_usage.sh"

            result = subprocess.run(
                ["bash", str(script_path)],
                capture_output=True,
                text=True,
                timeout=90,
                cwd=str(SCRIPT_DIR)
            )

            if result.returncode != 0:
                self.debug_item.title = f"Debug: Script error - {result.stderr[:50]}"
                self.set_error("Script error")
                return

            # Parse the key=value output from the script
            self.parse_script_output(result.stdout)

        except subprocess.TimeoutExpired:
            self.debug_item.title = "Debug: Timeout (90s)"
            self.set_error("Timeout")
        except Exception as e:
            self.debug_item.title = f"Debug: {str(e)[:50]}"
            self.set_error("Error")

    def parse_script_output(self, output):
        """Parse key=value output from fetch_usage.sh and update UI."""
        try:
            # Parse key=value lines
            data = {}
            for line in output.strip().split('\n'):
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip()

            # Extract values
            account_email = data.get('ACCOUNT_EMAIL')
            plan_type = data.get('PLAN_TYPE')
            session_remaining = data.get('SESSION_REMAINING', '??')
            weekly_remaining = data.get('WEEKLY_REMAINING', '??')
            extra_pct = data.get('EXTRA_USED')
            time_remaining_str = data.get('TIME_REMAINING_STR')
            confidence = data.get('CONFIDENCE')
            session_resets = data.get('SESSION_RESETS')
            weekly_resets = data.get('WEEKLY_RESETS')
            exhausts_before_reset = data.get('EXHAUSTS_BEFORE_RESET') == 'true'

            # Update debug - show exhaustion status
            exhaust_str = "WARN" if exhausts_before_reset else "ok"
            self.debug_item.title = f"Debug: OK ({len(data)} vals, {exhaust_str})"

            # Update UI - show warning if will exhaust before reset
            warning = "⚠️ " if exhausts_before_reset else ""
            if time_remaining_str and time_remaining_str != '??':
                self.title = f"{warning}🤖 {time_remaining_str} ({session_remaining}%)"
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

            if time_remaining_str and time_remaining_str != '??':
                time_text = f"Depletes in ~{time_remaining_str}"
                if confidence:
                    try:
                        conf = round(float(confidence) * 100)
                        time_text += f" ({conf}% conf)"
                    except:
                        pass
                if exhausts_before_reset:
                    time_text += " ⚠️ before reset!"
                self.time_remaining_item.title = time_text
            else:
                self.time_remaining_item.title = "Depletes: --"

            # Session reset time (from Claude)
            if session_resets:
                reset_text = f"Resets at {session_resets}"
                self.session_resets_item.title = reset_text
            else:
                self.session_resets_item.title = "Resets: --"

            weekly_text = f"Weekly remaining: {weekly_remaining}%"
            if extra_pct:
                weekly_text += f" (Extra: {extra_pct}% used)"
            self.weekly_item.title = weekly_text

            # Weekly reset time (from Claude)
            if weekly_resets:
                self.weekly_resets_item.title = f"Resets {weekly_resets}"
            else:
                self.weekly_resets_item.title = "Resets: --"

            from datetime import datetime
            self.last_updated_item.title = f"Last updated: {datetime.now().strftime('%H:%M:%S')}"

        except Exception as e:
            self.debug_item.title = f"Debug: Parse error - {str(e)[:30]}"
            self.set_error("Parse error")

    def set_error(self, msg):
        """Set error state in UI."""
        self.title = f"🤖 {msg}"
        self.session_item.title = f"Session: {msg}"
        self.weekly_item.title = f"Weekly: {msg}"


if __name__ == "__main__":
    ClaudeUsageApp().run()
