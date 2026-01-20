import GLib from 'gi://GLib';
import Gio from 'gi://Gio';
import St from 'gi://St';
import Clutter from 'gi://Clutter';

import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';

const REFRESH_INTERVAL_SECONDS = 300; // 5 minutes

export default class ClaudeUsageExtension extends Extension {
    _indicator = null;
    _timeout = null;
    _sessionUsage = 'Loading...';
    _weeklyUsage = 'Loading...';
    _timeRemaining = null;
    _lastUpdated = '';

    enable() {
        this._indicator = new PanelMenu.Button(0.0, 'Claude Usage', false);

        // Create the panel label
        this._panelLabel = new St.Label({
            text: '🤖 --',
            y_align: Clutter.ActorAlign.CENTER,
            style_class: 'panel-button-text'
        });
        this._indicator.add_child(this._panelLabel);

        // Create menu items
        this._sessionMenuItem = new PopupMenu.PopupMenuItem('Session: Loading...');
        this._sessionMenuItem.sensitive = false;
        this._indicator.menu.addMenuItem(this._sessionMenuItem);

        this._timeRemainingMenuItem = new PopupMenu.PopupMenuItem('Time remaining: --');
        this._timeRemainingMenuItem.sensitive = false;
        this._indicator.menu.addMenuItem(this._timeRemainingMenuItem);

        this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._weeklyMenuItem = new PopupMenu.PopupMenuItem('Weekly: Loading...');
        this._weeklyMenuItem.sensitive = false;
        this._indicator.menu.addMenuItem(this._weeklyMenuItem);

        this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        this._lastUpdatedMenuItem = new PopupMenu.PopupMenuItem('Last updated: Never');
        this._lastUpdatedMenuItem.sensitive = false;
        this._indicator.menu.addMenuItem(this._lastUpdatedMenuItem);

        this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // Refresh button
        const refreshItem = new PopupMenu.PopupMenuItem('Refresh Now');
        refreshItem.connect('activate', () => {
            this._fetchUsage();
        });
        this._indicator.menu.addMenuItem(refreshItem);

        Main.panel.addToStatusArea(this.uuid, this._indicator);

        // Initial fetch
        this._fetchUsage();

        // Set up periodic refresh
        this._timeout = GLib.timeout_add_seconds(
            GLib.PRIORITY_DEFAULT,
            REFRESH_INTERVAL_SECONDS,
            () => {
                this._fetchUsage();
                return GLib.SOURCE_CONTINUE;
            }
        );
    }

    disable() {
        if (this._timeout) {
            GLib.Source.remove(this._timeout);
            this._timeout = null;
        }

        if (this._indicator) {
            this._indicator.destroy();
            this._indicator = null;
        }
    }

    _fetchUsage() {
        // Update UI to show loading
        this._panelLabel.set_text('🤖 ⟳');

        // Get the extension's directory for the helper script
        const extensionDir = this.path;
        const scriptPath = GLib.build_filenamev([extensionDir, 'fetch_usage.sh']);

        try {
            const proc = Gio.Subprocess.new(
                ['bash', scriptPath],
                Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE
            );

            proc.communicate_utf8_async(null, null, (proc, res) => {
                try {
                    const [, stdout, stderr] = proc.communicate_utf8_finish(res);
                    this._parseUsage(stdout);
                } catch (e) {
                    log(`Claude Usage Extension: Error reading output: ${e.message}`);
                    this._setError('Error');
                }
            });
        } catch (e) {
            log(`Claude Usage Extension: Error spawning process: ${e.message}`);
            this._setError('Error');
        }
    }

    _parseUsage(output) {
        try {
            // Parse key=value format from fetch_usage.sh
            const lines = output.split('\n');
            const data = {};

            for (const line of lines) {
                const match = line.match(/^([A-Z_]+)=(.+)$/);
                if (match) {
                    data[match[1]] = match[2];
                }
            }

            // Get remaining percentages (already calculated by script)
            const sessionRemaining = data['SESSION_REMAINING'] || '??';
            const weeklyRemaining = data['WEEKLY_REMAINING'] || '??';
            const extraUsed = data['EXTRA_USED'];
            const timeRemainingStr = data['TIME_REMAINING_STR'];
            const confidence = data['CONFIDENCE'];

            // Update UI
            this._sessionUsage = `${sessionRemaining}%`;
            this._weeklyUsage = `${weeklyRemaining}%`;
            this._timeRemaining = timeRemainingStr;

            // Show time remaining in panel if available, otherwise show percentages
            if (timeRemainingStr) {
                this._panelLabel.set_text(`🤖 ${timeRemainingStr} (${sessionRemaining}%)`);
            } else {
                this._panelLabel.set_text(`🤖 W:${weeklyRemaining}% S:${sessionRemaining}%`);
            }

            this._sessionMenuItem.label.set_text(`Session remaining: ${sessionRemaining}%`);

            // Show predicted time remaining
            if (timeRemainingStr) {
                let timeText = `Predicted time left: ~${timeRemainingStr}`;
                if (confidence) {
                    const conf = Math.round(parseFloat(confidence) * 100);
                    timeText += ` (${conf}% conf)`;
                }
                this._timeRemainingMenuItem.label.set_text(timeText);
            } else {
                this._timeRemainingMenuItem.label.set_text('Predicted time left: --');
            }

            this._weeklyMenuItem.label.set_text(`Weekly remaining: ${weeklyRemaining}%`);

            if (extraUsed) {
                this._weeklyMenuItem.label.set_text(`Weekly remaining: ${weeklyRemaining}% (Extra: ${extraUsed}% used)`);
            }

            const now = new Date();
            this._lastUpdated = now.toLocaleTimeString();
            this._lastUpdatedMenuItem.label.set_text(`Last updated: ${this._lastUpdated}`);

        } catch (e) {
            log(`Claude Usage Extension: Error parsing usage: ${e.message}`);
            this._setError('Parse error');
        }
    }

    _setError(msg) {
        this._panelLabel.set_text(`🤖 ${msg}`);
        this._sessionMenuItem.label.set_text(`Session: ${msg}`);
        this._weeklyMenuItem.label.set_text(`Weekly: ${msg}`);
    }
}
