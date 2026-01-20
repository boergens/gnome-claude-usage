/* extension_legacy.js - For GNOME 42-44 (non-ESM)
 *
 * To use this version:
 * 1. Rename extension.js to extension_modern.js
 * 2. Rename this file to extension.js
 * 3. Update metadata.json to only include ["42", "43", "44"]
 */

const { GLib, Gio, St, Clutter } = imports.gi;
const Main = imports.ui.main;
const PanelMenu = imports.ui.panelMenu;
const PopupMenu = imports.ui.popupMenu;
const ExtensionUtils = imports.misc.extensionUtils;

const REFRESH_INTERVAL_SECONDS = 300;

let indicator = null;
let timeout = null;
let panelLabel = null;
let accountMenuItem = null;
let sessionMenuItem = null;
let timeRemainingMenuItem = null;
let weeklyMenuItem = null;
let lastUpdatedMenuItem = null;
let sessionResetsMenuItem = null;
let weeklyResetsMenuItem = null;

function init() {
    // Nothing to initialize
}

function enable() {
    indicator = new PanelMenu.Button(0.0, 'Claude Usage', false);

    panelLabel = new St.Label({
        text: '🤖 --',
        y_align: Clutter.ActorAlign.CENTER
    });
    indicator.add_child(panelLabel);

    accountMenuItem = new PopupMenu.PopupMenuItem('Account: --');
    accountMenuItem.sensitive = false;
    indicator.menu.addMenuItem(accountMenuItem);

    indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

    sessionMenuItem = new PopupMenu.PopupMenuItem('Session: Loading...');
    sessionMenuItem.sensitive = false;
    indicator.menu.addMenuItem(sessionMenuItem);

    timeRemainingMenuItem = new PopupMenu.PopupMenuItem('Time remaining: --');
    timeRemainingMenuItem.sensitive = false;
    indicator.menu.addMenuItem(timeRemainingMenuItem);

    sessionResetsMenuItem = new PopupMenu.PopupMenuItem('Resets: --');
    sessionResetsMenuItem.sensitive = false;
    indicator.menu.addMenuItem(sessionResetsMenuItem);

    indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

    weeklyMenuItem = new PopupMenu.PopupMenuItem('Weekly: Loading...');
    weeklyMenuItem.sensitive = false;
    indicator.menu.addMenuItem(weeklyMenuItem);

    weeklyResetsMenuItem = new PopupMenu.PopupMenuItem('Resets: --');
    weeklyResetsMenuItem.sensitive = false;
    indicator.menu.addMenuItem(weeklyResetsMenuItem);

    indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

    lastUpdatedMenuItem = new PopupMenu.PopupMenuItem('Last updated: Never');
    lastUpdatedMenuItem.sensitive = false;
    indicator.menu.addMenuItem(lastUpdatedMenuItem);

    indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

    const refreshItem = new PopupMenu.PopupMenuItem('Refresh Now');
    refreshItem.connect('activate', () => {
        fetchUsage();
    });
    indicator.menu.addMenuItem(refreshItem);

    Main.panel.addToStatusArea('claude-usage', indicator);

    fetchUsage();

    timeout = GLib.timeout_add_seconds(
        GLib.PRIORITY_DEFAULT,
        REFRESH_INTERVAL_SECONDS,
        () => {
            fetchUsage();
            return GLib.SOURCE_CONTINUE;
        }
    );
}

function disable() {
    if (timeout) {
        GLib.Source.remove(timeout);
        timeout = null;
    }

    if (indicator) {
        indicator.destroy();
        indicator = null;
    }
}

function fetchUsage() {
    panelLabel.set_text('🤖 ⟳');

    const extensionDir = ExtensionUtils.getCurrentExtension().path;
    const scriptPath = GLib.build_filenamev([extensionDir, 'fetch_usage.sh']);

    try {
        const proc = Gio.Subprocess.new(
            ['bash', scriptPath],
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_PIPE
        );

        proc.communicate_utf8_async(null, null, (proc, res) => {
            try {
                const [, stdout, stderr] = proc.communicate_utf8_finish(res);
                parseUsage(stdout);
            } catch (e) {
                log(`Claude Usage Extension: Error reading output: ${e.message}`);
                setError('Error');
            }
        });
    } catch (e) {
        log(`Claude Usage Extension: Error spawning process: ${e.message}`);
        setError('Error');
    }
}

function parseUsage(output) {
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

        // Get account info
        const accountEmail = data['ACCOUNT_EMAIL'];
        const planType = data['PLAN_TYPE'];

        // Get remaining percentages (already calculated by script)
        const sessionRemaining = data['SESSION_REMAINING'] || '??';
        const weeklyRemaining = data['WEEKLY_REMAINING'] || '??';
        const extraUsed = data['EXTRA_USED'];
        const timeRemainingStr = data['TIME_REMAINING_STR'];
        const confidence = data['CONFIDENCE'];
        const sessionResets = data['SESSION_RESETS'];
        const weeklyResets = data['WEEKLY_RESETS'];
        const exhaustsBeforeReset = data['EXHAUSTS_BEFORE_RESET'] === 'true';

        // Update account info
        if (accountEmail) {
            let accountText = `Account: ${accountEmail}`;
            if (planType) {
                accountText += ` (${planType})`;
            }
            accountMenuItem.label.set_text(accountText);
        } else {
            accountMenuItem.label.set_text('Account: --');
        }

        // Show time remaining in panel if available, otherwise show percentages
        // Add warning indicator if will exhaust before reset
        const warning = exhaustsBeforeReset ? '⚠️ ' : '';
        if (timeRemainingStr) {
            panelLabel.set_text(`${warning}🤖 ${timeRemainingStr} (${sessionRemaining}%)`);
        } else {
            panelLabel.set_text(`🤖 W:${weeklyRemaining}% S:${sessionRemaining}%`);
        }

        sessionMenuItem.label.set_text(`Session remaining: ${sessionRemaining}%`);

        // Show predicted time remaining
        if (timeRemainingStr) {
            let timeText = `Predicted time left: ~${timeRemainingStr}`;
            if (confidence) {
                const conf = Math.round(parseFloat(confidence) * 100);
                timeText += ` (${conf}% conf)`;
            }
            if (exhaustsBeforeReset) {
                timeText += ' ⚠️ before reset!';
            }
            timeRemainingMenuItem.label.set_text(timeText);
        } else {
            timeRemainingMenuItem.label.set_text('Predicted time left: --');
        }

        // Session reset time
        if (sessionResets) {
            sessionResetsMenuItem.label.set_text(`Resets at ${sessionResets}`);
        } else {
            sessionResetsMenuItem.label.set_text('Resets: --');
        }

        weeklyMenuItem.label.set_text(`Weekly remaining: ${weeklyRemaining}%`);

        if (extraUsed) {
            weeklyMenuItem.label.set_text(`Weekly remaining: ${weeklyRemaining}% (Extra: ${extraUsed}% used)`);
        }

        // Weekly reset time
        if (weeklyResets) {
            weeklyResetsMenuItem.label.set_text(`Resets ${weeklyResets}`);
        } else {
            weeklyResetsMenuItem.label.set_text('Resets: --');
        }

        const now = new Date();
        lastUpdatedMenuItem.label.set_text(`Last updated: ${now.toLocaleTimeString()}`);

    } catch (e) {
        log(`Claude Usage Extension: Error parsing usage: ${e.message}`);
        setError('Parse error');
    }
}

function setError(msg) {
    panelLabel.set_text(`🤖 ${msg}`);
    sessionMenuItem.label.set_text(`Session: ${msg}`);
    weeklyMenuItem.label.set_text(`Weekly: ${msg}`);
}
