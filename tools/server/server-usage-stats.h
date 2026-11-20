#pragma once

struct common_params;

// Sends one anonymous usage record a day (engine version, OS, CPU and GPU names, who started the engine) and, for an
// engine started by hand, one update check a day to the website (engine version and system), on a
// background thread. Call once, after the model has loaded. The record is turned off by --no-stats or TOKENWING_NO_STATS,
// the check by TOKENWING_NO_UPDATE_CHECK, both by DO_NOT_TRACK=1; nothing is sent when the build cannot make HTTPS requests.
void server_usage_stats_start(const common_params & params);

// Ends a send that is still running, so the process can exit at once. Safe to call when nothing was started.
void server_usage_stats_stop();
