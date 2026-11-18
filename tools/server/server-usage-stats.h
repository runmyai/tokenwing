#pragma once

struct common_params;

// Sends one anonymous usage record a day (engine version, OS, CPU and GPU names, who started the engine) on a
// background thread. Call once, after the model has loaded. Does nothing when turned off (--no-stats,
// TOKENWING_NO_STATS, DO_NOT_TRACK) or when the build cannot make HTTPS requests.
void server_usage_stats_start(const common_params & params);

// Ends a send that is still running, so the process can exit at once. Safe to call when nothing was started.
void server_usage_stats_stop();
