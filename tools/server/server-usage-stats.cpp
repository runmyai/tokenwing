#include "server-usage-stats.h"

#include "server-common.h"
#include "http.h"

#include "build-info.h"
#include "ggml-backend.h"

#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#    if defined(_MSC_VER)
#        pragma comment(lib, "advapi32.lib")
#    endif
#elif defined(__APPLE__)
#    include <sys/sysctl.h>
#    include <sys/utsname.h>
#else
#    include <sys/utsname.h>
#endif

namespace fs = std::filesystem;

// The project's public web API key: it names the database, it is not a secret.
static const char * STATS_URL = "https://firestore.googleapis.com/v1/projects/run-my-ai-5876c/databases/(default)/"
                                "documents:commit?key=AIzaSyDN01Y9bvnKceKI0wbmnsexVyOdC5M1JBU";
static const char * STATS_DOC = "projects/run-my-ai-5876c/databases/(default)/documents/engineStats/";
// The website's update check: it counts one check (engine version and system) and answers with the release list.
static const char * CHECK_URL = "https://runmyai.org/api/releases";

//
// OS functions: one small function per question, each OS with its own version
//

// The folder for the install id and the day files; empty when it cannot be found.
static fs::path os_state_dir() {
#if defined(_WIN32)
    const wchar_t * v = _wgetenv(L"LOCALAPPDATA");
    return (v && *v) ? fs::path(v) / L"TokenWing" : fs::path();
#elif defined(__APPLE__)
    const char * home = std::getenv("HOME");
    return (home && *home) ? fs::path(home) / "Library" / "Application Support" / "TokenWing" : fs::path();
#else
    // the XDG spec ignores a relative XDG_STATE_HOME
    const char * xdg = std::getenv("XDG_STATE_HOME");
    if (xdg && xdg[0] == '/') {
        return fs::path(xdg) / "tokenwing";
    }
    const char * home = std::getenv("HOME");
    return (home && *home) ? fs::path(home) / ".local" / "state" / "tokenwing" : fs::path();
#endif
}

#if !defined(_WIN32)
static std::string read_key_value(const char * path, const std::string & key, char sep) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto pos = line.find(sep);
        if (pos == std::string::npos) {
            continue;
        }
        std::string k = line.substr(0, pos);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) {
            k.pop_back();
        }
        if (k != key) {
            continue;
        }
        std::string v = line.substr(pos + 1);
        v.erase(0, v.find_first_not_of(" \t\""));
        while (!v.empty() && (v.back() == ' ' || v.back() == '\t' || v.back() == '"' || v.back() == '\r')) {
            v.pop_back();
        }
        return v;
    }
    return "";
}
#endif

static std::string os_name() {
#if defined(_WIN32)
    // GetVersionEx reports the version the program declares support for, RtlGetVersion the real one
    typedef LONG (WINAPI * rtl_get_version_t)(PRTL_OSVERSIONINFOW);
    HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    auto fn = ntdll ? reinterpret_cast<rtl_get_version_t>(reinterpret_cast<void *>(GetProcAddress(ntdll, "RtlGetVersion"))) : nullptr;
    RTL_OSVERSIONINFOW vi = {};
    vi.dwOSVersionInfoSize = sizeof(vi);
    if (fn && fn(&vi) == 0) {
        return "Windows " + std::to_string(vi.dwMajorVersion) + "." + std::to_string(vi.dwMinorVersion) +
               " (Build " + std::to_string(vi.dwBuildNumber) + ")";
    }
    return "Windows";
#elif defined(__APPLE__)
    char buf[64] = {};
    size_t len = sizeof(buf) - 1;
    if (sysctlbyname("kern.osproductversion", buf, &len, nullptr, 0) == 0 && buf[0]) {
        return std::string("macOS ") + buf;
    }
    return "macOS";
#else
    std::string name = read_key_value("/etc/os-release", "PRETTY_NAME", '=');
    if (name.empty()) {
        name = read_key_value("/usr/lib/os-release", "PRETTY_NAME", '=');
    }
    struct utsname u;
    if (uname(&u) == 0) {
        if (name.empty()) {
            name = u.sysname;
        }
        return name + ", kernel " + u.release;
    }
    return name.empty() ? "unknown" : name;
#endif
}

static std::string os_cpu_name() {
#if defined(_WIN32)
    char buf[256] = {};
    DWORD size = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS) {
        // the processor fills the name to a fixed width with spaces (AMD's at the end, some Intel ones in front)
        const std::string name = string_strip(buf);
        if (!name.empty()) {
            return name;
        }
    }
#    if defined(_M_ARM64)
    return "arm64";
#    elif defined(_M_X64)
    return "x86_64";
#    else
    return "unknown";
#    endif
#elif defined(__APPLE__)
    char buf[256] = {};
    size_t len = sizeof(buf) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0 && buf[0]) {
        return buf;
    }
    struct utsname u;
    return uname(&u) == 0 ? std::string(u.machine) : std::string("unknown");
#else
    // x86 and some ARM kernels give "model name"; other ARM kernels give "Hardware" or only "CPU part"
    for (const char * key : { "model name", "Hardware", "CPU part" }) {
        std::string v = read_key_value("/proc/cpuinfo", key, ':');
        if (!v.empty()) {
            return v;
        }
    }
    struct utsname u;
    return uname(&u) == 0 ? std::string(u.machine) : std::string("unknown");
#endif
}

// The system and processor kind in the app's words, e.g. "linux-x64".
static std::string os_kind() {
#if defined(_WIN32)
    std::string os = "windows";
#elif defined(__APPLE__)
    std::string os = "macos";
#else
    std::string os = "linux";
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    return os + "-arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return os + "-x64";
#else
    return os + "-other";
#endif
}

// Today's date in UTC, so every record of one day carries the same date wherever the PC is.
static std::string os_utc_day() {
    std::time_t t = std::time(nullptr);
    std::tm tmv = {};
#if defined(_WIN32)
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tmv);
    return buf;
}

//
// the record
//

static std::string env_str(const char * name) {
    const char * v = std::getenv(name);
    return v ? v : "";
}

static bool all_of_chars(const std::string & s, const char * allowed) {
    return s.find_first_not_of(allowed) == std::string::npos;
}

static bool is_install_id(const std::string & s) {
    return s.size() == 32 && all_of_chars(s, "0123456789abcdef");
}

static std::string random_chars(const char * alphabet, size_t n) {
    std::random_device rd;
    const size_t m = std::char_traits<char>::length(alphabet);
    std::uniform_int_distribution<size_t> pick(0, m - 1);
    std::string s;
    for (size_t i = 0; i < n; i++) {
        s += alphabet[pick(rd)];
    }
    return s;
}

static std::string read_first_line(const fs::path & p) {
    std::ifstream f(p);
    std::string line;
    std::getline(f, line);
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
        line.pop_back();
    }
    return line;
}

static bool write_file(const fs::path & p, const std::string & text) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << text << "\n";
    f.close();
    return !f.fail();
}

static std::string cut(std::string s, size_t n) {
    if (s.size() > n) {
        s.resize(n);
        // do not leave half of a UTF-8 character at the end
        while (!s.empty() && (static_cast<unsigned char>(s.back()) & 0xC0) == 0x80) {
            s.pop_back();
        }
        if (!s.empty() && (static_cast<unsigned char>(s.back()) & 0x80)) {
            s.pop_back();
        }
    }
    return s;
}

static json str_value(const std::string & s) {
    return json{ { "stringValue", s } };
}

struct usage_stats_state {
    std::mutex mutex;
    httplib::Client * client = nullptr; // the send in flight, for stop()
    std::atomic<bool> stopping{ false };
    std::atomic<bool> done{ false };
};

// Owned by the send thread too, so it stays valid when the thread outlives stop().
static std::shared_ptr<usage_stats_state> g_state;

static void send_record(const std::shared_ptr<usage_stats_state> & st, const std::string & launcher,
                        const std::vector<std::string> & gpus) {
    const fs::path dir = os_state_dir();
    if (dir.empty()) {
        throw std::runtime_error("no state folder");
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        throw std::runtime_error("cannot create the state folder: " + ec.message());
    }

    const std::string day = os_utc_day();
    const fs::path day_file = dir / ("stats-day-" + launcher);
    if (read_first_line(day_file) == day) {
        return; // today's record went out already
    }

    std::string install_id = env_str("TOKENWING_INSTALL_ID");
    if (!is_install_id(install_id)) {
        const fs::path id_file = dir / "install-id";
        install_id = read_first_line(id_file);
        if (!is_install_id(install_id)) {
            install_id = random_chars("0123456789abcdef", 32);
            if (!write_file(id_file, install_id)) {
                throw std::runtime_error("cannot write the install id");
            }
        }
    }

    const std::string version = std::string(llama_version()) + " (build " + std::to_string(llama_build_number()) +
                                ", commit " + llama_commit() + ")";
    json gpu_values = json::array();
    for (const auto & g : gpus) {
        gpu_values.push_back(str_value(cut(g, 120)));
    }
    json fields = {
        { "v",             json{ { "integerValue", "1" } } },
        { "installId",     str_value(install_id) },
        { "day",           str_value(day) },
        { "engineVersion", str_value(cut(version, 80)) },
        { "os",            str_value(cut(os_name(), 120)) },
        { "cpu",           str_value(cut(os_cpu_name(), 120)) },
        { "gpus",          json{ { "arrayValue", json{ { "values", gpu_values } } } } },
        { "launcher",      str_value(launcher) },
    };
    const std::string doc_id = random_chars("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789", 20);
    json body = {
        { "writes", json::array({ json{
            { "update",           json{ { "name", STATS_DOC + doc_id }, { "fields", fields } } },
            { "updateTransforms", json::array({ json{ { "fieldPath", "createdAt" }, { "setToServerValue", "REQUEST_TIME" } } }) },
            { "currentDocument",  json{ { "exists", false } } },
        } }) },
    };

    // test only: send to another address (plain http allowed), so tests never write to the real database
    std::string url = env_str("TOKENWING_STATS_URL");
    if (url.empty()) {
        url = STATS_URL;
    }
    auto [cli, parts] = common_http_client(url);
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(20, 0);
    cli.set_write_timeout(20, 0);
    cli.set_max_timeout(20000);
    {
        std::lock_guard<std::mutex> lock(st->mutex);
        if (st->stopping) {
            return;
        }
        st->client = &cli;
    }
    auto res = cli.Post(parts.path, body.dump(), "application/json");
    {
        std::lock_guard<std::mutex> lock(st->mutex);
        st->client = nullptr;
    }
    if (!res) {
        throw std::runtime_error("request failed: " + httplib::to_string(res.error()));
    }
    if (res->status != 200) {
        throw std::runtime_error("HTTP " + std::to_string(res->status) + ": " + cut(res->body, 300));
    }
    write_file(day_file, day);
    if (!st->stopping) {
        SRV_DBG("usage stats: record stored for %s\n", day.c_str());
    }
}

static std::string url_escape(const std::string & s) {
    static const char * hex = "0123456789ABCDEF";
    std::string out;
    for (unsigned char c : s) {
        if (isalnum(c) || c == '-' || c == '.' || c == '_' || c == '~') {
            out += (char) c;
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 15];
        }
    }
    return out;
}

// Asks the website's update check once a day, which counts this engine (its version and system, nothing that tells one
// PC from another). Only for an engine started by hand: the app makes its own check. The answer is not used.
static void send_update_check(const std::shared_ptr<usage_stats_state> & st) {
    const fs::path dir = os_state_dir();
    if (dir.empty()) {
        throw std::runtime_error("no state folder");
    }
    std::error_code ec;
    fs::create_directories(dir, ec);
    const std::string day = os_utc_day();
    const fs::path day_file = dir / "update-check-day";
    if (read_first_line(day_file) == day) {
        return;
    }
    const std::string version = cut(std::string(llama_version()) + " " + llama_commit(), 40);

    // test only: ask another address (plain http allowed)
    std::string url = env_str("TOKENWING_UPDATE_URL");
    if (url.empty()) {
        url = CHECK_URL;
    }
    auto [cli, parts] = common_http_client(url);
    cli.set_follow_location(false); // the count is made before the redirect; the list it leads to is not needed
    cli.set_connection_timeout(10, 0);
    cli.set_read_timeout(20, 0);
    cli.set_max_timeout(20000);
    {
        std::lock_guard<std::mutex> lock(st->mutex);
        if (st->stopping) {
            return;
        }
        st->client = &cli;
    }
    auto res = cli.Get(parts.path + "?repo=tokenwing&v=" + url_escape(version) + "&os=" + url_escape(os_kind()));
    {
        std::lock_guard<std::mutex> lock(st->mutex);
        st->client = nullptr;
    }
    if (!res) {
        throw std::runtime_error("request failed: " + httplib::to_string(res.error()));
    }
    // the website counts, then sends the asker on to the list (302); the list itself is not needed
    if (res->status != 302 && res->status != 200) {
        throw std::runtime_error("HTTP " + std::to_string(res->status));
    }
    write_file(day_file, day);
    if (!st->stopping) {
        SRV_DBG("update check: done for %s\n", day.c_str());
    }
}

void server_usage_stats_start(const common_params & params) {
    auto off = [](const char * name) {
        std::string v = env_str(name);
        return !v.empty() && v != "0";
    };
    if (g_state) {
        return;
    }
    std::string launcher = env_str("TOKENWING_LAUNCHER");
    if (launcher.empty() || launcher.size() > 32 || !all_of_chars(launcher, "abcdefghijklmnopqrstuvwxyz0123456789-")) {
        launcher = "direct";
    }
    const bool dnt = env_str("DO_NOT_TRACK") == "1";
    bool stats = params.usage_stats && !off("TOKENWING_NO_STATS") && !dnt;
    // --no-stats leaves the update check on, as in the app; it has its own switch
    bool check = launcher == "direct" && !off("TOKENWING_NO_UPDATE_CHECK") && !dnt;
#if !defined(CPPHTTPLIB_OPENSSL_SUPPORT)
    // a build without HTTPS cannot reach the database or the website; only the test addresses can be plain http
    stats = stats && env_str("TOKENWING_STATS_URL").rfind("http://", 0) == 0;
    check = check && env_str("TOKENWING_UPDATE_URL").rfind("http://", 0) == 0;
#endif
    if (!stats && !check) {
        return;
    }

    // read the device list here: the backends are freed on the main thread at exit
    std::vector<std::string> gpus;
    for (size_t i = 0; stats && i < ggml_backend_dev_count() && gpus.size() < 8; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            const char * desc = ggml_backend_dev_description(dev);
            gpus.push_back(desc ? desc : "");
        }
    }

    if (stats) {
        SRV_INF("%s", "usage stats: one anonymous record a day (engine version, OS, CPU and GPU names); "
                      "--no-stats or TOKENWING_NO_STATS=1 turns it off\n");
    }
    if (check) {
        SRV_INF("%s", "update check: once a day the engine asks runmyai.org for the newest release (sends the engine "
                      "version and system); TOKENWING_NO_UPDATE_CHECK=1 turns it off\n");
    }

    g_state = std::make_shared<usage_stats_state>();
    try {
        std::thread([st = g_state, launcher, gpus, stats, check]() {
            if (stats) {
                try {
                    send_record(st, launcher, gpus);
                } catch (const std::exception & e) {
                    if (!st->stopping) {
                        SRV_DBG("usage stats: not sent: %s\n", e.what());
                    }
                } catch (...) {
                }
            }
            if (check && !st->stopping) {
                try {
                    send_update_check(st);
                } catch (const std::exception & e) {
                    if (!st->stopping) {
                        SRV_DBG("update check: not done: %s\n", e.what());
                    }
                } catch (...) {
                }
            }
            st->done = true;
        }).detach();
    } catch (...) {
        g_state.reset(); // no thread could be started: send nothing
    }
}

void server_usage_stats_stop() {
    auto st = g_state;
    if (!st) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(st->mutex);
        st->stopping = true;
        if (st->client) {
            st->client->stop(); // a blocked read or write returns at once
        }
    }
    // give the thread a moment to finish; one stuck in a name lookup is left behind, it owns its state and logs nothing
    for (int i = 0; i < 100 && !st->done; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
