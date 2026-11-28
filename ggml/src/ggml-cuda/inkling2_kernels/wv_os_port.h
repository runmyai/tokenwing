#pragma once
/* Cross-platform OS portability, memory-mapping, WDDM commit inspection, and pinned RAM helpers
 * for wv_runtime_core (Windows _WIN32 + Linux POSIX). Zero PyTorch/pybind dependencies. */

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace wv {

std::string& tls_last_error();

inline int32_t set_error(const std::string& msg) {
    tls_last_error() = msg;
    return -1;
}

inline void clear_error() {
    tls_last_error().clear();
}

#define WV_CUDA_CHECK(expr)                                                               \
    do {                                                                                  \
        cudaError_t _wv_err = (expr);                                                     \
        if (_wv_err != cudaSuccess) {                                                     \
            return ::wv::set_error(std::string("CUDA error (L") +                         \
                                   std::to_string(__LINE__) + "): " +                     \
                                   cudaGetErrorString(_wv_err));                          \
        }                                                                                 \
    } while (0)

struct DeviceGuard {
    int prev_dev = -1;
    bool switched = false;
    explicit DeviceGuard(int target_dev) {
        if (target_dev >= 0 && cudaGetDevice(&prev_dev) == cudaSuccess && prev_dev != target_dev) {
            if (cudaSetDevice(target_dev) == cudaSuccess) {
                switched = true;
            }
        }
    }
    ~DeviceGuard() {
        if (switched && prev_dev >= 0) {
            cudaSetDevice(prev_dev);
        }
    }
    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
};

struct WvMmapFile {
    const uint8_t* data = nullptr;
    size_t size = 0;
#ifdef _WIN32
    HANDLE hFile = INVALID_HANDLE_VALUE;
    HANDLE hMap  = nullptr;
#else
    int fd = -1;
#endif

    WvMmapFile() = default;
    ~WvMmapFile() { close(); }
    WvMmapFile(const WvMmapFile&) = delete;
    WvMmapFile& operator=(const WvMmapFile&) = delete;

    bool open(const char* path) {
        close();
        if (!path || !*path) {
            set_error("Empty file path for mmap");
            return false;
        }
#ifdef _WIN32
        hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            set_error(std::string("CreateFileA failed for: ") + path);
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz) || sz.QuadPart <= 0) {
            set_error(std::string("GetFileSizeEx failed or empty file: ") + path);
            close();
            return false;
        }
        size = static_cast<size_t>(sz.QuadPart);
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            set_error(std::string("CreateFileMappingA failed for: ") + path);
            close();
            return false;
        }
        void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!view) {
            set_error(std::string("MapViewOfFile failed for: ") + path);
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(view);
        return true;
#else
        fd = ::open(path, O_RDONLY);
        if (fd < 0) {
            set_error(std::string("open() failed for: ") + path);
            return false;
        }
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size <= 0) {
            set_error(std::string("fstat() failed or empty file: ") + path);
            close();
            return false;
        }
        size = static_cast<size_t>(st.st_size);
        void* p = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) {
            set_error(std::string("mmap() failed for: ") + path);
            close();
            return false;
        }
        data = static_cast<const uint8_t*>(p);
        return true;
#endif
    }

    void close() {
#ifdef _WIN32
        if (data) { UnmapViewOfFile(data); data = nullptr; }
        if (hMap) { CloseHandle(hMap); hMap = nullptr; }
        if (hFile != INVALID_HANDLE_VALUE) { CloseHandle(hFile); hFile = INVALID_HANDLE_VALUE; }
#else
        if (data && size > 0) { munmap(const_cast<uint8_t*>(data), size); data = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
        size = 0;
    }
};

struct WvPinnedBuffer {
    uint8_t* ptr       = nullptr;
    size_t   size      = 0;
    bool     is_pinned = false;

    WvPinnedBuffer() = default;
    ~WvPinnedBuffer() { release(); }
    WvPinnedBuffer(const WvPinnedBuffer&) = delete;
    WvPinnedBuffer& operator=(const WvPinnedBuffer&) = delete;
    WvPinnedBuffer(WvPinnedBuffer&& o) noexcept : ptr(o.ptr), size(o.size), is_pinned(o.is_pinned) {
        o.ptr = nullptr; o.size = 0; o.is_pinned = false;
    }
    WvPinnedBuffer& operator=(WvPinnedBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr = o.ptr; size = o.size; is_pinned = o.is_pinned;
            o.ptr = nullptr; o.size = 0; o.is_pinned = false;
        }
        return *this;
    }

    bool allocate(size_t nbytes, bool allow_host_fallback = true) {
        release();
        if (nbytes == 0) return true;
        void* p = nullptr;
        cudaError_t err = cudaHostAlloc(&p, nbytes, cudaHostAllocPortable);
        if (err == cudaSuccess && p != nullptr) {
            ptr = static_cast<uint8_t*>(p);
            size = nbytes;
            is_pinned = true;
            return true;
        }
        cudaGetLastError();
        if (!allow_host_fallback) {
            set_error(std::string("cudaHostAlloc failed (") + std::to_string(nbytes) +
                      " B). On Windows, increase virtual memory Pagefile size.");
            return false;
        }
        void* fallback = std::malloc(nbytes);
        if (!fallback) {
            set_error("Host malloc fallback failed");
            return false;
        }
        ptr = static_cast<uint8_t*>(fallback);
        size = nbytes;
        is_pinned = false;
        return true;
    }

    void release() {
        if (ptr) {
            if (is_pinned) cudaFreeHost(ptr);
            else std::free(ptr);
            ptr = nullptr;
        }
        size = 0;
        is_pinned = false;
    }
};

} // namespace wv
