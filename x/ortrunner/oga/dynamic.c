#include "dynamic.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <wchar.h>

#ifdef _WIN32
#include <windows.h>
#define DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define DLCLOSE(handle) dlclose(handle)
#endif

#ifdef _WIN32
static int oga_dll_dirs_initialized = 0;
#endif

static int oga_dynamic_open(oga_dynamic_handle* handle, const char* path) {
    handle->ctx = NULL;
#ifdef _WIN32
    handle->directory_cookie = NULL;
    if (!oga_dll_dirs_initialized) {
        if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) return 1;
        oga_dll_dirs_initialized = 1;
    }
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (count == 0) return 1;
    wchar_t* wide_path = malloc((size_t)count * sizeof(wchar_t));
    if (!wide_path) return 1;
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide_path, count)) {
        free(wide_path);
        return 1;
    }
    DWORD flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    if (wcschr(wide_path, L'\\') || wcschr(wide_path, L'/')) {
        DWORD capacity = GetFullPathNameW(wide_path, 0, NULL, NULL);
        wchar_t* absolute = capacity ? malloc((size_t)capacity * sizeof(wchar_t)) : NULL;
        DWORD length = absolute ? GetFullPathNameW(wide_path, capacity, absolute, NULL) : 0;
        free(wide_path);
        if (!length || length >= capacity) {
            free(absolute);
            return 1;
        }
        wide_path = absolute;
        wchar_t* sep = wcsrchr(wide_path, L'\\');
        if (!sep) sep = wcsrchr(wide_path, L'/');
        if (!sep) {
            free(wide_path);
            return 1;
        }
        wchar_t* end = sep;
        if (sep == wide_path + 2 && wide_path[1] == L':') end = sep + 1;
        wchar_t saved = *end;
        *end = L'\0';
        // Keep only the successful load's directory for lazily loaded providers.
        handle->directory_cookie = AddDllDirectory(wide_path);
        *end = saved;
        if (!handle->directory_cookie) {
            free(wide_path);
            return 1;
        }
        flags |= LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
    }
    handle->ctx = (void*)LoadLibraryExW(wide_path, NULL, flags);
    free(wide_path);
    if (!handle->ctx && handle->directory_cookie) {
        RemoveDllDirectory(handle->directory_cookie);
        handle->directory_cookie = NULL;
    }
#else
    handle->ctx = (void*) DLOPEN(path);
#endif
    if (handle->ctx == NULL) {
        return 1;
    }
    return 0;
}

int oga_dynamic_load(oga_dynamic_handle* handle, const char *path) {
    return oga_dynamic_open(handle, path);
}

void oga_dynamic_unload(oga_dynamic_handle* handle) {
    if (handle->ctx) {
        DLCLOSE(handle->ctx);
        handle->ctx = NULL;
    }
#ifdef _WIN32
    if (handle->directory_cookie) {
        RemoveDllDirectory(handle->directory_cookie);
        handle->directory_cookie = NULL;
    }
#endif
}
