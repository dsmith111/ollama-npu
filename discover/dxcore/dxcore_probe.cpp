// DXCore NPU probe implementation — enumerates adapters and tests device creation.
// Pattern derived from ggml-directml.cpp init_dxcore() (lines 1240–1500).
// All DLLs are loaded dynamically so the binary runs on systems without them.

#include "dxcore_probe.h"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <initguid.h>
#include <dxcore.h>
#include <dxcore_interface.h>
#include <wrl/client.h>
#define DML_TARGET_VERSION_USE_LATEST
#include <DirectML.h>

#include <cstdio>
#include <cstring>

#if defined(__MINGW32__)
// These SDK-only interfaces are absent from MinGW's UUID registry.
template<> inline const GUID &__mingw_uuidof<IDXCoreAdapterFactory>() { return IID_IDXCoreAdapterFactory; }
template<> inline const GUID &__mingw_uuidof<IDXCoreAdapterFactory1>() { return IID_IDXCoreAdapterFactory1; }
template<> inline const GUID &__mingw_uuidof<IDXCoreAdapterList>() { return IID_IDXCoreAdapterList; }
template<> inline const GUID &__mingw_uuidof<IDXCoreAdapter>() { return IID_IDXCoreAdapter; }
__CRT_UUID_DECL(IDMLDevice1, 0xa0884f9a, 0xd2be, 0x4355, 0xaa, 0x5d, 0x59, 0x01, 0x28, 0x1a, 0xd1, 0xd2)
#endif

using Microsoft::WRL::ComPtr;

// MinGW's d3dcommon.h does not yet name the Windows SDK generic-ML level.
static constexpr D3D_FEATURE_LEVEL generic_ml_level = static_cast<D3D_FEATURE_LEVEL>(0x100);

// ---------------------------------------------------------------------------
// Dynamic loaders (same pattern as ggml-directml.cpp)
// ---------------------------------------------------------------------------

typedef HRESULT (WINAPI *PFN_DMLCreateDevice1)(
    ID3D12Device *d3d12Device, DML_CREATE_DEVICE_FLAGS flags,
    DML_FEATURE_LEVEL minFeatureLevel, REFIID riid, void **ppv);

typedef HRESULT (WINAPI *PFN_DXCoreCreateAdapterFactory)(REFIID riid, void **ppvFactory);

static PFN_DMLCreateDevice1           s_DMLCreateDevice1 = nullptr;
static PFN_DXCoreCreateAdapterFactory s_DXCoreCreateAdapterFactory = nullptr;
static HMODULE                        s_dml_module = nullptr;
static HMODULE                        s_dxcore_module = nullptr;

static bool load_dml() {
    if (s_dml_module) return true;
    s_dml_module = LoadLibraryW(L"DirectML.dll");
    if (!s_dml_module) return false;
    s_DMLCreateDevice1 = (PFN_DMLCreateDevice1)GetProcAddress(s_dml_module, "DMLCreateDevice1");
    if (!s_DMLCreateDevice1) {
        FreeLibrary(s_dml_module);
        s_dml_module = nullptr;
        return false;
    }
    return true;
}

static bool load_dxcore() {
    if (s_dxcore_module) return true;
    s_dxcore_module = LoadLibraryW(L"dxcore.dll");
    if (!s_dxcore_module) return false;
    s_DXCoreCreateAdapterFactory = (PFN_DXCoreCreateAdapterFactory)
        GetProcAddress(s_dxcore_module, "DXCoreCreateAdapterFactory");
    if (!s_DXCoreCreateAdapterFactory) {
        FreeLibrary(s_dxcore_module);
        s_dxcore_module = nullptr;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Probe implementation
// ---------------------------------------------------------------------------

extern "C" void dxcore_probe(DxcoreProbeResult *out) {
    memset(out, 0, sizeof(*out));

    if (!load_dxcore()) {
        snprintf(out->error, sizeof(out->error), "dxcore.dll not available");
        return;
    }
    out->dxcore_available = 1;

    bool dml_available = load_dml();

    // Create adapter factory
    ComPtr<IDXCoreAdapterFactory> factory;
    HRESULT hr = s_DXCoreCreateAdapterFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        snprintf(out->error, sizeof(out->error),
                 "DXCoreCreateAdapterFactory failed: 0x%08x", (unsigned)hr);
        return;
    }

    // Try modern CreateAdapterListByWorkload first (IDXCoreAdapterFactory1)
    ComPtr<IDXCoreAdapterList> adapter_list;
    ComPtr<IDXCoreAdapterFactory1> factory1;
    hr = factory->QueryInterface(IID_PPV_ARGS(&factory1));
    if (SUCCEEDED(hr) && factory1) {
        hr = factory1->CreateAdapterListByWorkload(
            DXCoreWorkload::MachineLearning,
            DXCoreRuntimeFilterFlags::D3D12,
            static_cast<DXCoreHardwareTypeFilterFlags>(
                static_cast<uint32_t>(DXCoreHardwareTypeFilterFlags::NPU) |
                static_cast<uint32_t>(DXCoreHardwareTypeFilterFlags::GPU)),
            IID_PPV_ARGS(&adapter_list));
    }

    // Fallback to legacy D3D12_CORE_COMPUTE attribute list
    if (!adapter_list) {
        const GUID attributes[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE };
        hr = factory->CreateAdapterList(1, attributes, IID_PPV_ARGS(&adapter_list));
        if (FAILED(hr) || !adapter_list) {
            snprintf(out->error, sizeof(out->error),
                     "DXCore CreateAdapterList failed: 0x%08x", (unsigned)hr);
            return;
        }
    }

    uint32_t count = adapter_list->GetAdapterCount();
    for (uint32_t i = 0; i < count && out->adapter_count < DXCORE_PROBE_MAX_ADAPTERS; i++) {
        ComPtr<IDXCoreAdapter> adapter;
        hr = adapter_list->GetAdapter(i, IID_PPV_ARGS(&adapter));
        if (FAILED(hr) || !adapter->IsValid()) continue;

        DxcoreAdapterInfo *info = &out->adapters[out->adapter_count];
        info->index = (int)i;

        // Description
        size_t desc_size = 0;
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverDescription)) {
            adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription, &desc_size);
            if (desc_size > 0 && desc_size <= sizeof(info->description) - 1) {
                adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
                                     desc_size, info->description);
            }
        }

        // LUID
        LUID adapter_luid = {};
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::InstanceLuid)) {
            adapter->GetProperty(DXCoreAdapterProperty::InstanceLuid,
                                 sizeof(adapter_luid), &adapter_luid);
            memcpy(info->luid, &adapter_luid, sizeof(adapter_luid));
        }

        // Hardware type classification
        info->is_npu = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU) ? 1 : 0;
        info->is_gpu = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU) ? 1 : 0;

        // D3D12 runtime support attributes
        info->has_d3d12_graphics     = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS) ? 1 : 0;
        info->has_d3d12_core_compute = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE) ? 1 : 0;
        info->has_d3d12_generic_ml   = adapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML) ? 1 : 0;

        // Driver version
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::DriverVersion)) {
            adapter->GetProperty(DXCoreAdapterProperty::DriverVersion,
                                 sizeof(info->driver_version), &info->driver_version);
        }

        // IsHardware
        bool is_hw = false;
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsHardware)) {
            adapter->GetProperty(DXCoreAdapterProperty::IsHardware, sizeof(is_hw), &is_hw);
        }
        info->is_hardware = is_hw ? 1 : 0;
        if (!is_hw) continue; // skip software adapters

        // IsIntegrated
        bool is_int = false;
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::IsIntegrated)) {
            adapter->GetProperty(DXCoreAdapterProperty::IsIntegrated, sizeof(is_int), &is_int);
        }
        info->is_integrated = is_int ? 1 : 0;

        // Memory
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::DedicatedAdapterMemory)) {
            size_t mem = 0;
            adapter->GetProperty(DXCoreAdapterProperty::DedicatedAdapterMemory, sizeof(mem), &mem);
            info->dedicated_mem_bytes = (uint64_t)mem;
        }
        if (adapter->IsPropertySupported(DXCoreAdapterProperty::SharedSystemMemory)) {
            size_t mem = 0;
            adapter->GetProperty(DXCoreAdapterProperty::SharedSystemMemory, sizeof(mem), &mem);
            info->shared_mem_bytes = (uint64_t)mem;
        }

        // Try D3D12 device creation
        ComPtr<ID3D12Device> d3d_device;
        if (!info->has_d3d12_graphics) {
            // NPU/MCDM: try Core/Generic feature levels
            struct { D3D_FEATURE_LEVEL fl; const char *name; } levels[] = {
                { generic_ml_level,             "1_0_GENERIC" },
                { D3D_FEATURE_LEVEL_1_0_CORE,    "1_0_CORE" },
            };
            for (auto &lvl : levels) {
                hr = D3D12CreateDevice(adapter.Get(), lvl.fl, IID_PPV_ARGS(&d3d_device));
                if (SUCCEEDED(hr)) {
                    strncpy(info->feature_level, lvl.name, sizeof(info->feature_level) - 1);
                    break;
                }
            }
        } else {
            // GPU: try DXGI LUID lookup first, then DXCore fallback
            ComPtr<IDXGIFactory4> dxgi_factory;
            hr = CreateDXGIFactory1(IID_PPV_ARGS(&dxgi_factory));
            if (SUCCEEDED(hr)) {
                ComPtr<IDXGIAdapter1> dxgi_adapter;
                hr = dxgi_factory->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&dxgi_adapter));
                if (SUCCEEDED(hr)) {
                    hr = D3D12CreateDevice(dxgi_adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                           IID_PPV_ARGS(&d3d_device));
                    if (SUCCEEDED(hr)) {
                        strncpy(info->feature_level, "11_0", sizeof(info->feature_level) - 1);
                    }
                }
            }

            // DXCore fallback for graphics adapters
            if (!d3d_device) {
                struct { D3D_FEATURE_LEVEL fl; const char *name; } levels[] = {
                    { D3D_FEATURE_LEVEL_11_0,       "11_0" },
                    { D3D_FEATURE_LEVEL_1_0_CORE,   "1_0_CORE" },
                    { generic_ml_level,             "1_0_GENERIC" },
                };
                for (auto &lvl : levels) {
                    hr = D3D12CreateDevice(adapter.Get(), lvl.fl, IID_PPV_ARGS(&d3d_device));
                    if (SUCCEEDED(hr)) {
                        strncpy(info->feature_level, lvl.name, sizeof(info->feature_level) - 1);
                        break;
                    }
                }
            }
        }

        info->has_d3d12 = d3d_device ? 1 : 0;

        // Try DML device creation if D3D12 succeeded
        if (d3d_device && dml_available) {
            ComPtr<IDMLDevice1> dml_device;
            hr = s_DMLCreateDevice1(d3d_device.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                    DML_FEATURE_LEVEL_1_0, IID_PPV_ARGS(&dml_device));
            info->has_dml = SUCCEEDED(hr) ? 1 : 0;
        }

        // Update counts
        if (info->is_npu) out->npu_count++;
        if (info->is_gpu) out->gpu_count++;
        out->adapter_count++;
    }
}

#else // !_WIN32

extern "C" void dxcore_probe(DxcoreProbeResult *out) {
    memset(out, 0, sizeof(*out));
    snprintf(out->error, sizeof(out->error), "DXCore only available on Windows");
}

#endif
