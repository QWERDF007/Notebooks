# C API (`dlpack.h`)

## Macros

- `DLPACK_EXTERN_C` ：与 C++ 的兼容性。

- `DLPACK_MAJOR_VERSION` ：dlpack 当前的主要版本。

- `DLPACK_MINOR_VERSION` ：dlpack 当前的次要版本。

- `DLPACK_DLL` ：Windows 的 DLPACK_DLL 前缀。

## Enumerations

### enum `DLDeviceType`

[DLDevice](#struct-DLDevice) 中的设备类型。

值：

- `kDLCPU` ：CPU 设备。
- `kDLCUDA` ：CUDA GPU 设备。
- `kDLCUDAHost` ：通过 `cudaMallocHost` 分配的锁页 CUDA GPU 内存。
- `kDLOpenCL` ：OpenCL 设备。
- `kDLVulkan` ：用于下一代图形的 Vulkan 缓冲区。
- `kDLMetal` ：Apple 的 GPU Metal
- 



## Structs

### struct `DLDevice`