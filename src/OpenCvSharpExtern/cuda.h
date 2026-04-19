#pragma once

#ifdef ENABLED_CUDA

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>

#pragma region Device

CVAPI(int) cuda_getCudaEnabledDeviceCount()
{
    return cv::cuda::getCudaEnabledDeviceCount();
}

CVAPI(void) cuda_setDevice(int device)
{
    cv::cuda::setDevice(device);
}

CVAPI(int) cuda_getDevice()
{
    return cv::cuda::getDevice();
}

CVAPI(void) cuda_resetDevice()
{
    cv::cuda::resetDevice();
}

CVAPI(int) cuda_deviceSupports(int feature_set)
{
    return cv::cuda::deviceSupports(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
}

// TargetArchs
CVAPI(int) cuda_TargetArchs_builtWith(int feature_set)
{
    return cv::cuda::TargetArchs::builtWith(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_has(int major, int minor)
{
    return cv::cuda::TargetArchs::has(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasPtx(int major, int minor)
{
    return cv::cuda::TargetArchs::hasPtx(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasBin(int major, int minor)
{
    return cv::cuda::TargetArchs::hasBin(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasEqualOrLessPtx(int major, int minor)
{
    return cv::cuda::TargetArchs::hasEqualOrLessPtx(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasEqualOrGreater(int major, int minor)
{
    return cv::cuda::TargetArchs::hasEqualOrGreater(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasEqualOrGreaterPtx(int major, int minor)
{
    return cv::cuda::TargetArchs::hasEqualOrGreaterPtx(major, minor) ? 1 : 0;
}
CVAPI(int) cuda_TargetArchs_hasEqualOrGreaterBin(int major, int minor)
{
    return cv::cuda::TargetArchs::hasEqualOrGreaterBin(major, minor) ? 1 : 0;
}

// DeviceInfo
CVAPI(cv::cuda::DeviceInfo *) cuda_DeviceInfo_new1()
{
    return new cv::cuda::DeviceInfo();
}
CVAPI(cv::cuda::DeviceInfo *) cuda_DeviceInfo_new2(int deviceId)
{
    return new cv::cuda::DeviceInfo(deviceId);
}
CVAPI(void) cuda_DeviceInfo_delete(cv::cuda::DeviceInfo *obj)
{
    delete obj;
}

CVAPI(void) cuda_DeviceInfo_name(cv::cuda::DeviceInfo *obj, char *buf, int bufLength)
{
    copyString(obj->name(), buf, bufLength);
}
CVAPI(int) cuda_DeviceInfo_majorVersion(cv::cuda::DeviceInfo *obj)
{
    return obj->majorVersion();
}
CVAPI(int) cuda_DeviceInfo_minorVersion(cv::cuda::DeviceInfo *obj)
{
    return obj->minorVersion();
}
CVAPI(int) cuda_DeviceInfo_multiProcessorCount(cv::cuda::DeviceInfo *obj)
{
    return obj->multiProcessorCount();
}
CVAPI(uint64_t) cuda_DeviceInfo_sharedMemPerBlock(cv::cuda::DeviceInfo *obj)
{
    return (uint64_t)obj->sharedMemPerBlock();
}
CVAPI(void) cuda_DeviceInfo_queryMemory(
    cv::cuda::DeviceInfo *obj, uint64_t *totalMemory, uint64_t *freeMemory)
{
    size_t totalMemory0, freeMemory0;
    obj->queryMemory(totalMemory0, freeMemory0);
    *totalMemory = (uint64_t)totalMemory0;
    *freeMemory = (uint64_t)freeMemory0;
}
CVAPI(uint64_t) cuda_DeviceInfo_freeMemory(cv::cuda::DeviceInfo *obj)
{
    return (uint64_t)obj->freeMemory();
}
CVAPI(uint64_t) cuda_DeviceInfo_totalMemory(cv::cuda::DeviceInfo *obj)
{
    return (uint64_t)obj->totalMemory();
}
CVAPI(int) cuda_DeviceInfo_supports(cv::cuda::DeviceInfo *obj, int feature_set)
{
    return obj->supports(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
}
CVAPI(int) cuda_DeviceInfo_isCompatible(cv::cuda::DeviceInfo *obj)
{
    return obj->isCompatible() ? 1 : 0;
}
CVAPI(int) cuda_DeviceInfo_deviceID(cv::cuda::DeviceInfo *obj)
{
    return obj->deviceID();
}
CVAPI(int) cuda_DeviceInfo_canMapHostMemory(cv::cuda::DeviceInfo *obj)
{
    return obj->canMapHostMemory() ? 1 : 0;
}

CVAPI(void) cuda_printCudaDeviceInfo(int device)
{
    cv::cuda::printCudaDeviceInfo(device);
}
CVAPI(void) cuda_printShortCudaDeviceInfo(int device)
{
    cv::cuda::printShortCudaDeviceInfo(device);
}

#pragma endregion

#pragma region Stream

CVAPI(cv::cuda::Stream *) cuda_Stream_new1()
{
    return new cv::cuda::Stream();
}
CVAPI(cv::cuda::Stream *) cuda_Stream_new2(cv::cuda::Stream *s)
{
    return new cv::cuda::Stream(*s);
}

CVAPI(void) cuda_Stream_delete(cv::cuda::Stream *obj)
{
    delete obj;
}

CVAPI(void) cuda_Stream_opAssign(cv::cuda::Stream *left, cv::cuda::Stream *right)
{
    *left = *right;
}

CVAPI(int) cuda_Stream_queryIfComplete(cv::cuda::Stream *obj)
{
    return obj->queryIfComplete() ? 1 : 0;
}
CVAPI(void) cuda_Stream_waitForCompletion(cv::cuda::Stream *obj)
{
    obj->waitForCompletion();
}

CVAPI(void) cuda_Stream_enqueueHostCallback(
    cv::cuda::Stream *obj, cv::cuda::Stream::StreamCallback callback, void *userData)
{
    obj->enqueueHostCallback(callback, userData);
}

CVAPI(cv::cuda::Stream *) cuda_Stream_Null()
{
    return const_cast<cv::cuda::Stream *>(&cv::cuda::Stream::Null());
}

CVAPI(int) cuda_Stream_bool(cv::cuda::Stream *obj)
{
    return (bool)(*obj) ? 1 : 0;
}

#pragma endregion

#endif
