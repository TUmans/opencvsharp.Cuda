#pragma once

// ReSharper disable IdentifierTypo
// ReSharper disable CppInconsistentNaming
// ReSharper disable CppNonInlineFunctionDefinitionInHeaderFile

#ifdef ENABLED_CUDA

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>

#pragma region Device

CVAPI(ExceptionStatus) cuda_getCudaEnabledDeviceCount(int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::getCudaEnabledDeviceCount();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_setDevice(int device)
{
    BEGIN_WRAP
    cv::cuda::setDevice(device);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_getDevice(int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::getDevice();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_resetDevice()
{
    BEGIN_WRAP
    cv::cuda::resetDevice();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_deviceSupports(int feature_set, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::deviceSupports(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
    END_WRAP
}

// TargetArchs
CVAPI(ExceptionStatus) cuda_TargetArchs_builtWith(int feature_set, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::builtWith(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_has(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::has(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasPtx(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasPtx(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasBin(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasBin(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasEqualOrLessPtx(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasEqualOrLessPtx(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasEqualOrGreater(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasEqualOrGreater(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasEqualOrGreaterPtx(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasEqualOrGreaterPtx(major, minor) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_TargetArchs_hasEqualOrGreaterBin(int major, int minor, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::TargetArchs::hasEqualOrGreaterBin(major, minor) ? 1 : 0;
    END_WRAP
}

// DeviceInfo
CVAPI(ExceptionStatus) cuda_DeviceInfo_new1(cv::cuda::DeviceInfo **returnValue)
{
    BEGIN_WRAP
    *returnValue = new cv::cuda::DeviceInfo();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_new2(int deviceId, cv::cuda::DeviceInfo **returnValue)
{
    BEGIN_WRAP
    *returnValue = new cv::cuda::DeviceInfo(deviceId);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_delete(cv::cuda::DeviceInfo *obj)
{
    BEGIN_WRAP
    delete obj;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_name(cv::cuda::DeviceInfo *obj, char *buf, int bufLength)
{
    BEGIN_WRAP
    copyString(obj->name(), buf, bufLength);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_majorVersion(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->majorVersion();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_minorVersion(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->minorVersion();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_multiProcessorCount(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->multiProcessorCount();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_sharedMemPerBlock(cv::cuda::DeviceInfo *obj, uint64_t *returnValue)
{
    BEGIN_WRAP
    *returnValue = static_cast<uint64_t>(obj->sharedMemPerBlock());
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_queryMemory(cv::cuda::DeviceInfo *obj, uint64_t *totalMemory, uint64_t *freeMemory)
{
    BEGIN_WRAP
    size_t totalMemory0, freeMemory0;
    obj->queryMemory(totalMemory0, freeMemory0);
    *totalMemory = static_cast<uint64_t>(totalMemory0);
    *freeMemory = static_cast<uint64_t>(freeMemory0);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_freeMemory(cv::cuda::DeviceInfo *obj, uint64_t *returnValue)
{
    BEGIN_WRAP
    *returnValue = static_cast<uint64_t>(obj->freeMemory());
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_totalMemory(cv::cuda::DeviceInfo *obj, uint64_t *returnValue)
{
    BEGIN_WRAP
    *returnValue = static_cast<uint64_t>(obj->totalMemory());
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_supports(cv::cuda::DeviceInfo *obj, int feature_set, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->supports(static_cast<cv::cuda::FeatureSet>(feature_set)) ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_isCompatible(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->isCompatible() ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_deviceID(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->deviceID();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_DeviceInfo_canMapHostMemory(cv::cuda::DeviceInfo *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->canMapHostMemory() ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_printCudaDeviceInfo(int device)
{
    BEGIN_WRAP
    cv::cuda::printCudaDeviceInfo(device);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_printShortCudaDeviceInfo(int device)
{
    BEGIN_WRAP
    cv::cuda::printShortCudaDeviceInfo(device);
    END_WRAP
}

#pragma endregion

#pragma region Stream

CVAPI(ExceptionStatus) cuda_Stream_new1(cv::cuda::Stream **returnValue)
{
    BEGIN_WRAP
    *returnValue = new cv::cuda::Stream();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_new2(cv::cuda::Stream *s, cv::cuda::Stream **returnValue)
{
    BEGIN_WRAP
    *returnValue = new cv::cuda::Stream(*s);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_delete(cv::cuda::Stream *obj)
{
    BEGIN_WRAP
    delete obj;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_opAssign(cv::cuda::Stream *left, cv::cuda::Stream *right)
{
    BEGIN_WRAP
    *left = *right;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_queryIfComplete(cv::cuda::Stream *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = obj->queryIfComplete() ? 1 : 0;
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_waitForCompletion(cv::cuda::Stream *obj)
{
    BEGIN_WRAP
    obj->waitForCompletion();
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_enqueueHostCallback(cv::cuda::Stream *obj, cv::cuda::Stream::StreamCallback callback, void *userData)
{
    BEGIN_WRAP
    obj->enqueueHostCallback(callback, userData);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_Null(cv::cuda::Stream **returnValue)
{
    BEGIN_WRAP
    *returnValue = const_cast<cv::cuda::Stream *>(&cv::cuda::Stream::Null());
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_Stream_bool(cv::cuda::Stream *obj, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = (bool)(*obj) ? 1 : 0;
    END_WRAP
}

#pragma endregion

#endif
