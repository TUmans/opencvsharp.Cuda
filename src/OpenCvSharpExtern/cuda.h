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

#pragma endregion

#pragma region Stream




#pragma endregion

#endif
