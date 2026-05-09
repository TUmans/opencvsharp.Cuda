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


// DeviceInfo

#pragma endregion



#endif
