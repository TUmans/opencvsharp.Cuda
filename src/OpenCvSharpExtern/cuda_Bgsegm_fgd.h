#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudalegacy.hpp>

// ---------- createBackgroundSubtractorFGD --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createBackgroundSubtractorFGD(cv::Ptr<cv::cuda::BackgroundSubtractorFGD> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createBackgroundSubtractorFGD();
    *returnValue = new cv::Ptr<cv::cuda::BackgroundSubtractorFGD>(ptr);
    END_WRAP
}

// ---------- BackgroundSubtractorFGD_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorFGD_get(cv::Ptr<cv::cuda::BackgroundSubtractorFGD> *ptr, cv::cuda::BackgroundSubtractorFGD **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- BackgroundSubtractorFGD_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorFGD_delete(cv::Ptr<cv::cuda::BackgroundSubtractorFGD> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- BackgroundSubtractorFGD_apply --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorFGD_apply(cv::cuda::BackgroundSubtractorFGD *obj, cv::_InputArray *image, cv::_OutputArray *fgmask, double learningRate)
{
    BEGIN_WRAP
    obj->apply(*image, *fgmask, learningRate);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_BackgroundSubtractorFGD_getForegroundRegions(cv::cuda::BackgroundSubtractorFGD *obj, cv::Mat ***outMats, int *outCount)
{
    BEGIN_WRAP
    // 1. Get the regions into a C++ vector
    std::vector<cv::Mat> regions;
    obj->getForegroundRegions(regions);

    // 2. Determine count
    *outCount = static_cast<int>(regions.size());

    // 3. Allocate an array of pointers
    cv::Mat **mats = new cv::Mat *[*outCount];

    // 4. Copy each region into a new heap-allocated Mat so C# can take ownership
    for (int i = 0; i < *outCount; i++)
    {
        mats[i] = new cv::Mat(regions[i]);
    }
    // 5. Assign output
    *outMats = mats;
    END_WRAP
}
