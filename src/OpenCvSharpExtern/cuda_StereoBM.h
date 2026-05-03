#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/cudastereo.hpp>

// ---------- cuda_createStereoBM --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createStereoBM(int numDisparities, int blockSize, cv::Ptr<cv::cuda::StereoBM> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createStereoBM(numDisparities, blockSize);
    *returnValue = new cv::Ptr<cv::cuda::StereoBM>(ptr);
    END_WRAP
}

// ---------- cuda_StereoBM_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoBM_get(cv::Ptr<cv::cuda::StereoBM> *ptr, cv::cuda::StereoBM **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_StereoBM_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoBM_delete(cv::Ptr<cv::cuda::StereoBM> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}
