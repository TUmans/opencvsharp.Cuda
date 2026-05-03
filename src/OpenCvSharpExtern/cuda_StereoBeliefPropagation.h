#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/cudastereo.hpp>

// ---------- cuda_createStereoBeliefPropagation --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createStereoBeliefPropagation(int ndisp, int iters, int levels, int msg_type, cv::Ptr<cv::cuda::StereoBeliefPropagation> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createStereoBeliefPropagation(ndisp, iters, levels, msg_type);
    *returnValue = new cv::Ptr<cv::cuda::StereoBeliefPropagation>(ptr);
    END_WRAP
}

// ---------- cuda_StereoBeliefPropagation_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoBeliefPropagation_get(cv::Ptr<cv::cuda::StereoBeliefPropagation> *ptr, cv::cuda::StereoBeliefPropagation **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_StereoBeliefPropagation_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoBeliefPropagation_delete(cv::Ptr<cv::cuda::StereoBeliefPropagation> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}
