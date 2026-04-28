#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/cudastereo.hpp>

// ---------- createDisparityBilateralFilter --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createDisparityBilateralFilter( int ndisp, int radius, int iters, cv::Ptr<cv::cuda::DisparityBilateralFilter> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createDisparityBilateralFilter(ndisp, radius, iters);
    *returnValue = new cv::Ptr<cv::cuda::DisparityBilateralFilter>(ptr);
    END_WRAP
}

// ---------- DisparityBilateralFilter_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_DisparityBilateralFilter_get( cv::Ptr<cv::cuda::DisparityBilateralFilter> *ptr, cv::cuda::DisparityBilateralFilter **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- DisparityBilateralFilter_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_DisparityBilateralFilter_delete(cv::Ptr<cv::cuda::DisparityBilateralFilter> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- DisparityBilateralFilter_apply --------------------------------------------------
CVAPI(ExceptionStatus) cuda_DisparityBilateralFilter_apply( cv::cuda::DisparityBilateralFilter *obj, cv::_InputArray *disparity,  cv::_InputArray *image, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->apply(*disparity, *image, *dst, streamRef);
    END_WRAP
}

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

// ---------- cuda_createStereoConstantSpaceBP --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createStereoConstantSpaceBP(int ndisp, int iters, int levels, int nr_plane, int msg_type, cv::Ptr<cv::cuda::StereoConstantSpaceBP> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createStereoConstantSpaceBP(ndisp, iters, levels, nr_plane, msg_type);
    *returnValue = new cv::Ptr<cv::cuda::StereoConstantSpaceBP>(ptr);
    END_WRAP
}

// ---------- cuda_StereoConstantSpaceBP_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoConstantSpaceBP_get(
    cv::Ptr<cv::cuda::StereoConstantSpaceBP> *ptr, cv::cuda::StereoConstantSpaceBP **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_StereoConstantSpaceBP_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_StereoConstantSpaceBP_delete(cv::Ptr<cv::cuda::StereoConstantSpaceBP> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}
