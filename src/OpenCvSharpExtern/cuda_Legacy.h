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


// ---------- calcOpticalFlowBM --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcOpticalFlowBM(
    cv::cuda::GpuMat *prev, cv::cuda::GpuMat *curr, cv::Size block_size, cv::Size shift_size, cv::Size max_range,
    int use_previous, cv::cuda::GpuMat *velx, cv::cuda::GpuMat *vely, cv::cuda::GpuMat *buf, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcOpticalFlowBM(*prev, *curr, block_size, shift_size, max_range, use_previous != 0,
                                *velx, *vely, *buf, streamRef);
    END_WRAP
}

// ---------- connectivityMask --------------------------------------------------
CVAPI(ExceptionStatus) cuda_connectivityMask(cv::cuda::GpuMat *image, cv::cuda::GpuMat *mask, cv::Scalar lo, cv::Scalar hi, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::connectivityMask(*image, entity(mask), lo, hi, streamRef);
    END_WRAP
}

// ---------- createBackgroundSubtractorGMG --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createBackgroundSubtractorGMG(int initializationFrames, double decisionThreshold,  cv::Ptr<cv::cuda::BackgroundSubtractorGMG> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createBackgroundSubtractorGMG(initializationFrames, decisionThreshold);
    *returnValue = new cv::Ptr<cv::cuda::BackgroundSubtractorGMG>(ptr);
    END_WRAP
}

// ---------- BackgroundSubtractorGMG_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorGMG_get(cv::Ptr<cv::cuda::BackgroundSubtractorGMG> *ptr, cv::cuda::BackgroundSubtractorGMG **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- BackgroundSubtractorGMG_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorGMG_delete(cv::Ptr<cv::cuda::BackgroundSubtractorGMG> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- BackgroundSubtractorGMG_apply --------------------------------------------------
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorGMG_apply(cv::cuda::BackgroundSubtractorGMG *obj, cv::_InputArray *image, cv::_OutputArray *fgmask, double learningRate, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->apply(*image, *fgmask, learningRate, streamRef);
    END_WRAP
}

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
CVAPI(ExceptionStatus) cuda_BackgroundSubtractorFGD_delete( cv::Ptr<cv::cuda::BackgroundSubtractorFGD> *ptr)
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

// ---------- cuda_createImagePyramid --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createImagePyramid(
    cv::_InputArray *img, int nLayers, cv::cuda::Stream *stream,
    cv::Ptr<cv::cuda::ImagePyramid> **returnValue)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    auto ptr = cv::cuda::createImagePyramid(*img, nLayers, streamRef);
    *returnValue = new cv::Ptr<cv::cuda::ImagePyramid>(ptr);
    END_WRAP
}

// ---------- cuda_ImagePyramid_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_ImagePyramid_get(cv::Ptr<cv::cuda::ImagePyramid> *ptr, cv::cuda::ImagePyramid **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_ImagePyramid_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_ImagePyramid_delete(cv::Ptr<cv::cuda::ImagePyramid> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_ImagePyramid_getLayer --------------------------------------------------
CVAPI(ExceptionStatus) cuda_ImagePyramid_getLayer(cv::cuda::ImagePyramid *obj, cv::_OutputArray *outImg, cv::Size dsize, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->getLayer(*outImg, dsize, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_createOpticalFlowNeedleMap(cv::cuda::GpuMat *u, cv::cuda::GpuMat *v, cv::cuda::GpuMat *vertex, cv::cuda::GpuMat *colors)
{
    BEGIN_WRAP
    cv::cuda::createOpticalFlowNeedleMap(*u, *v, *vertex, *colors);
    END_WRAP
}
