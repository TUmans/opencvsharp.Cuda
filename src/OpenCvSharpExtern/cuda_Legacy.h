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

CVAPI(ExceptionStatus) cuda_graphcut(cv::cuda::GpuMat *terminals, cv::cuda::GpuMat *leftTransp, cv::cuda::GpuMat *rightTransp, cv::cuda::GpuMat *top, cv::cuda::GpuMat *bottom, cv::cuda::GpuMat *labels, cv::cuda::GpuMat *buf, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::graphcut(*terminals, *leftTransp, *rightTransp, *top, *bottom, *labels, *buf, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_graphcut8(cv::cuda::GpuMat *terminals, cv::cuda::GpuMat *leftTransp, cv::cuda::GpuMat *rightTransp, cv::cuda::GpuMat *top, cv::cuda::GpuMat *topLeft, cv::cuda::GpuMat *topRight, cv::cuda::GpuMat *bottom, cv::cuda::GpuMat *bottomLeft, cv::cuda::GpuMat *bottomRight, cv::cuda::GpuMat *labels, cv::cuda::GpuMat *buf, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::graphcut(*terminals, *leftTransp, *rightTransp, *top, *topLeft, *topRight, *bottom, *bottomLeft, *bottomRight, *labels, *buf, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_interpolateFrames(cv::cuda::GpuMat *frame0, cv::cuda::GpuMat *frame1, cv::cuda::GpuMat *fu, cv::cuda::GpuMat *fv, cv::cuda::GpuMat *bu, cv::cuda::GpuMat *bv, float pos, cv::cuda::GpuMat *newFrame, cv::cuda::GpuMat *buf, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::interpolateFrames(*frame0, *frame1, *fu, *fv, *bu, *bv, pos, *newFrame, *buf, streamRef);
    END_WRAP
}


CVAPI(ExceptionStatus) cuda_labelComponents(cv::cuda::GpuMat *mask, cv::cuda::GpuMat *components, int flags, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::labelComponents(*mask, *components, flags, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_projectPoints(cv::cuda::GpuMat *src, cv::Mat *rvec, cv::Mat *tvec, cv::Mat *camera_mat, cv::Mat *dist_coef, cv::cuda::GpuMat *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::projectPoints(*src, *rvec, *tvec, *camera_mat, *dist_coef, *dst, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_solvePnPRansac(cv::Mat *object, cv::Mat *image, cv::Mat *camera_mat, cv::Mat *dist_coef, cv::Mat *rvec, cv::Mat *tvec, int use_extrinsic_guess, int num_iters, float max_dist, int min_inlier_count, cv::_OutputArray *inliers)
{
    BEGIN_WRAP
    std::vector<int> inliers_vec;

    cv::cuda::solvePnPRansac(*object, *image, *camera_mat, *dist_coef, *rvec, *tvec, use_extrinsic_guess != 0, num_iters, max_dist, min_inlier_count,  inliers ? &inliers_vec : NULL);

    if (inliers && inliers->needed())
    {
        cv::Mat(inliers_vec).copyTo(*inliers);
    }
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_transformPoints(cv::cuda::GpuMat *src, cv::Mat *rvec, cv::Mat *tvec, cv::cuda::GpuMat *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::transformPoints(*src, *rvec, *tvec, *dst, streamRef);
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

// Helper to prevent memory leaks by deleting the array of pointers
// (The actual cv::Mat objects will be deleted by the C# Garbage Collector)
CVAPI(ExceptionStatus) cuda_FreeMatPointerArray(cv::Mat **mats)
{
    BEGIN_WRAP
    delete[] mats;
    END_WRAP
}
