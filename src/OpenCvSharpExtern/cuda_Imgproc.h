#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/photo/cuda.hpp>

// ---------- Alpha Composite --------------------------------------------------
CVAPI(ExceptionStatus) cuda_alphaComp(cv::_InputArray *img1, cv::_InputArray *img2, cv::_OutputArray *dst, int alpha_op, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::alphaComp(*img1, *img2, *dst, alpha_op, streamRef);
    END_WRAP
}

// ---------- BilateralFilter --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bilateralFilter( cv::_InputArray *src, cv::_OutputArray *dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bilateralFilter(*src, *dst, kernel_size, sigma_color, sigma_spatial, borderMode, streamRef);
    END_WRAP
}

// ---------- blendLinear --------------------------------------------------
CVAPI(ExceptionStatus) cuda_blendLinear(cv::_InputArray *img1, cv::_InputArray *img2, cv::_InputArray *weights1, cv::_InputArray *weights2, cv::_OutputArray *result, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::blendLinear(*img1, *img2, *weights1, *weights2, *result, streamRef);
    END_WRAP
}

// ---------- calcHist --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcHist(cv::_InputArray *src, cv::_InputArray *mask, cv::_OutputArray *hist, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcHist(*src, mask ? *mask : cv::noArray(), *hist, streamRef);
    END_WRAP
}

// ---------- createCannyEdgeDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createCannyEdgeDetector(double low_thresh, double high_thresh, int apperture_size, int L2gradient,  cv::Ptr<cv::cuda::CannyEdgeDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createCannyEdgeDetector(low_thresh, high_thresh, apperture_size, L2gradient != 0);
    *returnValue = new cv::Ptr<cv::cuda::CannyEdgeDetector>(ptr);
    END_WRAP
}

// ---------- CannyEdgeDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_get(cv::Ptr<cv::cuda::CannyEdgeDetector> *ptr, cv::cuda::CannyEdgeDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- CannyEdgeDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_delete(cv::Ptr<cv::cuda::CannyEdgeDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- CannyEdgeDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_detect(cv::cuda::CannyEdgeDetector *obj, cv::_InputArray *image, cv::_OutputArray *edges, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*image, *edges, streamRef);
    END_WRAP
}

// ---------- CannyEdgeDetector_detect_dxdy --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CannyEdgeDetector_detect_dxdy(cv::cuda::CannyEdgeDetector *obj, cv::_InputArray *dx, cv::_InputArray *dy, cv::_OutputArray *edges, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*dx, *dy, *edges, streamRef);
    END_WRAP
}

// ---------- createCLAHE --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createCLAHE( double clipLimit, cv::Size tileGridSize, cv::Ptr<cv::cuda::CLAHE> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createCLAHE(clipLimit, tileGridSize);
    *returnValue = new cv::Ptr<cv::cuda::CLAHE>(ptr);
    END_WRAP
}

// ---------- CLAHE_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_get(cv::Ptr<cv::cuda::CLAHE> *ptr, cv::cuda::CLAHE **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- CLAHE_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_delete(cv::Ptr<cv::cuda::CLAHE> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- CLAHE_apply --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CLAHE_apply(cv::cuda::CLAHE *obj, cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->apply(*src, *dst, streamRef);
    END_WRAP
}

// ---------- createGeneralizedHoughBallard --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createGeneralizedHoughBallard(cv::Ptr<cv::GeneralizedHoughBallard> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createGeneralizedHoughBallard();
    *returnValue = new cv::Ptr<cv::GeneralizedHoughBallard>(ptr);
    END_WRAP
}

// ---------- createGeneralizedHoughBallard --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createGeneralizedHoughGuil(cv::Ptr<cv::GeneralizedHoughGuil> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createGeneralizedHoughGuil();
    *returnValue = new cv::Ptr<cv::GeneralizedHoughGuil>(ptr);
    END_WRAP
}

// ---------- createGoodFeaturesToTrackDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createGoodFeaturesToTrackDetector(int srcType, int maxCorners, double qualityLevel, double minDistance, int blockSize, int useHarrisDetector, double harrisK, cv::Ptr<cv::cuda::CornersDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createGoodFeaturesToTrackDetector(srcType, maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector != 0, harrisK);
    *returnValue = new cv::Ptr<cv::cuda::CornersDetector>(ptr);
    END_WRAP
}

// ---------- CornersDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornersDetector_get( cv::Ptr<cv::cuda::CornersDetector> *ptr, cv::cuda::CornersDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- CornersDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornersDetector_delete(cv::Ptr<cv::cuda::CornersDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- CornersDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornersDetector_detect( cv::cuda::CornersDetector *obj, cv::_InputArray *image, cv::_OutputArray *corners, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*image, *corners, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- cuda_createHarrisCorner --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHarrisCorner(int srcType, int blockSize, int ksize, double k, int borderType,cv::Ptr<cv::cuda::CornernessCriteria> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHarrisCorner(srcType, blockSize, ksize, k, borderType);
    *returnValue = new cv::Ptr<cv::cuda::CornernessCriteria>(ptr);
    END_WRAP
}

// ---------- cuda_CornernessCriteria_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornernessCriteria_get( cv::Ptr<cv::cuda::CornernessCriteria> *ptr, cv::cuda::CornernessCriteria **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_CornernessCriteria_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornernessCriteria_delete(cv::Ptr<cv::cuda::CornernessCriteria> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_CornernessCriteria_compute --------------------------------------------------
CVAPI(ExceptionStatus) cuda_CornernessCriteria_compute( cv::cuda::CornernessCriteria *obj, cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->compute(*src, *dst, streamRef);
    END_WRAP
}

// ---------- cuda_createHoughCirclesDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles, cv::Ptr<cv::cuda::HoughCirclesDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHoughCirclesDetector(dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles);
    *returnValue = new cv::Ptr<cv::cuda::HoughCirclesDetector>(ptr);
    END_WRAP
}

// ---------- cuda_HoughCirclesDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughCirclesDetector_get( cv::Ptr<cv::cuda::HoughCirclesDetector> *ptr, cv::cuda::HoughCirclesDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_HoughCirclesDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughCirclesDetector_delete(cv::Ptr<cv::cuda::HoughCirclesDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_HoughCirclesDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughCirclesDetector_detect(cv::cuda::HoughCirclesDetector *obj, cv::_InputArray *src, cv::_OutputArray *circles, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*src, *circles, streamRef);
    END_WRAP
}

// ---------- cuda_createHoughLinesDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHoughLinesDetector(float rho, float theta, int threshold, int doSort, int maxLines, cv::Ptr<cv::cuda::HoughLinesDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHoughLinesDetector(rho, theta, threshold, doSort != 0, maxLines);
    *returnValue = new cv::Ptr<cv::cuda::HoughLinesDetector>(ptr);
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_get( cv::Ptr<cv::cuda::HoughLinesDetector> *ptr, cv::cuda::HoughLinesDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_delete(cv::Ptr<cv::cuda::HoughLinesDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_HoughLinesDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughLinesDetector_detect(cv::cuda::HoughLinesDetector *obj, cv::_InputArray *src, cv::_OutputArray *lines, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*src, *lines, streamRef);
    END_WRAP
}

// ---------- cuda_createHoughSegmentDetector --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createHoughSegmentDetector(float rho, float theta, int minLineLength, int maxLineGap, int maxLines, cv::Ptr<cv::cuda::HoughSegmentDetector> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createHoughSegmentDetector(rho, theta, minLineLength, maxLineGap, maxLines);
    *returnValue = new cv::Ptr<cv::cuda::HoughSegmentDetector>(ptr);
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_get(cv::Ptr<cv::cuda::HoughSegmentDetector> *ptr, cv::cuda::HoughSegmentDetector **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_delete(cv::Ptr<cv::cuda::HoughSegmentDetector> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_HoughSegmentDetector_detect --------------------------------------------------
CVAPI(ExceptionStatus) cuda_HoughSegmentDetector_detect(cv::cuda::HoughSegmentDetector *obj, cv::_InputArray *src, cv::_OutputArray *lines, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->detect(*src, *lines, streamRef);
    END_WRAP
}

// ---------- cuda_createMinEigenValCorner --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createMinEigenValCorner(int srcType, int blockSize, int ksize, int borderType, cv::Ptr<cv::cuda::CornernessCriteria> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createMinEigenValCorner(srcType, blockSize, ksize, borderType);
    *returnValue = new cv::Ptr<cv::cuda::CornernessCriteria>(ptr);
    END_WRAP
}

// ---------- cuda_createTemplateMatching --------------------------------------------------
CVAPI(ExceptionStatus) cuda_createTemplateMatching(int srcType, int method, cv::Size user_block_size, cv::Ptr<cv::cuda::TemplateMatching> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createTemplateMatching(srcType, method, user_block_size);
    *returnValue = new cv::Ptr<cv::cuda::TemplateMatching>(ptr);
    END_WRAP
}

// ---------- cuda_TemplateMatching_get --------------------------------------------------
CVAPI(ExceptionStatus) cuda_TemplateMatching_get(
    cv::Ptr<cv::cuda::TemplateMatching> *ptr, cv::cuda::TemplateMatching **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- cuda_TemplateMatching_delete --------------------------------------------------
CVAPI(ExceptionStatus) cuda_TemplateMatching_delete(cv::Ptr<cv::cuda::TemplateMatching> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- cuda_TemplateMatching_match --------------------------------------------------
CVAPI(ExceptionStatus) cuda_TemplateMatching_match(cv::cuda::TemplateMatching *obj, cv::_InputArray *image, cv::_InputArray *templ, cv::_OutputArray *result, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->match(*image, *templ, *result, streamRef);
    END_WRAP
}

// ---------- cuda_TemplateMatching_match --------------------------------------------------
CVAPI(ExceptionStatus) cuda_cvtColor(cv::_InputArray *src, cv::_OutputArray *dst, int code, int dcn, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::cvtColor(*src, *dst, code, dcn, streamRef);
    END_WRAP
}

// ---------- cuda_demosaicing --------------------------------------------------
CVAPI(ExceptionStatus) cuda_demosaicing(cv::_InputArray *src, cv::_OutputArray *dst, int code, int dcn, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::demosaicing(*src, *dst, code, dcn, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_equalizeHist(cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::equalizeHist(*src, *dst, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_evenLevels(cv::_OutputArray *levels, int nLevels, int lowerLevel, int upperLevel, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::evenLevels(*levels, nLevels, lowerLevel, upperLevel, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_fastNlMeansDenoising(cv::_InputArray *src, cv::_OutputArray *dst, float h, int search_window, int block_size, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::fastNlMeansDenoising(*src, *dst, h, search_window, block_size, streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_gammaCorrection(cv::_InputArray *src, cv::_OutputArray *dst, int forward, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::gammaCorrection(*src, *dst, forward != 0, streamRef);
    END_WRAP
}
