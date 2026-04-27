#pragma once

// -----------------------------------------------------------------------
// OpenCvSharpExtern – cv::cuda arithmetic wrappers
// These are the C-linkage functions that the C# P/Invoke layer calls.
// Each function catches cv::Exception, stores it, and returns an
// ExceptionStatus so managed code can rethrow it as a .NET exception.
// -----------------------------------------------------------------------

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>

// ---------- abs ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_abs(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::abs(*src, *dst, streamRef);
    END_WRAP
}

// ---------- absdiff ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_absdiff(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::absdiff(*src1, *src2, *dst, streamRef);
    END_WRAP
}

// ---------- absdiff ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_absdiffWithScalar(cv::_InputArray *src1, cv::Scalar src2, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::absdiff(*src1, src2, *dst, streamRef);
    END_WRAP
}

// ---------- absSum ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_absSum(cv::_InputArray *src, cv::_InputArray *mask,  cv::Scalar *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::absSum(*src, mask ? *mask : cv::noArray());
    END_WRAP
}

// ---------- add ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_add(cv::_InputArray *src1, cv::_InputArray *src2, cv::_OutputArray *dst, cv::_InputArray *mask, int dtype, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::add(*src1, *src2, *dst, entity(mask), dtype, streamRef);
    END_WRAP
}

// ---------- addWeighted --------------------------------------------------
CVAPI(ExceptionStatus) cuda_addWeighted(cv::_InputArray *src1, double alpha,cv::_InputArray *src2, double beta, double gamma,cv::_OutputArray *dst, int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::addWeighted(*src1, alpha, *src2, beta, gamma, *dst, dtype, streamRef);
    END_WRAP
}

// ---------- Add with Scalar --------------------------------------------------
CVAPI(ExceptionStatus) cuda_addWithScalar(cv::_InputArray *src1, cv::Scalar src2, cv::_OutputArray *dst, cv::_InputArray *mask, int dtype, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::add(*src1, src2, *dst, mask ? *mask : cv::noArray(), dtype, streamRef);
    END_WRAP
}


// ---------- bitwise_and --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_and(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_and(*src1, *src2, *dst, entity(mask), streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_bitwise_and_with_scalar(cv::_InputArray *src1, cv::Scalar src2, cv::_OutputArray *dst, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_and(*src1, src2, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- bitwise_not --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_not(cv::_InputArray *src,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_not(*src, *dst, entity(mask), streamRef);
    END_WRAP
}

// ---------- bitwise_or ---------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_or(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_or(*src1, *src2, *dst, entity(mask), streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_bitwise_or_with_scalar(cv::_InputArray *src1, cv::Scalar src2, cv::_OutputArray *dst,  cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_or(*src1, src2, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- bitwise_xor --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_xor(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_xor(*src1, *src2, *dst, entity(mask), streamRef);
    END_WRAP
}

CVAPI(ExceptionStatus) cuda_bitwise_xor_with_scalar(cv::_InputArray *src1, cv::Scalar src2, cv::_OutputArray *dst,  cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::bitwise_xor(*src1, src2, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- calcAbsSum --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcAbsSum(
    cv::_InputArray *src, cv::_OutputArray *dst, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcAbsSum(*src, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- calcSqrSum --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcSqrSum(
    cv::_InputArray *src, cv::_OutputArray *dst, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcSqrSum(*src, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- calcSum --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcSum(
    cv::_InputArray *src, cv::_OutputArray *dst, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcSum(*src, *dst, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- calcNorm --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcNorm(
    cv::_InputArray *src, cv::_OutputArray *dst, int normType, cv::_InputArray *mask, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcNorm(*src, *dst, normType, mask ? *mask : cv::noArray(), streamRef);
    END_WRAP
}

// ---------- calcNormDiff --------------------------------------------------
CVAPI(ExceptionStatus) cuda_calcNormDiff(
    cv::_InputArray *src1, cv::_InputArray *src2, cv::_OutputArray *dst, int normType, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::calcNormDiff(*src1, *src2, *dst, normType, streamRef);
    END_WRAP
}

// ---------- cartToPolar --------------------------------------------------
CVAPI(ExceptionStatus) cuda_cartToPolar(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::_OutputArray *angle,    int angleInDegrees,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::cartToPolar(*x, *y, *magnitude, *angle, angleInDegrees != 0, streamRef);
    END_WRAP
}

// ---------- compare ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_compare(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst, int cmpop,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::compare(*src1, *src2, *dst, cmpop, streamRef);
    END_WRAP
}

// ---------- copyMakeborder ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_copyMakeBorder(
    cv::_InputArray *src, cv::_OutputArray *dst, int top, int bottom, int left, int right, int borderType, cv::Scalar value, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::copyMakeBorder(*src, *dst, top, bottom, left, right, borderType, value, streamRef);
    END_WRAP
}
// ---------- copyMakeborder ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_countNonZero_int( cv::_InputArray *src, int *returnValue)
{
    BEGIN_WRAP
    *returnValue = cv::cuda::countNonZero(*src);
    END_WRAP
}

// ---------- copyMakeborder ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_countNonZero_dst( cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::countNonZero(*src, *dst, streamRef);
    END_WRAP
}
// ---------- divide -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_divide(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,    double scale,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::divide(*src1, *src2, *dst, scale, dtype, streamRef);
    END_WRAP
}

// ---------- exp ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_exp(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::exp(*src, *dst, streamRef);
    END_WRAP
}

// ---------- log ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_log(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::log(*src, *dst, streamRef);
    END_WRAP
}

// ---------- lshift -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_lshift(cv::_InputArray *src, cv::Vec4i val, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    // Convert the Vec4i to Scalar_<int> to satisfy OpenCV's function signature
    cv::Scalar_<int> scalarVal(val[0], val[1], val[2], val[3]);
    cv::cuda::lshift(*src, scalarVal, *dst, streamRef);
    END_WRAP
}

// ---------- magnitude (complex form) ------------------------------------
CVAPI(ExceptionStatus) cuda_magnitude_1(cv::_InputArray *xy,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::magnitude(*xy, *magnitude, streamRef);
    END_WRAP
}

// ---------- magnitude (separate x/y) ------------------------------------
CVAPI(ExceptionStatus) cuda_magnitude_2(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::magnitude(*x, *y, *magnitude, streamRef);
    END_WRAP
}

// ---------- magnitudeSqr (complex form) ---------------------------------
CVAPI(ExceptionStatus) cuda_magnitudeSqr_1(cv::_InputArray *xy,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::magnitudeSqr(*xy, *magnitude, streamRef);
    END_WRAP
}

// ---------- magnitudeSqr (separate x/y) ---------------------------------
CVAPI(ExceptionStatus) cuda_magnitudeSqr_2(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::magnitudeSqr(*x, *y, *magnitude, streamRef);
    END_WRAP
}

// ---------- max ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_max(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::max(*src1, *src2, *dst, streamRef);
    END_WRAP
}

// ---------- min ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_min(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::min(*src1, *src2, *dst, streamRef);
    END_WRAP
}

// ---------- multiply -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_multiply(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,    double scale,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::multiply(*src1, *src2, *dst, scale, dtype, streamRef);
    END_WRAP
}

// ---------- phase --------------------------------------------------------
CVAPI(ExceptionStatus) cuda_phase(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *angle,    int angleInDegrees,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::phase(*x, *y, *angle, angleInDegrees != 0, streamRef);
    END_WRAP
}

// ---------- polarToCart --------------------------------------------------
CVAPI(ExceptionStatus) cuda_polarToCart(cv::_InputArray *magnitude, cv::_InputArray *angle, cv::_OutputArray *x, cv::_OutputArray *y, int angleInDegrees, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::polarToCart(*magnitude, *angle, *x, *y, angleInDegrees != 0, streamRef);
    END_WRAP
}

// ---------- pow ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_pow( cv::_InputArray *src, double power,  cv::_OutputArray *dst,  cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::pow(*src, power, *dst, streamRef);
    END_WRAP
}

// ---------- rshift -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_rshift(cv::_InputArray *src, cv::Vec4i val, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    // Convert the Vec4i to Scalar_<int> to satisfy OpenCV's function signature
    cv::Scalar_<int> scalarVal(val[0], val[1], val[2], val[3]);
    cv::cuda::rshift(*src, scalarVal, *dst, streamRef);

    END_WRAP
}

// ---------- scaleAdd -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_scaleAdd(cv::_InputArray *src1,    double alpha,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::scaleAdd(*src1, alpha, *src2, *dst, streamRef);
    END_WRAP
}

// ---------- sqr ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_sqr(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::sqr(*src, *dst, streamRef);
    END_WRAP
}

// ---------- sqrt ---------------------------------------------------------
CVAPI(ExceptionStatus) cuda_sqrt(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::sqrt(*src, *dst, streamRef);
    END_WRAP
}

// ---------- subtract -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_subtract(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    cv::cuda::subtract(*src1, *src2, *dst, entity(mask), dtype, streamRef);
    END_WRAP
}

// ---------- threshold ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_threshold(cv::_InputArray *src,cv::_OutputArray *dst,    double thresh,    double maxval,    int type,cv::cuda::Stream *stream,    double *retVal)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    *retVal = cv::cuda::threshold(*src, *dst, thresh, maxval, type, streamRef);
    END_WRAP
}

// ---------- createConvolution ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_createConvolution(cv::Size user_block_size, cv::Ptr<cv::cuda::Convolution> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createConvolution(user_block_size);
    *returnValue = new cv::Ptr<cv::cuda::Convolution>(ptr);
    END_WRAP
}

// ---------- Convolution_get ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_Convolution_get(cv::Ptr<cv::cuda::Convolution> *ptr, cv::cuda::Convolution **returnValue)
{
    BEGIN_WRAP
    *returnValue= ptr->get();
    END_WRAP
}

// ---------- Convolution_delete ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_Convolution_delete(cv::Ptr<cv::cuda::Convolution> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- Convolution_convolve ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_Convolution_convolve(cv::cuda::Convolution *obj, cv::_InputArray *image, cv::_InputArray *templ, cv::_OutputArray *result, int conj, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->convolve(*image, *templ, *result, conj != 0, streamRef);
    END_WRAP
}

// ---------- createDFT ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_createDFT(
    cv::Size dft_size, int flags, cv::Ptr<cv::cuda::DFT> **returnValue)
{
    BEGIN_WRAP
    auto ptr = cv::cuda::createDFT(dft_size, flags);
    *returnValue = new cv::Ptr<cv::cuda::DFT>(ptr);
    END_WRAP
}

// ---------- DFT_get ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_DFT_get(cv::Ptr<cv::cuda::DFT> *ptr, cv::cuda::DFT **returnValue)
{
    BEGIN_WRAP
    *returnValue = ptr->get();
    END_WRAP
}

// ---------- FT_delete ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_DFT_delete(cv::Ptr<cv::cuda::DFT> *ptr)
{
    BEGIN_WRAP
    delete ptr;
    END_WRAP
}

// ---------- DFT_compute ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_DFT_compute(
    cv::cuda::DFT *obj, cv::_InputArray *src, cv::_OutputArray *dst, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::Stream &streamRef = stream ? *stream : cv::cuda::Stream::Null();
    obj->compute(*src, *dst, streamRef);
    END_WRAP
}
