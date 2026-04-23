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
    cv::cuda::abs(*src, *dst, *stream);
    END_WRAP
}

// ---------- absdiff ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_absdiff(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::absdiff(*src1, *src2, *dst, *stream);
    END_WRAP
}

// ---------- add ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_add(cv::_InputArray *src1, cv::_InputArray *src2, cv::_OutputArray *dst, cv::_InputArray *mask, int dtype, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::add(*src1, *src2, *dst, entity(mask), dtype, *stream);
    END_WRAP
}

// ---------- addWeighted --------------------------------------------------
CVAPI(ExceptionStatus) cuda_addWeighted(cv::_InputArray *src1, double alpha,cv::_InputArray *src2, double beta, double gamma,cv::_OutputArray *dst, int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::addWeighted(*src1, alpha, *src2, beta, gamma, *dst, dtype, *stream);
    END_WRAP
}

// ---------- bitwise_and --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_and(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::bitwise_and(*src1, *src2, *dst, entity(mask), *stream);
    END_WRAP
}

// ---------- bitwise_not --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_not(cv::_InputArray *src,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::bitwise_not(*src, *dst, entity(mask), *stream);
    END_WRAP
}

// ---------- bitwise_or ---------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_or(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::bitwise_or(*src1, *src2, *dst, entity(mask), *stream);
    END_WRAP
}

// ---------- bitwise_xor --------------------------------------------------
CVAPI(ExceptionStatus) cuda_bitwise_xor(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::bitwise_xor(*src1, *src2, *dst, entity(mask), *stream);
    END_WRAP
}

// ---------- cartToPolar --------------------------------------------------
CVAPI(ExceptionStatus) cuda_cartToPolar(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::_OutputArray *angle,    int angleInDegrees,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::cartToPolar(*x, *y, *magnitude, *angle, angleInDegrees != 0, *stream);
    END_WRAP
}

// ---------- compare ------------------------------------------------------
CVAPI(ExceptionStatus) cuda_compare(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst, int cmpop,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::compare(*src1, *src2, *dst, cmpop, *stream);
    END_WRAP
}

// ---------- divide -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_divide(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,    double scale,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::divide(*src1, *src2, *dst, scale, dtype, *stream);
    END_WRAP
}

// ---------- exp ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_exp(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::exp(*src, *dst, *stream);
    END_WRAP
}

// ---------- log ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_log(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::log(*src, *dst, *stream);
    END_WRAP
}

// ---------- lshift -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_lshift(cv::_InputArray *src,    MyCvScalar val,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::lshift(*src, cpp(val), *dst, *stream);
    END_WRAP
}

// ---------- magnitude (complex form) ------------------------------------
CVAPI(ExceptionStatus) cuda_magnitude_1(cv::_InputArray *xy,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::magnitude(*xy, *magnitude, *stream);
    END_WRAP
}

// ---------- magnitude (separate x/y) ------------------------------------
CVAPI(ExceptionStatus) cuda_magnitude_2(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::magnitude(*x, *y, *magnitude, *stream);
    END_WRAP
}

// ---------- magnitudeSqr (complex form) ---------------------------------
CVAPI(ExceptionStatus) cuda_magnitudeSqr_1(cv::_InputArray *xy,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::magnitudeSqr(*xy, *magnitude, *stream);
    END_WRAP
}

// ---------- magnitudeSqr (separate x/y) ---------------------------------
CVAPI(ExceptionStatus) cuda_magnitudeSqr_2(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *magnitude,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::magnitudeSqr(*x, *y, *magnitude, *stream);
    END_WRAP
}

// ---------- max ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_max(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::max(*src1, *src2, *dst, *stream);
    END_WRAP
}

// ---------- min ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_min(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::min(*src1, *src2, *dst, *stream);
    END_WRAP
}

// ---------- multiply -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_multiply(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,    double scale,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::multiply(*src1, *src2, *dst, scale, dtype, *stream);
    END_WRAP
}

// ---------- phase --------------------------------------------------------
CVAPI(ExceptionStatus) cuda_phase(cv::_InputArray *x,cv::_InputArray *y,cv::_OutputArray *angle,    int angleInDegrees,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::phase(*x, *y, *angle, angleInDegrees != 0, *stream);
    END_WRAP
}

// ---------- polarToCart --------------------------------------------------
CVAPI(ExceptionStatus) cuda_polarToCart(cv::_InputArray *magnitude, cv::_InputArray *angle, cv::_OutputArray *x, cv::_OutputArray *y, int angleInDegrees, cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::polarToCart(*magnitude, *angle, *x, *y, angleInDegrees != 0, *stream);
    END_WRAP
}

// ---------- pow ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_pow( cv::_InputArray *src, double power,  cv::_OutputArray *dst,  cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::pow(*src, power, *dst, *stream);
    END_WRAP
}

// ---------- rshift -------------------------------------------------------
CVAPI(ExceptionStatus) cuda_rshift(cv::_InputArray *src, MyCvScalar val, cv::_OutputArray *dst,  cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::rshift(*src, cpp(val), *dst, *stream);
    END_WRAP
}

// ---------- scaleAdd -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_scaleAdd(cv::_InputArray *src1,    double alpha,cv::_InputArray *src2,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::scaleAdd(*src1, alpha, *src2, *dst, *stream);
    END_WRAP
}

// ---------- sqr ----------------------------------------------------------
CVAPI(ExceptionStatus) cuda_sqr(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::sqr(*src, *dst, *stream);
    END_WRAP
}

// ---------- sqrt ---------------------------------------------------------
CVAPI(ExceptionStatus) cuda_sqrt(cv::_InputArray *src,cv::_OutputArray *dst,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::sqrt(*src, *dst, *stream);
    END_WRAP
}

// ---------- subtract -----------------------------------------------------
CVAPI(ExceptionStatus) cuda_subtract(cv::_InputArray *src1,cv::_InputArray *src2,cv::_OutputArray *dst,cv::_InputArray *mask,    int dtype,cv::cuda::Stream *stream)
{
    BEGIN_WRAP
    cv::cuda::subtract(*src1, *src2, *dst, entity(mask), dtype, *stream);
    END_WRAP
}

// ---------- threshold ----------------------------------------------------
CVAPI(ExceptionStatus) cuda_threshold(cv::_InputArray *src,cv::_OutputArray *dst,    double thresh,    double maxval,    int type,cv::cuda::Stream *stream,    double *retVal)
{
    BEGIN_WRAP
    *retVal = cv::cuda::threshold(*src, *dst, thresh, maxval, type, *stream);
    END_WRAP
}
