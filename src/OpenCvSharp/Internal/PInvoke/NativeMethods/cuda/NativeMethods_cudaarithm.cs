using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace OpenCvSharp.Internal;

// ReSharper disable InconsistentNaming

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_abs(IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_absdiff(IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_add(
        IntPtr src1, IntPtr src2, IntPtr dst,
        IntPtr mask, int dtype, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_addWeighted(
        IntPtr src1, double alpha,
        IntPtr src2, double beta,
        double gamma, IntPtr dst,
        int dtype, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_bitwise_and(
        IntPtr src1, IntPtr src2, IntPtr dst,
        IntPtr mask, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_bitwise_not(
        IntPtr src, IntPtr dst,
        IntPtr mask, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_bitwise_or(
        IntPtr src1, IntPtr src2, IntPtr dst,
        IntPtr mask, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_bitwise_xor(
        IntPtr src1, IntPtr src2, IntPtr dst,
        IntPtr mask, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_cartToPolar(
        IntPtr x, IntPtr y,
        IntPtr magnitude, IntPtr angle,
        int angleInDegrees, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_compare(
        IntPtr src1, IntPtr src2, IntPtr dst,
        int cmpop, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_divide(
        IntPtr src1, IntPtr src2, IntPtr dst,
        double scale, int dtype, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_exp(IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_log(IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_lshift(
        IntPtr src, Vec4i val, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_magnitude_1(
        IntPtr xy, IntPtr magnitude, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_magnitude_2(
        IntPtr x, IntPtr y, IntPtr magnitude, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_magnitudeSqr_1(
        IntPtr xy, IntPtr magnitude, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_magnitudeSqr_2(
        IntPtr x, IntPtr y, IntPtr magnitude, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_max(
        IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_min(
        IntPtr src1, IntPtr src2, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_multiply(
        IntPtr src1, IntPtr src2, IntPtr dst,
        double scale, int dtype, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_phase(
        IntPtr x, IntPtr y, IntPtr angle,
        int angleInDegrees, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_polarToCart(
        IntPtr magnitude, IntPtr angle,
        IntPtr x, IntPtr y,
        int angleInDegrees, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_pow(
        IntPtr src, double power, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_rshift(
        IntPtr src, Vec4i val, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_scaleAdd(
        IntPtr src1, double alpha, IntPtr src2, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_sqr(IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_sqrt(IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_subtract(
        IntPtr src1, IntPtr src2, IntPtr dst,
        IntPtr mask, int dtype, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_threshold(
        IntPtr src, IntPtr dst,
        double thresh, double maxval,
        int type, IntPtr stream,
        out double retVal);
}


