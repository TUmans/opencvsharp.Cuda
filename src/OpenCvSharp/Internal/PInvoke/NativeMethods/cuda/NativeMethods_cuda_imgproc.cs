using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using OpenCvSharp.Modules.core.Enum;

namespace OpenCvSharp.Internal;

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_alphaComp(
    IntPtr img1, IntPtr img2, IntPtr dst, AlphaCompTypes alpha_op, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_bilateralFilter(
        IntPtr src, IntPtr dst, int kernel_size, float sigma_color, float sigma_spatial, int borderMode, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_blendLinear(IntPtr img1, IntPtr img2, IntPtr weights1, IntPtr weights2, IntPtr result, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createCannyEdgeDetector( double lowThresh, double highThresh, int appertureSize, int l2Gradient, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CannyEdgeDetector_get(IntPtr ptr, out IntPtr ReturnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CannyEdgeDetector_delete(IntPtr ptr);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CannyEdgeDetector_detect(IntPtr obj, IntPtr image, IntPtr edges, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CannyEdgeDetector_detect_dxdy(IntPtr obj, IntPtr dx, IntPtr dy, IntPtr edges, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createCLAHE( double clipLimit, Size tileGridSize, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CLAHE_get(IntPtr ptr, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CLAHE_delete(IntPtr ptr);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_CLAHE_apply(IntPtr obj, IntPtr src, IntPtr dst, IntPtr stream);
}
