using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace OpenCvSharp.Internal;

// ReSharper disable InconsistentNaming

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_Filter_get(IntPtr ptr, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_Filter_delete(IntPtr ptr);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_Filter_apply(IntPtr obj, IntPtr src, IntPtr dst, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createBoxFilter(int srcType, int dstType, Size ksize, Point anchor, int borderMode, Scalar borderVal, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createBoxMaxFilter(int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createBoxMinFilter(int srcType, Size ksize, Point anchor, int borderMode, Scalar borderVal, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createColumnSumFilter(int srcType, int dstType, int ksize, int anchor, int borderMode, Scalar borderVal, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createDerivFilter(int srcType, int dstType, int dx, int dy, int ksize, int normalize, double scale, int rowBorderMode, int columnBorderMode, out IntPtr returnValue);
}
