using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace OpenCvSharp.Internal;

// ReSharper disable InconsistentNaming

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createBackgroundSubtractorMOG2( int history, double varThreshold, int detectShadows, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG2_get(IntPtr ptr, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG2_delete(IntPtr ptr);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG2_apply(IntPtr obj, IntPtr image, IntPtr fgmask, double learningRate, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_createBackgroundSubtractorMOG(int history, int nmixtures, double backgroundRatio, double noiseSigma, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG_get(IntPtr ptr, out IntPtr returnValue);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG_delete(IntPtr ptr);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_BackgroundSubtractorMOG_apply(IntPtr obj, IntPtr image, IntPtr fgmask, double learningRate, IntPtr stream);
}
