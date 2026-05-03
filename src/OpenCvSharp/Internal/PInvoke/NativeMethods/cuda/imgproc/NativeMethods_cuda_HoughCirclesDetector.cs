using System;
using System.Runtime.InteropServices;
using System.Text;
using OpenCvSharp.Cuda;

#pragma warning disable 1591

namespace OpenCvSharp.Internal
{
    static partial class NativeMethods
    {
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_createHoughCirclesDetector(float dp, float minDist, int cannyThreshold, int votesThreshold, int minRadius, int maxRadius, int maxCircles, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_HoughCirclesDetector_get(IntPtr ptr, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_HoughCirclesDetector_delete(IntPtr ptr);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_HoughCirclesDetector_detect(IntPtr obj, IntPtr src, IntPtr circles, IntPtr stream);
    }
}
