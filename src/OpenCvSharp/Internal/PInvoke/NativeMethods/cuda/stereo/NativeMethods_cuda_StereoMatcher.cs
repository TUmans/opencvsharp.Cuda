using System.Runtime.InteropServices;

#pragma warning disable 1591

namespace OpenCvSharp.Internal;

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_StereoMatcher_compute(
        IntPtr obj, IntPtr left, IntPtr right, IntPtr disparity, IntPtr stream);
}
