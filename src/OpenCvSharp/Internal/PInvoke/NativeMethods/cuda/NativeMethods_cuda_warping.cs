using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using OpenCvSharp.Modules.core.Enum;

namespace OpenCvSharp.Internal;

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_buildWarpAffineMaps(IntPtr M, int inverse, Size dsize, IntPtr xmap, IntPtr ymap, IntPtr stream);

    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_buildWarpPerspectiveMaps(IntPtr M, int inverse, Size dsize, IntPtr xmap, IntPtr ymap, IntPtr stream);
}
