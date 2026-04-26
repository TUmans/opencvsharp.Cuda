using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using OpenCvSharp.Modules.core.Enum;

namespace OpenCvSharp.Internal;

static partial class NativeMethods
{
    [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
    public static extern ExceptionStatus cuda_calcOpticalFlowBM(IntPtr prev, IntPtr curr, Size blockSize, Size shiftSize, Size maxRange, int usePrevious, IntPtr velx, IntPtr vely, IntPtr buf, IntPtr stream);
}
