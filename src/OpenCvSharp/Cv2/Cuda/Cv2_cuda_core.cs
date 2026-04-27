using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using OpenCvSharp.Internal;
using OpenCvSharp.Modules.core.Enum;

namespace OpenCvSharp;

public static partial class Cv2
{
    public static partial class Cuda
    {
        /// <summary>
        /// Converts an array to half precision floating number.
        /// </summary>
        /// <remarks>
        /// OpenCV CUDA does NOT consistently use CV_16F for convertFp16 outputs across bindings
        /// </remarks>
        public static void ConvertFp16(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_convertFp16(src.CvPtr, dst.CvPtr, ToPtr(stream)));

            GC.KeepAlive(src);
            dst.Fix();
        }
    }
}
