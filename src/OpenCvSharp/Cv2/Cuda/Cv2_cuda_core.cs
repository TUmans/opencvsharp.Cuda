using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using OpenCvSharp.Cuda;
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

        /// <summary>
        /// Ensures that size of the given matrix is not less than (rows, cols) size
        /// and matrix type is match specified one too
        /// </summary>
        /// <param name="rows">Number of rows in a 2D array.</param>
        /// <param name="cols">Number of columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void EnsureSizeIsEnough(int rows, int cols, MatType type, OpenCvSharp.Cuda.OutputArray m)
        {
            ThrowIfGpuNotAvailable();
            if (m is null)
                throw new ArgumentNullException(nameof(m));
            NativeMethods.cuda_ensureSizeIsEnough(rows, cols, (int)type, m.CvPtr);
            GC.KeepAlive(m);
        }

        /// <summary>
        /// Ensures that size of the given matrix is not less than (rows, cols) size
        /// and matrix type is match specified one too
        /// </summary>
        /// <param name="size">Number of rows and columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void EnsureSizeIsEnough(Size size, MatType type, OpenCvSharp.Cuda.OutputArray m)
        {
            ThrowIfGpuNotAvailable();
            EnsureSizeIsEnough(size.Height, size.Width, type, m);
        }
    }
}
