using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Internal;

namespace OpenCvSharp;

public static partial class Cv2
{
    public static partial class Cuda
    {
        /// <summary>
        /// Builds transformation maps for affine transformation. 
        /// </summary>
        public static void BuildWarpAffineMaps(OpenCvSharp.Cuda.InputArray src, bool inverse, Size dsize, OpenCvSharp.Cuda.OutputArray xmap, OpenCvSharp.Cuda.OutputArray ymap, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (xmap is null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap is null)
                throw new ArgumentNullException(nameof(ymap));
            src.ThrowIfDisposed();
            xmap.ThrowIfNotReady();
            ymap.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_buildWarpAffineMaps(src.CvPtr, inverse ? 1 : 0, dsize, xmap.CvPtr, ymap.CvPtr, ToPtr(stream)));

            GC.KeepAlive(src);
            xmap.Fix();
            ymap.Fix();
        }

        /// <summary>
        /// Builds transformation maps for affine transformation. 
        /// </summary>
        public static void BuildWarpAffineMaps(InputArray src, bool inverse, Size dsize, OpenCvSharp.Cuda.OutputArray xmap, OpenCvSharp.Cuda.OutputArray ymap, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (xmap is null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap is null)
                throw new ArgumentNullException(nameof(ymap));
            src.ThrowIfDisposed();
            xmap.ThrowIfNotReady();
            ymap.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_buildWarpAffineMaps(src.CvPtr, inverse ? 1 : 0, dsize, xmap.CvPtr, ymap.CvPtr, ToPtr(stream)));

            GC.KeepAlive(src);
            xmap.Fix();
            ymap.Fix();
        }

        /// <summary>
        /// Builds transformation maps for perspective transformation. 
        /// </summary>
        public static void BuildWarpPerspectiveMaps(InputArray src, bool inverse, Size dsize, OpenCvSharp.Cuda.OutputArray xmap, OpenCvSharp.Cuda.OutputArray ymap, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (xmap is null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap is null)
                throw new ArgumentNullException(nameof(ymap));
            src.ThrowIfDisposed();
            xmap.ThrowIfNotReady();
            ymap.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_buildWarpPerspectiveMaps(src.CvPtr, inverse ? 1 : 0, dsize, xmap.CvPtr, ymap.CvPtr, ToPtr(stream)));

            GC.KeepAlive(src);
            xmap.Fix();
            ymap.Fix();
        }

        /// <summary>
        /// Builds transformation maps for perspective transformation. 
        /// </summary>
        public static void BuildWarpPerspectiveMaps(OpenCvSharp.Cuda.InputArray src, bool inverse, Size dsize, OpenCvSharp.Cuda.OutputArray xmap, OpenCvSharp.Cuda.OutputArray ymap, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (xmap is null)
                throw new ArgumentNullException(nameof(xmap));
            if (ymap is null)
                throw new ArgumentNullException(nameof(ymap));
            src.ThrowIfDisposed();
            xmap.ThrowIfNotReady();
            ymap.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_buildWarpPerspectiveMaps(src.CvPtr, inverse ? 1 : 0, dsize, xmap.CvPtr, ymap.CvPtr, ToPtr(stream)));
            
            GC.KeepAlive(src);
            xmap.Fix();
            ymap.Fix();
        }
    }
}
