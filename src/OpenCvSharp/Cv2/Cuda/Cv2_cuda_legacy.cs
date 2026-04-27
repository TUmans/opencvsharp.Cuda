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
        /// Calculates optical flow for 2 images using block matching algorithm.
        /// </summary>
        /// <remarks>
        /// Only works when opencv is build with legacy support
        /// </remarks>
        public static void CalcOpticalFlowBM(
            GpuMat prev, GpuMat curr,
            Size blockSize, Size shiftSize, Size maxRange, bool usePrevious,
            GpuMat velx, GpuMat vely, GpuMat buf, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (prev is null) 
                throw new ArgumentNullException(nameof(prev));
            if (curr is null) 
                throw new ArgumentNullException(nameof(curr));
            if (velx is null) 
                throw new ArgumentNullException(nameof(velx));
            if (vely is null) 
                throw new ArgumentNullException(nameof(vely));
            if (buf is null) 
                throw new ArgumentNullException(nameof(buf));

            prev.ThrowIfDisposed();
            curr.ThrowIfDisposed();
            velx.ThrowIfDisposed();
            vely.ThrowIfDisposed();
            buf.ThrowIfDisposed();

            NativeMethods.HandleException(
                NativeMethods.cuda_calcOpticalFlowBM(
                    prev.CvPtr, curr.CvPtr, blockSize, shiftSize, maxRange,
                    usePrevious ? 1 : 0, velx.CvPtr, vely.CvPtr, buf.CvPtr, ToPtr(stream)));

            GC.KeepAlive(prev);
            GC.KeepAlive(curr);
            GC.KeepAlive(velx);
            GC.KeepAlive(vely);
            GC.KeepAlive(buf);
        }

        /// <summary>
        /// compute mask for Generalized Flood fill componetns labeling. 
        /// </summary>
        /// <remarks>
        /// Only works when opencv is build with legacy support. Use InRange.
        /// </remarks>
        public static void ConnectivityMask(GpuMat image, GpuMat mask, Scalar lo, Scalar hi, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (image is null) 
                throw new ArgumentNullException(nameof(image));
            if (mask is null) 
                throw new ArgumentNullException(nameof(mask));

            image.ThrowIfDisposed();
            mask.ThrowIfDisposed();

            NativeMethods.HandleException(
                NativeMethods.cuda_connectivityMask(image.CvPtr, mask.CvPtr, lo, hi, ToPtr(stream)));

            GC.KeepAlive(image);
            GC.KeepAlive(mask);
        }
    }
}
