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
        /// Composites two images using alpha opacity values contained in each image. 
        /// </summary>
        public static void AlphaComp(OpenCvSharp.Cuda.InputArray img1, OpenCvSharp.Cuda.InputArray img2, OpenCvSharp.Cuda.OutputArray dst, AlphaCompTypes alphaOp, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (img1 is null) 
                throw new ArgumentNullException(nameof(img1));
            if (img2 is null) 
                throw new ArgumentNullException(nameof(img2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            img1.ThrowIfDisposed();
            img2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_alphaComp(img1.CvPtr, img2.CvPtr, dst.CvPtr, alphaOp, stream?.CvPtr ?? IntPtr.Zero));
           
            GC.KeepAlive(img1);
            GC.KeepAlive(img2);
            dst.Fix();
       
        }

        /// <summary>
        /// Performs bilateral filtering of passed image.
        /// </summary>
        public static void BilateralFilter(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,  int kernelSize, float sigmaColor, float sigmaSpatial, BorderTypes borderMode = BorderTypes.Default, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bilateralFilter(
                    src.CvPtr, dst.CvPtr, kernelSize, sigmaColor, sigmaSpatial, (int)borderMode, stream?.CvPtr ?? IntPtr.Zero));
            
            GC.KeepAlive(src);
            dst.Fix();
         
        }

        /// <summary>
        /// Performs linear blending of two images. 
        /// </summary>
        public static void BlendLinear(OpenCvSharp.Cuda.InputArray img1, OpenCvSharp.Cuda.InputArray img2, OpenCvSharp.Cuda.InputArray weights1, OpenCvSharp.Cuda.InputArray weights2, OpenCvSharp.Cuda.OutputArray result, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (img1 is null)
                throw new ArgumentNullException(nameof(img1));
            if (img2 is null)
                throw new ArgumentNullException(nameof(img2));
            if (weights1 is null)
                throw new ArgumentNullException(nameof(weights1));
            if (weights2 is null)
                throw new ArgumentNullException(nameof(weights2));
            img1.ThrowIfDisposed();
            img2.ThrowIfDisposed();
            weights1.ThrowIfDisposed();
            weights2.ThrowIfDisposed();
            result.ThrowIfNotReady();


            NativeMethods.HandleException(
                NativeMethods.cuda_blendLinear(img1.CvPtr, img2.CvPtr, weights1.CvPtr, weights2.CvPtr, result.CvPtr, ToPtr(stream)));
       
            GC.KeepAlive(img1); 
            GC.KeepAlive(img2); 
            GC.KeepAlive(weights1); 
            GC.KeepAlive(weights2);
            result.Fix();
        }

        /// <summary>
        /// Converts an image from one color space to another.
        /// </summary>
        public static void CvtColor( OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, ColorConversionCodes code, int dcn = 0, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_cvtColor(src.CvPtr, dst.CvPtr, (int)code, dcn, ToPtr(stream)));

            GC.KeepAlive(src);
            dst.Fix();
            
        }

        /// <summary>
        /// Converts an image from Bayer pattern to RGB or grayscale.
        /// </summary>
        /// <param name="src">Source image (Bayer pattern, 8-bit or 16-bit single channel).</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="code">Demosaicing code (e.g. ColorConversionCodes.BayerRG2BGR).</param>
        /// <param name="dcn">Number of channels in the destination image. -1 means derived automatically.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void Demosaicing(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, ColorConversionCodes code, int dcn = -1, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_demosaicing(src.CvPtr, dst.CvPtr, (int)code, dcn, ToPtr(stream)));

            GC.KeepAlive(src);
            dst.Fix();
        }

        /// <summary>
        /// Equalizes the histogram of a grayscale image.
        /// </summary>
        /// <param name="src">Source 8-bit single channel image.</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void EqualizeHist(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (src.Type() != MatType.CV_8UC1)
                throw new InvalidDataException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_equalizeHist(src.CvPtr, dst.CvPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            dst.Fix();
            
        }

        /// <summary>
        /// Computes levels with even distribution.
        /// </summary>
        /// <param name="levels">Output 1D matrix of type CV_32SC1.</param>
        /// <param name="nLevels">Number of levels to compute.</param>
        /// <param name="lowerLevel">Lower bound of the levels.</param>
        /// <param name="upperLevel">Upper bound of the levels.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void EvenLevels(OpenCvSharp.Cuda.OutputArray levels, int nLevels, int lowerLevel, int upperLevel,  OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (levels is null) 
                throw new ArgumentNullException(nameof(levels));
            levels.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_evenLevels(levels.CvPtr, nLevels, lowerLevel, upperLevel, ToPtr(stream)));

            levels.Fix();
        }

        /// <summary>
        /// Perform image denoising using Non-local Means Denoising algorithm.
        /// </summary>
        /// <param name="src">Input 8-bit single-channel image.</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="h">Parameter regulating filter strength. Big h value perfectly removes noise but also removes image details.</param>
        /// <param name="searchWindow">Size in pixels of window that is used to compute weights for pixel.</param>
        /// <param name="blockSize">Size in pixels of template patch that is used to compute weights.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void FastNlMeansDenoising(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, float h, int searchWindow = 21, int blockSize = 7, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_fastNlMeansDenoising(src.CvPtr, dst.CvPtr, h, searchWindow, blockSize, ToPtr(stream)));

            GC.KeepAlive(src);
            dst.Fix();
        }

        /// <summary>
        /// Routines for correcting image color gamma.
        /// </summary>
        /// <param name="src">Source image (8-bit 3-channel BGR/RGB or grayscale).</param>
        /// <param name="dst">Destination image.</param>
        /// <param name="forward">If true, forward gamma correction is performed (gamma=1/2.2). If false, inverse correction (gamma=2.2) is performed.</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public static void GammaCorrection(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, bool forward = true, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));

            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_gammaCorrection(src.CvPtr, dst.CvPtr, forward ? 1 : 0, ToPtr(stream)));
            
            GC.KeepAlive(src);
            dst.Fix();
        }
    }
}
