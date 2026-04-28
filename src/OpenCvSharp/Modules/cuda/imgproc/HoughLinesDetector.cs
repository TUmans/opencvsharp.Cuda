using System;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda
{
    /// <summary>
    /// Base class for Hough lines detector.
    /// </summary>
    public class HoughLinesDetector : Algorithm
    {
        protected HoughLinesDetector(IntPtr smartPtr, IntPtr rawPtr)
            : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_HoughLinesDetector_delete(p)))
        {
        }

        /// <summary>
        /// Creates implementation for cuda::HoughLinesDetector.
        /// </summary>
        /// <param name="rho">Distance resolution of the accumulator in pixels.</param>
        /// <param name="theta">Angle resolution of the accumulator in radians.</param>
        /// <param name="threshold">Accumulator threshold parameter. Only those lines are returned that get enough votes.</param>
        /// <param name="doSort">Whether to sort lines by votes.</param>
        /// <param name="maxLines">Maximum number of lines to return.</param>
        /// <returns></returns>
        public static HoughLinesDetector Create(
            float rho, float theta, int threshold, bool doSort = false, int maxLines = 4096)
        {
            NativeMethods.HandleException(
                NativeMethods.cuda_createHoughLinesDetector(
                    rho, theta, threshold, doSort ? 1 : 0, maxLines, out var smartPtr));

            NativeMethods.HandleException(
                NativeMethods.cuda_HoughLinesDetector_get(smartPtr, out var rawPtr));

            return new HoughLinesDetector(smartPtr, rawPtr);
        }

        /// <summary>
        /// Finds lines in a binary image using the classical Hough transform.
        /// </summary>
        /// <param name="src">8-bit, single-channel, binary source image.</param>
        /// <param name="lines">Output matrix of detected lines (CV_32FC2).</param>
        /// <param name="stream">Stream for the asynchronous version.</param>
        public virtual void Detect(
            OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray lines, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (lines is null) throw new ArgumentNullException(nameof(lines));

            src.ThrowIfDisposed();
            lines.ThrowIfNotReady();
            ThrowIfDisposed();

            NativeMethods.HandleException(
                NativeMethods.cuda_HoughLinesDetector_detect(RawPtr, src.CvPtr, lines.CvPtr,stream?.CvPtr ?? IntPtr.Zero));

            lines.Fix();
            GC.KeepAlive(this);
            GC.KeepAlive(src);
        }
    }
}
