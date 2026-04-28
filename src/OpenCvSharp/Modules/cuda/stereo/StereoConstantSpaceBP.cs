using System;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda
{
    /// <summary>
    /// Class computing stereo correspondence using the constant space belief propagation algorithm.
    /// </summary>
    public class StereoConstantSpaceBP : OpenCvSharp.Cuda.StereoMatcher
    {
        protected StereoConstantSpaceBP(IntPtr smartPtr, IntPtr rawPtr)
            : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_StereoConstantSpaceBP_delete(p)))
        {
        }

        /// <summary>
        /// Creates StereoConstantSpaceBP object.
        /// </summary>
        /// <param name="ndisp">Number of disparities.</param>
        /// <param name="iters">Number of BP iterations on each level.</param>
        /// <param name="levels">Number of levels.</param>
        /// <param name="nrPlane">Number of disparity planes on each level.</param>
        /// <param name="msgType">Type for messages. CV_16SC1 and CV_32FC1 types are supported.</param>
        /// <returns></returns>
        public static StereoConstantSpaceBP Create(
            int ndisp = 128, int iters = 8, int levels = 4, int nrPlane = 4, MatType? msgType = null)
        {
            int type = msgType?.Value ?? (int)MatType.CV_32F;

            NativeMethods.HandleException(
                NativeMethods.cuda_createStereoConstantSpaceBP(ndisp, iters, levels, nrPlane, type, out var smartPtr));

            NativeMethods.HandleException(
                NativeMethods.cuda_StereoConstantSpaceBP_get(smartPtr, out var rawPtr));

            return new StereoConstantSpaceBP(smartPtr, rawPtr);
        }
    }
}
