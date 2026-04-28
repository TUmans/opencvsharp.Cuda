using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;


public class StereoBeliefPropagation : OpenCvSharp.Cuda. StereoMatcher
{
    protected StereoBeliefPropagation(IntPtr smartPtr, IntPtr rawPtr)
        : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_StereoBeliefPropagation_delete(p)))
    {
    }

    /// <summary>
    /// Creates StereoBeliefPropagation object.
    /// </summary>
    /// <param name="ndisp">Number of disparities.</param>
    /// <param name="iters">Number of BP iterations on each level.</param>
    /// <param name="levels">Number of levels.</param>
    /// <param name="msgType">Type for messages. CV_16SC1 and CV_32FC1 types are supported.</param>
    /// <returns></returns>
    public static StereoBeliefPropagation Create(
        int ndisp = 64, int iters = 5, int levels = 5, MatType? msgType = null)
    {
        int type = msgType?.Value ?? (int)MatType.CV_32F;

        NativeMethods.HandleException(
            NativeMethods.cuda_createStereoBeliefPropagation(ndisp, iters, levels, type, out var smartPtr));

        NativeMethods.HandleException(
            NativeMethods.cuda_StereoBeliefPropagation_get(smartPtr, out var rawPtr));

        return new StereoBeliefPropagation(smartPtr, rawPtr);
    }
}
