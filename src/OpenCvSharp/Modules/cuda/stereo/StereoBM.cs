using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class StereoBM : OpenCvSharp.Cuda.StereoMatcher
{
    protected StereoBM(IntPtr smartPtr, IntPtr rawPtr)
        : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_StereoBM_delete(p)))
    {
    }

    /// <summary>
    /// Creates StereoBM object.
    /// </summary>
    /// <param name="numDisparities">The number of disparities. Must be a multiple of 16.</param>
    /// <param name="blockSize">The linear size of the blocks compared by the algorithm. Must be odd.</param>
    /// <returns></returns>
    public static StereoBM Create(int numDisparities = 64, int blockSize = 19)
    {
        NativeMethods.HandleException(
            NativeMethods.cuda_createStereoBM(numDisparities, blockSize, out var smartPtr));

        NativeMethods.HandleException(
            NativeMethods.cuda_StereoBM_get(smartPtr, out var rawPtr));

        return new StereoBM(smartPtr, rawPtr);
    }
}
