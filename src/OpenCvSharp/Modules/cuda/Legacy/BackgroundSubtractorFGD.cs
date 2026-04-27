using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class BackgroundSubtractorFGD : BackgroundSubtractor
{
    protected BackgroundSubtractorFGD(IntPtr smartPtr, IntPtr rawPtr)
         : base(smartPtr, rawPtr, p=> NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorFGD_delete(p)))
    {
    }

    /// <summary>
    /// Creates an FGD Background Subtractor using default parameters.
    /// </summary>
    /// <returns></returns>
    public static BackgroundSubtractorFGD Create()
    {
        NativeMethods.HandleException(
            NativeMethods.cuda_createBackgroundSubtractorFGD(out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorFGD_get(smartPtr, out IntPtr rawPtr));

        return new BackgroundSubtractorFGD(smartPtr, rawPtr);
    }

    /// <summary>
    /// Updates the background model and computes the foreground mask, with CUDA Stream support.
    /// </summary>
    public virtual void Apply(OpenCvSharp.Cuda.InputArray image, OpenCvSharp.Cuda.OutputArray fgmask, double learningRate = -1)
    {
        if (image is null) 
            throw new ArgumentNullException(nameof(image));
        if (fgmask is null) 
            throw new ArgumentNullException(nameof(fgmask));

        image.ThrowIfDisposed();
        fgmask.ThrowIfNotReady();
        ThrowIfDisposed();

        NativeMethods.HandleException(
            NativeMethods.cuda_BackgroundSubtractorFGD_apply(
                RawPtr, image.CvPtr, fgmask.CvPtr, learningRate));

        fgmask.Fix();
        GC.KeepAlive(this);
        GC.KeepAlive(image);
    }
}
