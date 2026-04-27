using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;


public class BackgroundSubtractorGMG : BackgroundSubtractor
{
    /// <summary>
    /// Internal constructor that satisfies the base class cv::Ptr requirements.
    /// </summary>
    protected BackgroundSubtractorGMG(IntPtr smartPtr, IntPtr rawPtr)
        : base(smartPtr, rawPtr,p=> NativeMethods.cuda_BackgroundSubtractorGMG_delete(p))
    {
    }

    /// <summary>
    /// Creates GMG Background Subtractor.
    /// </summary>
    /// <param name="initializationFrames">Number of frames used to initialize the background models.</param>
    /// <param name="decisionThreshold">Threshold value, above which it is marked foreground.</param>
    /// <returns></returns>
    public static BackgroundSubtractorGMG Create(int initializationFrames = 120, double decisionThreshold = 0.8)
    {
        NativeMethods.HandleException(
            NativeMethods.cuda_createBackgroundSubtractorGMG(
                initializationFrames, decisionThreshold, out var smartPtr));

        // Extract the raw pointer for the base class functionality
       
        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorGMG_get(smartPtr, out IntPtr rawPtr));

        return new BackgroundSubtractorGMG(smartPtr, rawPtr);
    }

    /// <summary>
    /// Updates the background model and computes the foreground mask, with CUDA Stream support.
    /// </summary>
    /// <param name="image">Next video frame.</param>
    /// <param name="fgmask">The output foreground mask as an 8-bit binary image.</param>
    /// <param name="learningRate">The value between 0 and 1 that indicates how fast the background model is learnt.</param>
    /// <param name="stream">Stream for the asynchronous version.</param>
    public virtual void Apply(OpenCvSharp.Cuda.InputArray image, OpenCvSharp.Cuda.OutputArray fgmask, double learningRate = -1, OpenCvSharp.Cuda.Stream? stream = null)
    {
        if (image is null)
            throw new ArgumentNullException(nameof(image));
        if (fgmask is null)
            throw new ArgumentNullException(nameof(fgmask));

        image.ThrowIfDisposed();
        fgmask.ThrowIfNotReady();
        ThrowIfDisposed();

        NativeMethods.HandleException(
            NativeMethods.cuda_BackgroundSubtractorGMG_apply(
                RawPtr, image.CvPtr, fgmask.CvPtr, learningRate, stream?.CvPtr ?? IntPtr.Zero));

        GC.KeepAlive(this);
        GC.KeepAlive(image);
        fgmask.Fix();

    }
}

