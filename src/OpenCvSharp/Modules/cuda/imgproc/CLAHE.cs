using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class CLAHE : Algorithm
{
    protected CLAHE(IntPtr smartPtr, IntPtr rawPtr)
        : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_CLAHE_delete(p)))
    {
    }

    /// <summary>
    /// Creates implementation for cuda::CLAHE.
    /// </summary>
    /// <param name="clipLimit">Threshold for contrast limiting.</param>
    /// <param name="tileGridSize">Size of grid for histogram equalization. Input image will be divided into equally sized rectangular tiles.</param>
    public static CLAHE Create(double clipLimit = 40.0, Size? tileGridSize = null)
    {
        Size grid = tileGridSize ?? new Size(8, 8);

        NativeMethods.HandleException(NativeMethods.cuda_createCLAHE(
            clipLimit, grid, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_CLAHE_get(smartPtr, out IntPtr rawPtr));

        return new CLAHE(smartPtr, rawPtr);
    }

    /// <summary>
    /// Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.
    /// </summary>
    public virtual void Apply(
        OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
    {
        if (src is null) throw new ArgumentNullException(nameof(src));
        if (dst is null) throw new ArgumentNullException(nameof(dst));

        src.ThrowIfDisposed();
        dst.ThrowIfNotReady();
        ThrowIfDisposed();

        NativeMethods.HandleException(
            NativeMethods.cuda_CLAHE_apply(RawPtr, src.CvPtr, dst.CvPtr, stream?.CvPtr ?? IntPtr.Zero));

        dst.Fix();
        GC.KeepAlive(this);
        GC.KeepAlive(src);
    }
}

