using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class BackgroundSubtractorMOG : BackgroundSubtractor
{
    protected BackgroundSubtractorMOG(IntPtr smartPtr, IntPtr rawPtr)
           : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG_delete(p))) { }

    public static BackgroundSubtractorMOG Create(
          int history = 200, int nmixtures = 5, double backgroundRatio = 0.7, double noiseSigma = 0)
    {
        NativeMethods.HandleException(NativeMethods.cuda_createBackgroundSubtractorMOG(
            history, nmixtures, backgroundRatio, noiseSigma, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG_get(smartPtr, out IntPtr rawPtr));
        return new BackgroundSubtractorMOG(smartPtr, rawPtr);
    }

    public virtual void Apply(
            OpenCvSharp.Cuda.InputArray image, OpenCvSharp.Cuda.OutputArray fgmask,
            double learningRate = -1, OpenCvSharp.Cuda.Stream? stream = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (fgmask is null) throw new ArgumentNullException(nameof(fgmask));
        image.ThrowIfDisposed(); fgmask.ThrowIfNotReady(); ThrowIfDisposed();

        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG_apply(
            RawPtr, image.CvPtr, fgmask.CvPtr, learningRate, stream?.CvPtr ?? IntPtr.Zero));

        fgmask.Fix(); GC.KeepAlive(this); GC.KeepAlive(image);
    }
}
