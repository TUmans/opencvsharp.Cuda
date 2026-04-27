using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class BackgroundSubtractorMOG2 : BackgroundSubtractor
{
    protected BackgroundSubtractorMOG2(IntPtr smartPtr, IntPtr rawPtr)
           : base(smartPtr, rawPtr, p => NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG2_delete(p))) { }

    public static BackgroundSubtractorMOG2 Create(
        int history = 500, double varThreshold = 16, bool detectShadows = true)
    {
        NativeMethods.HandleException(NativeMethods.cuda_createBackgroundSubtractorMOG2(
            history, varThreshold, detectShadows ? 1 : 0, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG2_get(smartPtr, out IntPtr rawPtr));
        return new BackgroundSubtractorMOG2(smartPtr, rawPtr);
    }

    public virtual void Apply(
        OpenCvSharp.Cuda.InputArray image, OpenCvSharp.Cuda.OutputArray fgmask,
        double learningRate = -1, OpenCvSharp.Cuda.Stream? stream = null)
    {
        if (image is null) throw new ArgumentNullException(nameof(image));
        if (fgmask is null) throw new ArgumentNullException(nameof(fgmask));
        image.ThrowIfDisposed(); fgmask.ThrowIfNotReady(); ThrowIfDisposed();

        NativeMethods.HandleException(NativeMethods.cuda_BackgroundSubtractorMOG2_apply(
            RawPtr, image.CvPtr, fgmask.CvPtr, learningRate, stream?.CvPtr ?? IntPtr.Zero));

        fgmask.Fix(); GC.KeepAlive(this); GC.KeepAlive(image);
    }
}
