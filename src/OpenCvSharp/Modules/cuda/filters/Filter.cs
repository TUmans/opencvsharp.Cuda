using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda;

public class Filter : Algorithm
{
    protected Filter(IntPtr smartPtr, IntPtr rawPtr)
           : base(smartPtr, rawPtr, p=> NativeMethods.HandleException(NativeMethods.cuda_Filter_delete(p)))
    {
    }

    /// <summary>
    /// Applies the specified filter to the image.
    /// </summary>
    public virtual void Apply(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
    {
        if (src is null) 
            throw new ArgumentNullException(nameof(src));
        if (dst is null) 
            throw new ArgumentNullException(nameof(dst));

        src.ThrowIfDisposed();
        dst.ThrowIfNotReady();
        ThrowIfDisposed();

        NativeMethods.HandleException(
            NativeMethods.cuda_Filter_apply(RawPtr, src.CvPtr, dst.CvPtr, stream?.CvPtr ?? IntPtr.Zero));

        GC.KeepAlive(this);
        GC.KeepAlive(src);
        dst.Fix();
    }

    /// <summary>
    /// Creates a normalized 2D box filter.
    /// </summary>
    public static Filter CreateBoxFilter(MatType srcType, MatType dstType, Size ksize, Point? anchor = null, BorderTypes borderMode = BorderTypes.Default, Scalar? borderVal = null)
    {
        Point pt = anchor ?? new Point(-1, -1);
        Scalar val = borderVal ?? new Scalar(0);

        NativeMethods.HandleException(NativeMethods.cuda_createBoxFilter(
            (int)srcType, (int)dstType, ksize, pt, (int)borderMode, val, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_Filter_get(smartPtr, out IntPtr rawPtr));
        return new Filter(smartPtr, rawPtr);
    }

    /// <summary>
    /// Creates the maximum filter.
    /// </summary>
    public static Filter CreateBoxMaxFilter(MatType srcType, Size ksize, Point? anchor = null, BorderTypes borderMode = BorderTypes.Default, Scalar? borderVal = null)
    {
        Point pt = anchor ?? new Point(-1, -1);
        Scalar val = borderVal ?? new Scalar(0);

        NativeMethods.HandleException(NativeMethods.cuda_createBoxMaxFilter(
            (int)srcType, ksize, pt, (int)borderMode, val, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_Filter_get(smartPtr, out IntPtr rawPtr));
        return new Filter(smartPtr, rawPtr);
    }

    /// <summary>
    /// Creates the minimum filter. 
    /// </summary>
    public static Filter CreateBoxMinFilter(MatType srcType, Size ksize, Point? anchor = null, BorderTypes borderMode = BorderTypes.Default, Scalar? borderVal = null)
    {
        Point pt = anchor ?? new Point(-1, -1);
        Scalar val = borderVal ?? new Scalar(0);

        NativeMethods.HandleException(NativeMethods.cuda_createBoxMinFilter(
            (int)srcType, ksize, pt, (int)borderMode, val, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_Filter_get(smartPtr, out IntPtr rawPtr));
        return new Filter(smartPtr, rawPtr);
    }

    /// <summary>
    /// Creates a vertical 1D box filter.
    /// </summary>
    public static Filter CreateColumnSumFilter(MatType srcType, MatType dstType, int ksize, int anchor = -1, BorderTypes borderMode = BorderTypes.Default, Scalar? borderVal = null)
    {
        Scalar val = borderVal ?? new Scalar(0);

        NativeMethods.HandleException(NativeMethods.cuda_createColumnSumFilter(
            (int)srcType, (int)dstType, ksize, anchor, (int)borderMode, val, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_Filter_get(smartPtr, out IntPtr rawPtr));
        return new Filter(smartPtr, rawPtr);
    }

    /// <summary>
    /// Creates a generalized Deriv operator.
    /// </summary>
    public static Filter CreateDerivFilter(MatType srcType, MatType dstType, int dx, int dy, int ksize, bool normalize = false, double scale = 1, BorderTypes rowBorderMode = BorderTypes.Default, int columnBorderMode = -1)
    {
        NativeMethods.HandleException(NativeMethods.cuda_createDerivFilter(
            (int)srcType, (int)dstType, dx, dy, ksize, normalize ? 1 : 0, scale,
            (int)rowBorderMode, columnBorderMode, out var smartPtr));

        NativeMethods.HandleException(NativeMethods.cuda_Filter_get(smartPtr, out IntPtr rawPtr));
        return new Filter(smartPtr, rawPtr);
    }
}
