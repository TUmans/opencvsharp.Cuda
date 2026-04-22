#if ENABLED_CUDA

using System;
using System.Collections.Generic;
using OpenCvSharp.Cuda;
using OpenCvSharp.Internal;

namespace OpenCvSharp
{
    static partial class Cv2
    {
        #region Hardware

        /// <summary>
        /// Returns the number of installed CUDA-enabled devices.
        /// Use this function before any other GPU functions calls. 
        /// If OpenCV is compiled without GPU support, this function returns 0.
        /// </summary>
        /// <returns></returns>
        public static int GetCudaEnabledDeviceCount()
        {
            NativeMethods.HandleException(NativeMethods.cuda_getCudaEnabledDeviceCount(out int res));
            return res;
        }

        /// <summary>
        /// Returns the current device index set by SetDevice() or initialized by default.
        /// </summary>
        /// <returns></returns>
        public static int GetDevice()
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.HandleException(NativeMethods.cuda_getDevice(out int res));
            return res;
        }

        /// <summary>
        /// Sets a device and initializes it for the current thread.
        /// </summary>
        /// <param name="device">System index of a GPU device starting with 0.</param>
        /// <returns></returns>
        public static int SetDevice(int device)
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.HandleException(NativeMethods.cuda_getDevice(out int res));
            return res;
        }

        /// <summary>
        /// Explicitly destroys and cleans up all resources associated with the current device in the current process.
        /// Any subsequent API call to this device will reinitialize the device.
        /// </summary>
        public static void ResetDevice()
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.cuda_resetDevice();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="device"></param>
        public static void PrintCudaDeviceInfo(int device)
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.cuda_printCudaDeviceInfo(device);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="device"></param>
        public static void PrintShortCudaDeviceInfo(int device)
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.cuda_printShortCudaDeviceInfo(device);
        }

        #endregion

        #region CudaMem

        /// <summary>
        /// Page-locks the matrix m memory and maps it for the device(s)
        /// </summary>
        /// <param name="m"></param>
        public static void RegisterPageLocked(Mat m)
        {
            ThrowIfGpuNotAvailable();
            if (m is null)
                throw new ArgumentNullException(nameof(m));
            NativeMethods.cuda_registerPageLocked(m.CvPtr);
            GC.KeepAlive(m);
        }

        /// <summary>
        /// Unmaps the memory of matrix m, and makes it pageable again.
        /// </summary>
        /// <param name="m"></param>
        public static void UnregisterPageLocked(Mat m)
        {
            ThrowIfGpuNotAvailable();
            if (m is null)
                throw new ArgumentNullException(nameof(m));
            NativeMethods.cuda_unregisterPageLocked(m.CvPtr);
            GC.KeepAlive(m);
        }

        #endregion

        #region GpuMat

        /// <summary>
        /// Creates continuous GPU matrix
        /// </summary>
        /// <param name="rows">Number of rows in a 2D array.</param>
        /// <param name="cols">Number of columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void CreateContinuous(int rows, int cols, MatType type, GpuMat m)
        {
            ThrowIfGpuNotAvailable();
            if (m is null)
                throw new ArgumentNullException(nameof(m));
            NativeMethods.cuda_createContinuous1(rows, cols, (int)type, m.CvPtr);
            GC.KeepAlive(m);
        }

        /// <summary>
        /// Creates continuous GPU matrix
        /// </summary>
        /// <param name="rows">Number of rows in a 2D array.</param>
        /// <param name="cols">Number of columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <returns></returns>
        public static GpuMat CreateContinuous(int rows, int cols, MatType type)
        {
            ThrowIfGpuNotAvailable();
            NativeMethods.HandleException(NativeMethods.cuda_createContinuous2(rows, cols, (int)type, out IntPtr ret));
            return new GpuMat(ret);
        }

        /// <summary>
        /// Creates continuous GPU matrix
        /// </summary>
        /// <param name="size">Number of rows and columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void CreateContinuous(Size size, MatType type, GpuMat m)
        {
            ThrowIfGpuNotAvailable();
            CreateContinuous(size.Height, size.Width, type, m);
        }

        /// <summary>
        /// Creates continuous GPU matrix
        /// </summary>
        /// <param name="size">Number of rows and columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <returns></returns>
        public static GpuMat CreateContinuous(Size size, MatType type)
        {
            ThrowIfGpuNotAvailable();
            return CreateContinuous(size.Height, size.Width, type);
        }

        /// <summary>
        /// Ensures that size of the given matrix is not less than (rows, cols) size
        /// and matrix type is match specified one too
        /// </summary>
        /// <param name="rows">Number of rows in a 2D array.</param>
        /// <param name="cols">Number of columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void EnsureSizeIsEnough(int rows, int cols, MatType type, GpuMat m)
        {
            ThrowIfGpuNotAvailable();
            if (m is null)
                throw new ArgumentNullException(nameof(m));
            NativeMethods.cuda_ensureSizeIsEnough(rows, cols, (int)type, m.CvPtr);
            GC.KeepAlive(m);
        }

        /// <summary>
        /// Ensures that size of the given matrix is not less than (rows, cols) size
        /// and matrix type is match specified one too
        /// </summary>
        /// <param name="size">Number of rows and columns in a 2D array.</param>
        /// <param name="type">Array type.</param>
        /// <param name="m"></param>
        public static void EnsureSizeIsEnough(Size size, MatType type, GpuMat m)
        {
            ThrowIfGpuNotAvailable();
            EnsureSizeIsEnough(size.Height, size.Width, type, m);
        }

        #endregion

        /// <summary>
        /// 
        /// </summary>
        public static void ThrowIfGpuNotAvailable()
        {
            if (GetCudaEnabledDeviceCount() < 1)
                throw new OpenCvSharpException("GPU module cannot be used.");
        }

        #region CudaArithm


        // -----------------------------------------------------------------------
        // abs
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes an absolute value of each matrix element.
        /// </summary>
        public static void Abs(Cuda.InputArray src, Cuda.OutputArray dst, Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_abs(src.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // absdiff
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes per-element absolute difference of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void Absdiff(Cuda.InputArray src1, Cuda.InputArray src2, Cuda.OutputArray dst, Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_absdiff(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // add
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar sum.
        /// </summary>
        public static void Add(Cuda.InputArray src1, Cuda.InputArray src2, Cuda.OutputArray dst, Cuda.InputArray? mask = null,
            int dtype = -1,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_add(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    ToPtr(mask), dtype,
                    stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // addWeighted
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the weighted sum of two arrays:
        /// dst = alpha * src1 + beta * src2 + gamma.
        /// </summary>
        public static void AddWeighted(
            Cuda.InputArray src1, double alpha,
            Cuda.InputArray src2, double beta,
            double gamma,
            Cuda.OutputArray dst,
            int dtype = -1,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_addWeighted(
                    src1.CvPtr, alpha,
                    src2.CvPtr, beta,
                    gamma, dst.CvPtr,
                    dtype, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_and
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise conjunction of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseAnd(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            Cuda.InputArray? mask = null,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_and(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    ToPtr(mask), stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_not
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise inversion.
        /// </summary>
        public static void BitwiseNot(Cuda.InputArray src, Cuda.OutputArray dst,
            Cuda.InputArray? mask = null,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_not(
                    src.CvPtr, dst.CvPtr,
                    ToPtr(mask), stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_or
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise disjunction of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseOr(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            Cuda.InputArray? mask = null,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_or(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    ToPtr(mask), stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_xor
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise exclusive-or of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseXor(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            Cuda.InputArray? mask = null,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_xor(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    ToPtr(mask), stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // cartToPolar
        // -----------------------------------------------------------------------

        /// <summary>
        /// Converts Cartesian coordinates into polar.
        /// </summary>
        public static void CartToPolar(Cuda.InputArray x, Cuda.InputArray y,
            Cuda.OutputArray magnitude, Cuda.OutputArray angle,
            bool angleInDegrees = false,
            Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            if (angle is null) throw new ArgumentNullException(nameof(angle));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();
            angle.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_cartToPolar(
                    x.CvPtr, y.CvPtr,
                    magnitude.CvPtr, angle.CvPtr,
                    angleInDegrees ? 1 : 0,
                    stream.CvPtr));

            magnitude.Fix();
            angle.Fix();
        }

        // -----------------------------------------------------------------------
        // compare
        // -----------------------------------------------------------------------

        /// <summary>
        /// Compares elements of two matrices (or of a matrix and a scalar).
        /// </summary>
        public static void Compare(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst, CmpTypes cmpop,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_compare(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    (int)cmpop, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // divide
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar division.
        /// </summary>
        public static void Divide(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            double scale = 1.0,
            int dtype = -1,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_divide(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    scale, dtype, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // exp
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes an exponent of each matrix element.
        /// </summary>
        public static void Exp(Cuda.InputArray src, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_exp(
                    src.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // log
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a natural logarithm of the absolute value of each matrix
        /// element.
        /// </summary>
        public static void Log(Cuda.InputArray src, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_log(
                    src.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // lshift
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs pixel-by-pixel left shift of an image by a constant value.
        /// </summary>
        public static void Lshift(Cuda.InputArray src, Vec4i val, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_lshift(
                    src.CvPtr, val, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // magnitude
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes magnitudes of complex matrix elements (2-channel input).
        /// </summary>
        public static void Magnitude(Cuda.InputArray xy, Cuda.OutputArray magnitude,
            Cuda.Stream? stream = null)
        {
            if (xy is null) throw new ArgumentNullException(nameof(xy));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitude_1(
                    xy.CvPtr, magnitude.CvPtr, stream.CvPtr));

            magnitude.Fix();
        }

        /// <summary>
        /// Computes magnitudes of complex matrix elements from separate X and Y
        /// planes.
        /// </summary>
        public static void Magnitude(Cuda.InputArray x, Cuda.InputArray y,
            Cuda.OutputArray magnitude,
            Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitude_2(
                    x.CvPtr, y.CvPtr, magnitude.CvPtr, stream.CvPtr));

            magnitude.Fix();
        }

        // -----------------------------------------------------------------------
        // magnitudeSqr
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes squared magnitudes of complex matrix elements (2-channel
        /// input).
        /// </summary>
        public static void MagnitudeSqr(Cuda.InputArray xy, Cuda.OutputArray magnitude,
            Cuda.Stream? stream = null)
        {
            if (xy is null) throw new ArgumentNullException(nameof(xy));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitudeSqr_1(
                    xy.CvPtr, magnitude.CvPtr, stream.CvPtr));

            magnitude.Fix();
        }

        /// <summary>
        /// Computes squared magnitudes from separate X and Y planes.
        /// </summary>
        public static void MagnitudeSqr(Cuda.InputArray x, Cuda.InputArray y,
            Cuda.OutputArray magnitude,
            Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitudeSqr_2(
                    x.CvPtr, y.CvPtr, magnitude.CvPtr, stream.CvPtr));

            magnitude.Fix();
        }

        // -----------------------------------------------------------------------
        // max
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the per-element maximum of two matrices (or a matrix and a
        /// scalar).
        /// </summary>
        public static void Max(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst, Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_max(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // min
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the per-element minimum of two matrices (or a matrix and a
        /// scalar).
        /// </summary>
        public static void Min(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst, Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_min(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // multiply
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar per-element product.
        /// </summary>
        public static void Multiply(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            double scale = 1.0,
            int dtype = -1,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_multiply(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    scale, dtype, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // phase
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes polar angles of complex matrix elements.
        /// </summary>
        public static void Phase(Cuda.InputArray x, Cuda.InputArray y, Cuda.OutputArray angle,
            bool angleInDegrees = false,
            Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (angle is null) throw new ArgumentNullException(nameof(angle));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            angle.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_phase(
                    x.CvPtr, y.CvPtr, angle.CvPtr,
                    angleInDegrees ? 1 : 0,
                    stream.CvPtr));

            angle.Fix();
        }

        // -----------------------------------------------------------------------
        // polarToCart
        // -----------------------------------------------------------------------

        /// <summary>
        /// Converts polar coordinates into Cartesian.
        /// </summary>
        public static void PolarToCart(Cuda.InputArray magnitude, Cuda.InputArray angle,
            Cuda.OutputArray x, Cuda.OutputArray y,
            bool angleInDegrees = false,
            Cuda.Stream? stream = null)
        {
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            if (angle is null) throw new ArgumentNullException(nameof(angle));
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            magnitude.ThrowIfDisposed();
            angle.ThrowIfDisposed();
            x.ThrowIfNotReady();
            y.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_polarToCart(
                    magnitude.CvPtr, angle.CvPtr,
                    x.CvPtr, y.CvPtr,
                    angleInDegrees ? 1 : 0,
                    stream.CvPtr));

            x.Fix();
            y.Fix();
        }

        // -----------------------------------------------------------------------
        // pow
        // -----------------------------------------------------------------------

        /// <summary>
        /// Raises every matrix element to a power.
        /// </summary>
        public static void Pow(Cuda.InputArray src, double power, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_pow(
                    src.CvPtr, power, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // rshift
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs pixel-by-pixel right shift of an image by a constant value.
        /// </summary>
        public static void Rshift(Cuda.InputArray src, Vec4i val, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_rshift(
                    src.CvPtr, val, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // scaleAdd
        // -----------------------------------------------------------------------

        /// <summary>
        /// Adds a scaled array to another one: dst = alpha * src1 + src2.
        /// </summary>
        public static void ScaleAdd(Cuda.InputArray src1, double alpha,
            Cuda.InputArray src2, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_scaleAdd(
                    src1.CvPtr, alpha, src2.CvPtr, dst.CvPtr,
                    stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // sqr
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a square value of each matrix element.
        /// </summary>
        public static void Sqr(Cuda.InputArray src, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_sqr(
                    src.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // sqrt
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a square root of each matrix element.
        /// </summary>
        public static void Sqrt(Cuda.InputArray src, Cuda.OutputArray dst,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_sqrt(
                    src.CvPtr, dst.CvPtr, stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // subtract
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar difference.
        /// </summary>
        public static void Subtract(Cuda.InputArray src1, Cuda.InputArray src2,
            Cuda.OutputArray dst,
            Cuda.InputArray? mask = null,
            int dtype = -1,
            Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_subtract(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    ToPtr(mask), dtype,
                    stream.CvPtr));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // threshold
        // -----------------------------------------------------------------------

        /// <summary>
        /// Applies a fixed-level threshold to each array element.
        /// Returns the computed threshold value (relevant for Otsu / Triangle).
        /// </summary>
        public static double Threshold(Cuda.InputArray src, Cuda.OutputArray dst,
            double thresh, double maxval,
            ThresholdTypes type,
            Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_threshold(
                    src.CvPtr, dst.CvPtr,
                    thresh, maxval, (int)type,
                    stream.CvPtr,
                    out double retVal));

            dst.Fix();
            return retVal;
        }

        // -----------------------------------------------------------------------
        // helpers
        // -----------------------------------------------------------------------

        private static IntPtr ToPtr(Cuda.InputArray? arr) =>
            arr?.CvPtr ?? IntPtr.Zero;
    }
}


        #endregion


#endif
