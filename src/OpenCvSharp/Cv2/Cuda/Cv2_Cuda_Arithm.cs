using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Internal;

namespace OpenCvSharp;

public static partial class Cv2
{
    public static partial class Cuda
    {
        #region CudaArithm


        // -----------------------------------------------------------------------
        // abs
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes an absolute value of each matrix element.
        /// </summary>
        public static void Abs(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_abs(src.CvPtr, dst.CvPtr,ToPtr(stream)));

            GC.KeepAlive(src);
            GC.KeepAlive(dst);
            dst.Fix();
        }


        // -----------------------------------------------------------------------
        // absdiff
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes per-element absolute difference of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void Absdiff(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_absdiff( src1.CvPtr, src2.CvPtr, dst.CvPtr, ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(dst);
            dst.Fix();
        }

        /// <summary>
        /// Computes per-element absolute difference of a matrix and scalar. 
        /// </summary>
        public static void AbsdiffWithScalar(OpenCvSharp.Cuda.InputArray src1, Scalar src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (dst is null) throw new ArgumentNullException(nameof(dst));

            src1.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_absdiffWithScalar(src1.CvPtr, src2, dst.CvPtr, ToPtr(stream)));

            
            GC.KeepAlive(src1);
            GC.KeepAlive(dst);
            dst.Fix();
        }
        // -----------------------------------------------------------------------
        // absSum
        // -----------------------------------------------------------------------

        public static Scalar AbsSum(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.InputArray? mask = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            src.ThrowIfDisposed();
            mask?.ThrowIfDisposed();

            // If mask is null, pass IntPtr.Zero. The C++ side will convert this to cv::noArray()
            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;

            NativeMethods.HandleException(NativeMethods.cuda_absSum(src.CvPtr, maskPtr, out var ret));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            return ret;
        }

        // -----------------------------------------------------------------------
        // add
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar sum.
        /// </summary>
        public static void Add(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, int dtype = -1, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null) 
                throw new ArgumentNullException(nameof(src2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed(); 
            src2.ThrowIfDisposed(); 
            dst.ThrowIfNotReady();
            NativeMethods.HandleException(NativeMethods.cuda_add(src1.CvPtr, src2.CvPtr, dst.CvPtr, mask?.CvPtr ?? IntPtr.Zero, dtype, ToPtr(stream)));
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(mask);
            dst.Fix();
        }


        // -----------------------------------------------------------------------
        // addWeighted
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the weighted sum of two arrays:
        /// dst = alpha * src1 + beta * src2 + gamma.
        /// </summary>
        public static void AddWeighted(OpenCvSharp.Cuda.InputArray src1, double alpha, OpenCvSharp.Cuda.InputArray src2, double beta, double gamma, OpenCvSharp.Cuda.OutputArray dst, int dtype = -1, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null) 
                throw new ArgumentNullException(nameof(src2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_addWeighted(
                    src1.CvPtr, alpha,
                    src2.CvPtr, beta,
                    gamma, dst.CvPtr,
                    dtype,ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // addWeighted
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-scalar sum. 
        /// </summary>
        public static void Add( OpenCvSharp.Cuda.InputArray src1, Scalar src2, OpenCvSharp.Cuda.OutputArray dst,OpenCvSharp.Cuda.InputArray? mask = null, int dtype = -1, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;

            NativeMethods.HandleException(
                NativeMethods.cuda_addWithScalar(src1.CvPtr, src2, dst.CvPtr, maskPtr, dtype, ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(mask);
            dst.Fix();

        }

        // -----------------------------------------------------------------------
        // bitwise_and
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise conjunction of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseAnd(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null) 
                throw new ArgumentNullException(nameof(src2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_and(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    mask?.CvPtr ?? IntPtr.Zero, ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        /// <summary>
        /// Performs a per-element bitwise conjunction of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseAnd( OpenCvSharp.Cuda.InputArray src1, Scalar src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_and_with_scalar(src1.CvPtr, src2, dst.CvPtr, maskPtr, ToPtr(stream)));

            GC.KeepAlive(src1); 
            GC.KeepAlive(mask);
            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_not
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise inversion.
        /// </summary>
        public static void BitwiseNot(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,  OpenCvSharp.Cuda.InputArray? mask = null,  OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) 
                throw new ArgumentNullException(nameof(src));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_not(
                    src.CvPtr, dst.CvPtr,
                    mask?.CvPtr ?? IntPtr.Zero, ToPtr(stream)));

            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // bitwise_or
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise disjunction of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseOr(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null) 
                throw new ArgumentNullException(nameof(src2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            mask?.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_or(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    mask?.CvPtr ?? IntPtr.Zero,ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        /// <summary>
        /// Performs a per-element bitwise disjunction (OR) of a matrix and scalar.
        /// </summary>
        public static void BitwiseOr(OpenCvSharp.Cuda.InputArray src1, Scalar src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_or_with_scalar(src1.CvPtr, src2, dst.CvPtr, maskPtr, ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(mask);
            dst.Fix();
         
        }


        // -----------------------------------------------------------------------
        // bitwise_xor
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs a per-element bitwise exclusive-or of two matrices (or of a
        /// matrix and a scalar).
        /// </summary>
        public static void BitwiseXor(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null) 
                throw new ArgumentNullException(nameof(src2));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            mask?.ThrowIfDisposed();    
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_xor(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,
                    mask?.CvPtr ?? IntPtr.Zero,ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        /// <summary>
        /// Performs a per-element bitwise exclusive or (XOR) operation of a matrix and a scalar.
        /// </summary>
        public static void BitwiseXor( OpenCvSharp.Cuda.InputArray src1, Scalar src2, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) 
                throw new ArgumentNullException(nameof(src1));
            if (dst is null) 
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;

            NativeMethods.HandleException(
                NativeMethods.cuda_bitwise_xor_with_scalar(src1.CvPtr, src2, dst.CvPtr, maskPtr, ToPtr(stream)));

            GC.KeepAlive(src1);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // CalcAbsSum
        // -----------------------------------------------------------------------

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the absSum() function only in what argument(s) it accepts. 
        /// </summary>

        public static void CalcAbsSum(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;
            NativeMethods.HandleException(NativeMethods.cuda_calcAbsSum(src.CvPtr, dst.CvPtr, maskPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            dst.Fix();
        }

       

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the Hist() function only in what argument(s) it accepts. 
        /// </summary>
        public static void CalcHist(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray hist, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (hist is null)
                throw new ArgumentNullException(nameof(hist));
            src.ThrowIfDisposed();
            hist.ThrowIfNotReady();
            mask?.ThrowIfDisposed();
            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;
            NativeMethods.HandleException(NativeMethods.cuda_calcHist(src.CvPtr, maskPtr, hist.CvPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            hist.Fix();
        }

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the Norm() function only in what argument(s) it accepts. 
        /// </summary>
        public static void CalcNorm(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, NormTypes normType, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();
            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;
            NativeMethods.HandleException(NativeMethods.cuda_calcNorm(src.CvPtr, dst.CvPtr, (int)normType, maskPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the NormDiff() function only in what argument(s) it accepts. 
        /// </summary>
        public static void CalcNormDiff(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst, NormTypes normType = NormTypes.L2, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null)
                throw new ArgumentNullException(nameof(src1));
            if (src2 is null)
                throw new ArgumentNullException(nameof(src2));
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(NativeMethods.cuda_calcNormDiff(src1.CvPtr, src2.CvPtr, dst.CvPtr, (int)normType, ToPtr(stream)));
            GC.KeepAlive(src1);
            GC.KeepAlive(src2);
            dst.Fix();
        }

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the SqrSum() function only in what argument(s) it accepts. 
        /// </summary>
        public static void CalcSqrSum(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;
            NativeMethods.HandleException(NativeMethods.cuda_calcSqrSum(src.CvPtr, dst.CvPtr, maskPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            dst.Fix();
        }

        /// <summary>
        /// This is an overloaded member function, provided for convenience. It differs from the Sum() function only in what argument(s) it accepts. 
        /// </summary>
        public static void CalcSum(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.InputArray? mask = null, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null)
                throw new ArgumentNullException(nameof(src));
            if (dst is null)
                throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();
            mask?.ThrowIfDisposed();

            IntPtr maskPtr = mask?.CvPtr ?? IntPtr.Zero;
            NativeMethods.HandleException(NativeMethods.cuda_calcSum(src.CvPtr, dst.CvPtr, maskPtr, ToPtr(stream)));
            GC.KeepAlive(src);
            GC.KeepAlive(mask);
            dst.Fix();
        }
        // -----------------------------------------------------------------------
        // cartToPolar
        // -----------------------------------------------------------------------

        /// <summary>
        /// Converts Cartesian coordinates into polar.
        /// </summary>
        public static void CartToPolar(OpenCvSharp.Cuda.InputArray x, OpenCvSharp.Cuda.InputArray y,
            OpenCvSharp.Cuda.OutputArray magnitude, OpenCvSharp.Cuda.OutputArray angle,
            bool angleInDegrees = false,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                   ToPtr(stream)));

            magnitude.Fix();
            angle.Fix();
        }

        // -----------------------------------------------------------------------
        // compare
        // -----------------------------------------------------------------------

        /// <summary>
        /// Compares elements of two matrices (or of a matrix and a scalar).
        /// </summary>
        public static void Compare(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst, CmpTypes cmpop,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                    (int)cmpop,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // divide
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar division.
        /// </summary>
        public static void Divide(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst,
            double scale = 1.0,
            int dtype = -1,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                    scale, dtype,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // exp
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes an exponent of each matrix element.
        /// </summary>
        public static void Exp(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_exp(
                    src.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // log
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a natural logarithm of the absolute value of each matrix
        /// element.
        /// </summary>
        public static void Log(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_log(
                    src.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // lshift
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs pixel-by-pixel left shift of an image by a constant value.
        /// </summary>
        public static void Lshift(OpenCvSharp.Cuda.InputArray src, Vec4i val, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_lshift(
                    src.CvPtr, val, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // magnitude
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes magnitudes of complex matrix elements (2-channel input).
        /// </summary>
        public static void Magnitude(OpenCvSharp.Cuda.InputArray xy, OpenCvSharp.Cuda.OutputArray magnitude,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (xy is null) throw new ArgumentNullException(nameof(xy));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitude_1(
                    xy.CvPtr, magnitude.CvPtr,ToPtr(stream)));

            magnitude.Fix();
        }

        /// <summary>
        /// Computes magnitudes of complex matrix elements from separate X and Y
        /// planes.
        /// </summary>
        public static void Magnitude(OpenCvSharp.Cuda.InputArray x, OpenCvSharp.Cuda.InputArray y,
            OpenCvSharp.Cuda.OutputArray magnitude,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitude_2(
                    x.CvPtr, y.CvPtr, magnitude.CvPtr,ToPtr(stream)));

            magnitude.Fix();
        }

        // -----------------------------------------------------------------------
        // magnitudeSqr
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes squared magnitudes of complex matrix elements (2-channel
        /// input).
        /// </summary>
        public static void MagnitudeSqr(OpenCvSharp.Cuda.InputArray xy, OpenCvSharp.Cuda.OutputArray magnitude,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (xy is null) throw new ArgumentNullException(nameof(xy));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            xy.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitudeSqr_1(
                    xy.CvPtr, magnitude.CvPtr,ToPtr(stream)));

            magnitude.Fix();
        }

        /// <summary>
        /// Computes squared magnitudes from separate X and Y planes.
        /// </summary>
        public static void MagnitudeSqr(OpenCvSharp.Cuda.InputArray x, OpenCvSharp.Cuda.InputArray y,
            OpenCvSharp.Cuda.OutputArray magnitude,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (y is null) throw new ArgumentNullException(nameof(y));
            if (magnitude is null) throw new ArgumentNullException(nameof(magnitude));
            x.ThrowIfDisposed();
            y.ThrowIfDisposed();
            magnitude.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_magnitudeSqr_2(
                    x.CvPtr, y.CvPtr, magnitude.CvPtr,ToPtr(stream)));

            magnitude.Fix();
        }

        // -----------------------------------------------------------------------
        // max
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the per-element maximum of two matrices (or a matrix and a
        /// scalar).
        /// </summary>
        public static void Max(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_max(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // min
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes the per-element minimum of two matrices (or a matrix and a
        /// scalar).
        /// </summary>
        public static void Min(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst, OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src1 is null) throw new ArgumentNullException(nameof(src1));
            if (src2 is null) throw new ArgumentNullException(nameof(src2));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src1.ThrowIfDisposed();
            src2.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_min(
                    src1.CvPtr, src2.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // multiply
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar per-element product.
        /// </summary>
        public static void Multiply(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst,
            double scale = 1.0,
            int dtype = -1,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                    scale, dtype,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // phase
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes polar angles of complex matrix elements.
        /// </summary>
        public static void Phase(OpenCvSharp.Cuda.InputArray x, OpenCvSharp.Cuda.InputArray y, OpenCvSharp.Cuda.OutputArray angle,
            bool angleInDegrees = false,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                   ToPtr(stream)));

            angle.Fix();
        }

        // -----------------------------------------------------------------------
        // polarToCart
        // -----------------------------------------------------------------------

        /// <summary>
        /// Converts polar coordinates into Cartesian.
        /// </summary>
        public static void PolarToCart(OpenCvSharp.Cuda.InputArray magnitude, OpenCvSharp.Cuda.InputArray angle,
            OpenCvSharp.Cuda.OutputArray x, OpenCvSharp.Cuda.OutputArray y,
            bool angleInDegrees = false,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                   ToPtr(stream)));

            x.Fix();
            y.Fix();
        }

        // -----------------------------------------------------------------------
        // pow
        // -----------------------------------------------------------------------

        /// <summary>
        /// Raises every matrix element to a power.
        /// </summary>
        public static void Pow(OpenCvSharp.Cuda.InputArray src, double power, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_pow(
                    src.CvPtr, power, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // rshift
        // -----------------------------------------------------------------------

        /// <summary>
        /// Performs pixel-by-pixel right shift of an image by a constant value.
        /// </summary>
        public static void Rshift(OpenCvSharp.Cuda.InputArray src, Vec4i val, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_rshift(
                    src.CvPtr, val, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // scaleAdd
        // -----------------------------------------------------------------------

        /// <summary>
        /// Adds a scaled array to another one: dst = alpha * src1 + src2.
        /// </summary>
        public static void ScaleAdd(OpenCvSharp.Cuda.InputArray src1, double alpha,
            OpenCvSharp.Cuda.InputArray src2, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                   ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // sqr
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a square value of each matrix element.
        /// </summary>
        public static void Sqr(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_sqr(
                    src.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // sqrt
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a square root of each matrix element.
        /// </summary>
        public static void Sqrt(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_sqrt(
                    src.CvPtr, dst.CvPtr,ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // subtract
        // -----------------------------------------------------------------------

        /// <summary>
        /// Computes a matrix-matrix or matrix-scalar difference.
        /// </summary>
        public static void Subtract(OpenCvSharp.Cuda.InputArray src1, OpenCvSharp.Cuda.InputArray src2,
            OpenCvSharp.Cuda.OutputArray dst,
            OpenCvSharp.Cuda.InputArray? mask = null,
            int dtype = -1,
            OpenCvSharp.Cuda.Stream? stream = null)
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
                    mask?.CvPtr ?? IntPtr.Zero , dtype,
                   ToPtr(stream)));

            dst.Fix();
        }

        // -----------------------------------------------------------------------
        // threshold
        // -----------------------------------------------------------------------

        /// <summary>
        /// Applies a fixed-level threshold to each array element.
        /// Returns the computed threshold value (relevant for Otsu / Triangle).
        /// </summary>
        public static double Threshold(OpenCvSharp.Cuda.InputArray src, OpenCvSharp.Cuda.OutputArray dst,
            double thresh, double maxval,
            ThresholdTypes type,
            OpenCvSharp.Cuda.Stream? stream = null)
        {
            if (src is null) throw new ArgumentNullException(nameof(src));
            if (dst is null) throw new ArgumentNullException(nameof(dst));
            src.ThrowIfDisposed();
            dst.ThrowIfNotReady();

            NativeMethods.HandleException(
                NativeMethods.cuda_threshold(
                    src.CvPtr, dst.CvPtr,
                    thresh, maxval, (int)type,
                   ToPtr(stream),
                    out double retVal));

            dst.Fix();
            return retVal;
        }

    }
        #endregion
}

