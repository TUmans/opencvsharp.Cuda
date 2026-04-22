using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace OpenCvSharp.Tests.cuda;

public class CudaArthim : CudaTestBase
{
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// <summary>
    /// Skips the test when no CUDA-capable device is present.
    /// Call at the top of every test.
    /// </summary>
    private static void SkipIfNoCuda()
    {
          if (Cv2.GetCudaEnabledDeviceCount() == 0) return;
            throw new SkipException("No CUDA device available.");
    }

    /// <summary>Creates a single-channel float Cuda.GpuMat filled with <paramref name="value"/>.</summary>
    private static Cuda.GpuMat MakeFloat(int rows, int cols, float value)
    {
        using var cpu = new Mat(rows, cols, MatType.CV_32FC1, new Scalar(value));
        var gpu = new Cuda.GpuMat();
        gpu.Upload(cpu);
        return gpu;
    }

    /// <summary>Creates a single-channel byte Cuda.GpuMat filled with <paramref name="value"/>.</summary>
    private static Cuda.GpuMat MakeByte(int rows, int cols, byte value)
    {
        using var cpu = new Mat(rows, cols, MatType.CV_8UC1, new Scalar(value));
        var gpu = new Cuda.GpuMat();
        gpu.Upload(cpu);
        return gpu;
    }

    /// <summary>Downloads a Cuda.GpuMat and returns the value of pixel (0,0) as float.</summary>
    private static float PixelF(Cuda.GpuMat gpu)
    {
        using var cpu = new Mat();
        gpu.Download(cpu);
        return cpu.At<float>(0, 0);
    }

    /// <summary>Downloads a Cuda.GpuMat and returns the value of pixel (0,0) as byte.</summary>
    private static byte PixelB(Cuda.GpuMat gpu)
    {
        using var cpu = new Mat();
        gpu.Download(cpu);
        return cpu.At<byte>(0, 0);
    }

    private const int Rows = 4;
    private const int Cols = 4;
    private const float Tolerance = 1e-4f;

    public void Dispose() { }

    // -----------------------------------------------------------------------
    // abs
    // -----------------------------------------------------------------------

    [Fact]
    public void Abs_NegativeValues_ReturnsAbsolute()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, -7f);
        using var dst = new Cuda.GpuMat();

        Cv2.Abs(src, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    [Fact]
    public void Abs_PositiveValues_Unchanged()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 5f);
        using var dst = new Cuda.GpuMat();

        Cv2.Abs(src, dst);

        Assert.Equal(5f, PixelF(dst), 3);
    }

    [Fact]
    public void Abs_NullSrc_Throws() =>
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Abs(null!, new Cuda.GpuMat()));

    [Fact]
    public void Abs_NullDst_Throws()
    {
        SkipIfNoCuda();
        using var src = MakeFloat(Rows, Cols, 1f);
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Abs(src, null!));
    }

    // -----------------------------------------------------------------------
    // absdiff
    // -----------------------------------------------------------------------

    [Fact]
    public void Absdiff_TwoMatrices_ReturnsCorrectDifference()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Absdiff(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    [Fact]
    public void Absdiff_ReversedOrder_StillPositive()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 3f);
        using var src2 = MakeFloat(Rows, Cols, 10f);
        using var dst = new Cuda.GpuMat();

        Cv2.Absdiff(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // add
    // -----------------------------------------------------------------------

    [Fact]
    public void Add_TwoMatrices_ReturnsSum()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 4f);
        using var src2 = MakeFloat(Rows, Cols, 6f);
        using var dst = new Cuda.GpuMat();

        Cv2.Add(src1, src2, dst);

        Assert.Equal(10f, PixelF(dst), 3);
    }

    [Fact]
    public void Add_WithStream_DoesNotThrow()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 1f);
        using var src2 = MakeFloat(Rows, Cols, 2f);
        using var dst = new Cuda.GpuMat();
        using var stream = new Cuda.Stream();

        var ex = Record.Exception(() =>
        {
            Cv2.Add(src1, src2, dst, stream: stream);
            stream.WaitForCompletion();
        });

        Assert.Null(ex);
    }

    // -----------------------------------------------------------------------
    // addWeighted
    // -----------------------------------------------------------------------

    [Fact]
    public void AddWeighted_ComputesCorrectResult()
    {
        SkipIfNoCuda();

        // dst = 2*3 + 3*4 + 1 = 19
        using var src1 = MakeFloat(Rows, Cols, 3f);
        using var src2 = MakeFloat(Rows, Cols, 4f);
        using var dst = new Cuda.GpuMat();

        Cv2.AddWeighted(src1, 2.0, src2, 3.0, 1.0, dst);

        Assert.Equal(19f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // bitwise_and
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseAnd_ByteMatrices_ReturnsCorrectBits()
    {
        SkipIfNoCuda();

        // 0b1111_0000 & 0b1010_1010 = 0b1010_0000 = 160
        using var src1 = MakeByte(Rows, Cols, 0b1111_0000);
        using var src2 = MakeByte(Rows, Cols, 0b1010_1010);
        using var dst = new Cuda.GpuMat();

        Cv2.BitwiseAnd(src1, src2, dst);

        Assert.Equal((byte)0b1010_0000, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // bitwise_not
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseNot_ByteMatrix_ReturnsComplement()
    {
        SkipIfNoCuda();

        // ~0b0000_1111 = 0b1111_0000 = 240
        using var src = MakeByte(Rows, Cols, 0b0000_1111);
        using var dst = new Cuda.GpuMat();

        Cv2.BitwiseNot(src, dst);

        Assert.Equal((byte)0b1111_0000, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // bitwise_or
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseOr_ByteMatrices_ReturnsCorrectBits()
    {
        SkipIfNoCuda();

        // 0b1100_0000 | 0b0000_1100 = 0b1100_1100 = 204
        using var src1 = MakeByte(Rows, Cols, 0b1100_0000);
        using var src2 = MakeByte(Rows, Cols, 0b0000_1100);
        using var dst = new Cuda.GpuMat();

        Cv2.BitwiseOr(src1, src2, dst);

        Assert.Equal((byte)0b1100_1100, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // bitwise_xor
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseXor_ByteMatrices_ReturnsCorrectBits()
    {
        SkipIfNoCuda();

        // 0b1111_0000 ^ 0b1010_1010 = 0b0101_1010 = 90
        using var src1 = MakeByte(Rows, Cols, 0b1111_0000);
        using var src2 = MakeByte(Rows, Cols, 0b1010_1010);
        using var dst = new Cuda.GpuMat();

        Cv2.BitwiseXor(src1, src2, dst);

        Assert.Equal((byte)0b0101_1010, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // cartToPolar
    // -----------------------------------------------------------------------

    [Fact]
    public void CartToPolar_UnitVector_MagnitudeIsOne()
    {
        SkipIfNoCuda();

        // x=1, y=0 → magnitude=1, angle=0°
        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 0f);
        using var mag = new Cuda.GpuMat();
        using var ang = new Cuda.GpuMat();

        Cv2.CartToPolar(x, y, mag, ang, angleInDegrees: true);

        Assert.Equal(1f, PixelF(mag), 3);
        Assert.Equal(0f, PixelF(ang), 3);
    }

    [Fact]
    public void CartToPolar_DiagonalVector_Magnitude_IsRootTwo()
    {
        SkipIfNoCuda();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 1f);
        using var mag = new Cuda.GpuMat();
        using var ang = new Cuda.GpuMat();

        Cv2.CartToPolar(x, y, mag, ang);

        Assert.Equal(Math.Sqrt(2f), PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // compare
    // -----------------------------------------------------------------------

    [Fact]
    public void Compare_EqualMatrices_AllPixelsNonZero()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 5f);
        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst = new Cuda.GpuMat();

        Cv2.Compare(src1, src2, dst, CmpTypes.EQ);

        // OpenCV returns 255 for true comparisons
        using var cpu = new Mat();
        dst.Download(cpu);
        Assert.Equal(255, cpu.At<byte>(0, 0));
    }

    [Fact]
    public void Compare_NotEqual_AllPixelsZero()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 5f);
        using var src2 = MakeFloat(Rows, Cols, 6f);
        using var dst = new Cuda.GpuMat();

        Cv2.Compare(src1, src2, dst, CmpTypes.EQ);

        using var cpu = new Mat();
        dst.Download(cpu);
        Assert.Equal(0, cpu.At<byte>(0, 0));
    }

    // -----------------------------------------------------------------------
    // divide
    // -----------------------------------------------------------------------

    [Fact]
    public void Divide_TwoMatrices_ReturnsQuotient()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 4f);
        using var dst = new Cuda.GpuMat();

        Cv2.Divide(src1, src2, dst);

        Assert.Equal(2.5f, PixelF(dst), 3);
    }

    [Fact]
    public void Divide_WithScale_ReturnsScaledQuotient()
    {
        SkipIfNoCuda();

        // (10 / 2) * 3 = 15
        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 2f);
        using var dst = new Cuda.GpuMat();

        Cv2.Divide(src1, src2, dst, scale: 3.0);

        Assert.Equal(15f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // exp
    // -----------------------------------------------------------------------

    [Fact]
    public void Exp_Zero_ReturnsOne()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 0f);
        using var dst = new Cuda.GpuMat();

        Cv2.Exp(src, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    [Fact]
    public void Exp_One_ReturnsE()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 1f);
        using var dst = new Cuda.GpuMat();

        Cv2.Exp(src, dst);

        Assert.Equal(Math.E, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // log
    // -----------------------------------------------------------------------

    [Fact]
    public void Log_One_ReturnsZero()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 1f);
        using var dst = new Cuda.GpuMat();

        Cv2.Log(src, dst);

        Assert.Equal(0f, PixelF(dst), 3);
    }

    [Fact]
    public void Log_E_ReturnsOne()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, (float) Math.E);
        using var dst = new Cuda.GpuMat();

        Cv2.Log(src, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // lshift / rshift
    // -----------------------------------------------------------------------

    [Fact]
    public void Lshift_ByteMatrix_ShiftsByGivenAmount()
    {
        SkipIfNoCuda();

        // 1 << 3 = 8
        using var src = MakeByte(Rows, Cols, 1);
        using var dst = new Cuda.GpuMat();

        Cv2.Lshift(src, new Vec4i(3, 0, 0, 0), dst);

        Assert.Equal((byte)8, PixelB(dst));
    }

    [Fact]
    public void Rshift_ByteMatrix_ShiftsByGivenAmount()
    {
        SkipIfNoCuda();

        // 16 >> 2 = 4
        using var src = MakeByte(Rows, Cols, 16);
        using var dst = new Cuda.GpuMat();

        Cv2.Rshift(src, new Vec4i(2, 0, 0, 0), dst);

        Assert.Equal((byte)4, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // magnitude
    // -----------------------------------------------------------------------

    [Fact]
    public void Magnitude_SeparatePlanes_Returns345Hypotenuse()
    {
        SkipIfNoCuda();

        // 3-4-5 right triangle
        using var x = MakeFloat(Rows, Cols, 3f);
        using var y = MakeFloat(Rows, Cols, 4f);
        using var mag = new Cuda.GpuMat();

        Cv2.Magnitude(x, y, mag);

        Assert.Equal(5f, PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // magnitudeSqr
    // -----------------------------------------------------------------------

    [Fact]
    public void MagnitudeSqr_SeparatePlanes_ReturnsSquaredMagnitude()
    {
        SkipIfNoCuda();

        // sqrt(3² + 4²) = 5, so sqr = 25
        using var x = MakeFloat(Rows, Cols, 3f);
        using var y = MakeFloat(Rows, Cols, 4f);
        using var mag = new Cuda.GpuMat();

        Cv2.MagnitudeSqr(x, y, mag);

        Assert.Equal(25f, PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // max
    // -----------------------------------------------------------------------

    [Fact]
    public void Max_TwoMatrices_ReturnsLarger()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 7f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Max(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // min
    // -----------------------------------------------------------------------

    [Fact]
    public void Min_TwoMatrices_ReturnsSmaller()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 7f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Min(src1, src2, dst);

        Assert.Equal(3f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // multiply
    // -----------------------------------------------------------------------

    [Fact]
    public void Multiply_TwoMatrices_ReturnsProduct()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 6f);
        using var src2 = MakeFloat(Rows, Cols, 7f);
        using var dst = new Cuda.GpuMat();

        Cv2.Multiply(src1, src2, dst);

        Assert.Equal(42f, PixelF(dst), 3);
    }

    [Fact]
    public void Multiply_WithScale_ReturnsScaledProduct()
    {
        SkipIfNoCuda();

        // (2 * 3) * 0.5 = 3
        using var src1 = MakeFloat(Rows, Cols, 2f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Multiply(src1, src2, dst, scale: 0.5);

        Assert.Equal(3f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // phase
    // -----------------------------------------------------------------------

    [Fact]
    public void Phase_PositiveXZeroY_AngleIsZero()
    {
        SkipIfNoCuda();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 0f);
        using var ang = new Cuda.GpuMat();

        Cv2.Phase(x, y, ang, angleInDegrees: true);

        Assert.Equal(0f, PixelF(ang), 2);
    }

    [Fact]
    public void Phase_EqualXY_AngleIs45Degrees()
    {
        SkipIfNoCuda();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 1f);
        using var ang = new Cuda.GpuMat();

        Cv2.Phase(x, y, ang, angleInDegrees: true);

        Assert.Equal(45f, PixelF(ang), 1);
    }

    // -----------------------------------------------------------------------
    // polarToCart
    // -----------------------------------------------------------------------

    [Fact]
    public void PolarToCart_UnitMagnitudeZeroAngle_XIsOneYIsZero()
    {
        SkipIfNoCuda();

        using var mag = MakeFloat(Rows, Cols, 1f);
        using var ang = MakeFloat(Rows, Cols, 0f);
        using var x = new Cuda.GpuMat();
        using var y = new Cuda.GpuMat();

        Cv2.PolarToCart(mag, ang, x, y, angleInDegrees: true);

        Assert.Equal(1f, PixelF(x), 3);
        Assert.Equal(0f, PixelF(y), 3);
    }

    // -----------------------------------------------------------------------
    // pow
    // -----------------------------------------------------------------------

    [Fact]
    public void Pow_BaseAndExponent_ReturnsCorrectPower()
    {
        SkipIfNoCuda();

        // 3^4 = 81
        using var src = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Pow(src, 4.0, dst);

        Assert.Equal(81f, PixelF(dst), 1);
    }

    [Fact]
    public void Pow_ExponentZero_ReturnsOne()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 99f);
        using var dst = new Cuda.GpuMat();

        Cv2.Pow(src, 0.0, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // scaleAdd
    // -----------------------------------------------------------------------

    [Fact]
    public void ScaleAdd_ComputesAlphaSrc1PlusSrc2()
    {
        SkipIfNoCuda();

        // 2.5 * 4 + 3 = 13
        using var src1 = MakeFloat(Rows, Cols, 4f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.ScaleAdd(src1, 2.5, src2, dst);

        Assert.Equal(13f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // sqr
    // -----------------------------------------------------------------------

    [Fact]
    public void Sqr_PositiveValue_ReturnsSquare()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 9f);
        using var dst = new Cuda.GpuMat();

        Cv2.Sqr(src, dst);

        Assert.Equal(81f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // sqrt
    // -----------------------------------------------------------------------

    [Fact]
    public void Sqrt_PerfectSquare_ReturnsRoot()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 144f);
        using var dst = new Cuda.GpuMat();

        Cv2.Sqrt(src, dst);

        Assert.Equal(12f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // subtract
    // -----------------------------------------------------------------------

    [Fact]
    public void Subtract_TwoMatrices_ReturnsDifference()
    {
        SkipIfNoCuda();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new Cuda.GpuMat();

        Cv2.Subtract(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // threshold
    // -----------------------------------------------------------------------

    [Fact]
    public void Threshold_BinaryType_PixelsAboveThresholdBecome255()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 200f);
        using var dst = new Cuda.GpuMat();

        Cv2.Threshold(src, dst, thresh: 100.0, maxval: 255.0,
            ThresholdTypes.Binary);

        Assert.Equal(255f, PixelF(dst), 0);
    }

    [Fact]
    public void Threshold_BinaryType_PixelsBelowThresholdBecomeZero()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 50f);
        using var dst = new Cuda.GpuMat();

        Cv2.Threshold(src, dst, thresh: 100.0, maxval: 255.0,
            ThresholdTypes.Binary);

        Assert.Equal(0f, PixelF(dst), 0);
    }

    [Fact]
    public void Threshold_ReturnsThresholdValue()
    {
        SkipIfNoCuda();

        using var src = MakeFloat(Rows, Cols, 50f);
        using var dst = new Cuda.GpuMat();

        double returned = Cv2.Threshold(src, dst,
            thresh: 128.0, maxval: 255.0, ThresholdTypes.Binary);

        Assert.Equal(128.0, returned, 3);
    }

    // -----------------------------------------------------------------------
    // null-argument guard tests (no CUDA needed — pure managed layer)
    // -----------------------------------------------------------------------

    [Fact]
    public void Add_NullSrc1_Throws() =>
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Add(null!, new Cuda.GpuMat(), new Cuda.GpuMat()));

    [Fact]
    public void Add_NullSrc2_Throws()
    {
        using var src1 = new Cuda.GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Add(src1, null!, new Cuda.GpuMat()));
    }

    [Fact]
    public void Add_NullDst_Throws()
    {
        using var src1 = new Cuda.GpuMat();
        using var src2 = new Cuda.GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Add(src1, src2, null!));
    }

    [Fact]
    public void Threshold_NullSrc_Throws() =>
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Threshold(null!, new Cuda.GpuMat(), 0, 255, ThresholdTypes.Binary));

    [Fact]
    public void Threshold_NullDst_Throws()
    {
        using var src = new Cuda.GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Threshold(src, null!, 0, 255, ThresholdTypes.Binary));
    }
}

// ---------------------------------------------------------------------------
// Minimal skip-test support (replace with xunit.skip NuGet if available)
// ---------------------------------------------------------------------------

/// <summary>
/// Thrown to signal that a test should be skipped (no CUDA device present).
/// Wire this up with a custom <see cref="SkippableFactAttribute"/> or use the
/// <c>Xunit.SkippableFact</c> NuGet package instead.
/// </summary>
public sealed class SkipException : Exception
{
    public SkipException(string reason) : base(reason) { }
}


