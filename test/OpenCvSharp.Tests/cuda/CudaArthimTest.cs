using Xunit;
using OpenCvSharp.Cuda;

namespace OpenCvSharp.Tests.Cuda;

public class CudaArthimTest : CudaTestBase
{
    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// <summary>
    /// Skips the test when no CUDA-capable device is present.
    /// Call at the top of every test.
    /// </summary>
    

    /// <summary>Creates a single-channel float GpuMat filled with <paramref name="value"/>.</summary>
    private static GpuMat MakeFloat(int rows, int cols, float value)
    {
        using var cpu = new Mat(rows, cols, MatType.CV_32FC1, new Scalar(value));
        var gpu = new GpuMat();
        gpu.Upload(cpu);
        return gpu;
    }

    /// <summary>Creates a single-channel byte GpuMat filled with <paramref name="value"/>.</summary>
    private static GpuMat MakeByte(int rows, int cols, byte value)
    {
        using var cpu = new Mat(rows, cols, MatType.CV_8UC1, new Scalar(value));
        var gpu = new GpuMat();
        gpu.Upload(cpu);
        return gpu;
    }

    /// <summary>Downloads a GpuMat and returns the value of pixel (0,0) as float.</summary>
    private static float PixelF(GpuMat gpu)
    {
        using var cpu = new Mat();
        gpu.Download(cpu);
        return cpu.At<float>(0, 0);
    }

    /// <summary>Downloads a GpuMat and returns the value of pixel (0,0) as byte.</summary>
    private static byte PixelB(GpuMat gpu)
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
    public void Abs()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, -7f);
        using var dst = new GpuMat();

        Cv2.Cuda.Abs(src, dst);
        Assert.Equal(7f, PixelF(dst), 3);

        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst2 = new GpuMat();

        Cv2.Cuda.Abs(src2, dst2);

        Assert.Equal(5f, PixelF(dst2), 3);
    }

    // -----------------------------------------------------------------------
    // absdiff
    // -----------------------------------------------------------------------

    [Fact]
    public void Absdiff()
    {
        VerifyCudaSupport();

        // part 1
        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst1 = new GpuMat();

        Cv2.Cuda.Absdiff(src1, src2, dst1);

        Assert.Equal(7f, PixelF(dst1), 3);

        // part 2
        using var src3 = MakeFloat(Rows, Cols, 3f);
        using var src4 = MakeFloat(Rows, Cols, 10f);
        using var dst2 = new GpuMat();

        Cv2.Cuda.Absdiff(src3, src4, dst2);

        Assert.Equal(7f, PixelF(dst2), 3);
    }

   
    [Fact]
    public void AbsdiffWithScalar()
    {
        // 1. Check if a CUDA device is available. If not, skip the test.
        VerifyCudaSupport();

        int rows = 10;
        int cols = 10;

        // Set up test values
        byte matrixValue = 50;
        double scalarValue = 120;
        byte expectedValue = 70; // |50 - 120| = 70

        // 2. Initialize host (CPU) matrices
        using var cpuSrc = new Mat(rows, cols, MatType.CV_8UC1, new Scalar(matrixValue));
        using var cpuDst = new Mat();

        // 3. Initialize device (GPU) matrices and upload data
        using var gpuSrc = new GpuMat();
        using var gpuDst = new GpuMat();

        gpuSrc.Upload(cpuSrc);

        // 4. Call wrapped method
        // (Assuming your static method is in a class named `CudaInterop` or similar)
        Scalar scalarToSubtract = new Scalar(scalarValue);

        Cv2.Cuda.Absdiff(gpuSrc, scalarToSubtract, gpuDst);

        // 5. Download the result from GPU back to CPU to verify
        gpuDst.Download(cpuDst);

        // 6. Assertions (Verify the output)
        Assert.False(cpuDst.Empty(), "Destination matrix should not be empty.");
        Assert.Equal(rows, cpuDst.Rows);
        Assert.Equal(cols, cpuDst.Cols);
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        // 7. Verify the math of every pixel
        var indexer = cpuDst.GetGenericIndexer<byte>();
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < cols; x++)
            {
                Assert.Equal(expectedValue,indexer[y, x]);
            }
        }
    }

    // -----------------------------------------------------------------------
    // absSum
    // -----------------------------------------------------------------------

    [Fact]
    public void AbsSum_1()
    {
        VerifyCudaSupport();

        // Create a 2x2 matrix filled with -5 (Using 32-bit signed int, as 64F is not allowed)
        using var cpuSrc1 = new Mat(2, 2, MatType.CV_32SC1, new Scalar(-5));
        using var gpuSrc1 = new GpuMat();
        gpuSrc1.Upload(cpuSrc1);

        // Calculate AbsSum
        // |-5| * 4 pixels = 20
        Scalar result1 = Cv2.Cuda.AbsSum(gpuSrc1);

        // Val0 contains the sum for the first channel
        Assert.Equal(20, result1.Val0);
    }

    [Fact]
    public void AbsSum_2()
    {
        VerifyCudaSupport();

        using var cpuSrc2 = new Mat(2, 2, MatType.CV_32SC1, new Scalar(-5));
        using var gpuSrc2 = new GpuMat();
        gpuSrc2.Upload(cpuSrc2);

        // Create a mask. Only the top row (2 pixels) will be active (255).
        // The bottom row will be ignored (0).
        using var cpuMask2 = new Mat(2, 2, MatType.CV_8UC1, new Scalar(0));
        cpuMask2.Set<byte>(0, 0, 255);
        cpuMask2.Set<byte>(0, 1, 255);

        using var gpuMask2 = new GpuMat();
        gpuMask2.Upload(cpuMask2);

        // Calculate AbsSum with mask
        // |-5| * 2 active pixels = 10
        Scalar result2 = Cv2.Cuda.AbsSum(gpuSrc2, gpuMask2);

        Assert.Equal(10, result2.Val0);
    }

    // -----------------------------------------------------------------------
    // add
    // -----------------------------------------------------------------------

    [Fact]
    public void Add()
    {
        VerifyCudaSupport();

        // same test as Core
        using var src1 = Mat.FromPixelData(2, 2, MatType.CV_8UC1, new byte[] { 1, 2, 3, 4 });
        using var src2 = Mat.FromPixelData(2, 2, MatType.CV_8UC1, new byte[] { 1, 2, 3, 4 });

        // port everything to cuda
        using GpuMat gpuSrc1 = new GpuMat(src1);
        using GpuMat gpuSrc2 = new GpuMat(src2);
        using GpuMat gpuDst = new GpuMat();
        // the actual function
        Cv2.Cuda.Add(gpuSrc1, gpuSrc2, gpuDst);
        // downlaod results
        using Mat dst = new Mat();
        gpuDst.Download(dst);

        // same test as Core
        Assert.Equal(MatType.CV_8UC1, dst.Type());
        Assert.Equal(2, dst.Rows);
        Assert.Equal(2, dst.Cols);

        Assert.Equal(2, dst.At<byte>(0, 0));
        Assert.Equal(4, dst.At<byte>(0, 1));
        Assert.Equal(6, dst.At<byte>(1, 0));
        Assert.Equal(8, dst.At<byte>(1, 1));
    }

    // -----------------------------------------------------------------------
    // addWeighted
    // -----------------------------------------------------------------------

    [Fact]
    public void AddWeighted()
    {
        VerifyCudaSupport();

        // dst = 2*3 + 3*4 + 1 = 19
        using var src1 = MakeFloat(Rows, Cols, 3f);
        using var src2 = MakeFloat(Rows, Cols, 4f);
        using var dst = new GpuMat();

        Cv2.Cuda.AddWeighted(src1, 2.0, src2, 3.0, 1.0, dst);

        Assert.Equal(19f, PixelF(dst), 2);
    }

    [Fact]
    public void AddScalar()
    {
        using var srcCpu = Mat.FromPixelData(2, 2, MatType.CV_8UC1, new byte[] { 1, 2, 3, 4 });
        using var srcGpu = new GpuMat(srcCpu);
        using GpuMat gpuDst = new GpuMat();
        Cv2.Cuda.Add(new Scalar(10), srcGpu, gpuDst);

        using var dst = new Mat();
        gpuDst.Download(dst);

        Assert.Equal(MatType.CV_8UC1, dst.Type());
        Assert.Equal(2, dst.Rows);
        Assert.Equal(2, dst.Cols);

        Assert.Equal(11, dst.At<byte>(0, 0));
        Assert.Equal(12, dst.At<byte>(0, 1));
        Assert.Equal(13, dst.At<byte>(1, 0));
        Assert.Equal(14, dst.At<byte>(1, 1));

        Cv2.Cuda.Add(srcGpu, new Scalar(10), gpuDst);
        gpuDst.Download(dst);
        Assert.Equal(11, dst.At<byte>(0, 0));
        Assert.Equal(12, dst.At<byte>(0, 1));
        Assert.Equal(13, dst.At<byte>(1, 0));
        Assert.Equal(14, dst.At<byte>(1, 1));

        using var inputArray = OpenCvSharp.Cuda.InputArray.Create(10.0);
        Cv2.Cuda.Add(srcGpu, inputArray, gpuDst);
        gpuDst.Download(dst);
        Assert.Equal(11, dst.At<byte>(0, 0));
        Assert.Equal(12, dst.At<byte>(0, 1));
        Assert.Equal(13, dst.At<byte>(1, 0));
        Assert.Equal(14, dst.At<byte>(1, 1));
    }

    // -----------------------------------------------------------------------
    // bitwise_and
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseAnd()
    {
        VerifyCudaSupport();

        // 0b1111_0000 & 0b1010_1010 = 0b1010_0000 = 160
        using var src1 = MakeByte(Rows, Cols, 0b1111_0000);
        using var src2 = MakeByte(Rows, Cols, 0b1010_1010);
        using var dst = new GpuMat();

        Cv2.Cuda.BitwiseAnd(src1, src2, dst);

        Assert.Equal((byte)0b1010_0000, PixelB(dst));
    }

    [Fact]
    public void BitwiseAndWithScalar()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a matrix filled with 12
        using var cpuSrc = new Mat(3, 3, MatType.CV_8UC1, new Scalar(12));
        using var gpuSrc = new GpuMat();
        gpuSrc.Upload(cpuSrc);

        // 2. Act: Bitwise AND with a scalar value of 10
        using var gpuDst = new GpuMat();
        Cv2.Cuda.BitwiseAnd(gpuSrc, new Scalar(10), gpuDst);
        
        // 3. Download to verify
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // 4. Assert
        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        // Validate that 12 & 10 = 8
        Assert.Equal(8, cpuDst.At<byte>(0, 0));
        Assert.Equal(8, cpuDst.At<byte>(2, 2)); // Check another pixel to ensure matrix-wide execution
    }


    // -----------------------------------------------------------------------
    // bitwise_not
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseNot()
    {
        VerifyCudaSupport();

        // ~0b0000_1111 = 0b1111_0000 = 240
        using var src = MakeByte(Rows, Cols, 0b0000_1111);
        using var dst = new GpuMat();

        Cv2.Cuda.BitwiseNot(src, dst);

        Assert.Equal((byte)0b1111_0000, PixelB(dst));
    }

    [Fact]
    public void BitwiseNot_WithEmptyMask()
    {
        VerifyCudaSupport();

        using var src = MakeByte(Rows, Cols, 0b0000_1111);
        using var mask = MakeByte(Rows, Cols, 0);
        using var dst = MakeByte(Rows, Cols, 0b1010_1010); // Known background

        Cv2.Cuda.BitwiseNot(src, dst, mask: mask);

        Assert.Equal((byte)0b1010_1010, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // bitwise_or
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseOr()
    {
        VerifyCudaSupport();

        // 0b1100_0000 | 0b0000_1100 = 0b1100_1100 = 204
        using var src1 = MakeByte(Rows, Cols, 0b1100_0000);
        using var src2 = MakeByte(Rows, Cols, 0b0000_1100);
        using var dst = new GpuMat();

        Cv2.Cuda.BitwiseOr(src1, src2, dst);

        Assert.Equal((byte)0b1100_1100, PixelB(dst));
    }

    [Fact]
    public void BitwiseOrWithScalar()
    {
        VerifyCudaSupport();

        using var cpuSrc = new Mat(3, 3, MatType.CV_8UC1, new Scalar(10));
        using var gpuSrc = new GpuMat();
        gpuSrc.Upload(cpuSrc);

        using var gpuDst = new GpuMat();
        Cv2.Cuda.BitwiseOr(gpuSrc, new Scalar(12), gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        // 10 | 12 = 14
        Assert.Equal(14, cpuDst.At<byte>(0, 0));
    }

    // -----------------------------------------------------------------------
    // bitwise_xor
    // -----------------------------------------------------------------------

    [Fact]
    public void BitwiseXor_1()
    {
        VerifyCudaSupport();

        // 0b1111_0000 ^ 0b1010_1010 = 0b0101_1010 = 90
        using var src1 = MakeByte(Rows, Cols, 0b1111_0000);
        using var src2 = MakeByte(Rows, Cols, 0b1010_1010);
        using var dst = new GpuMat();

        Cv2.Cuda.BitwiseXor(src1, src2, dst);

        Assert.Equal((byte)0b0101_1010, PixelB(dst));
    }
    [Fact]
    public void BitwiseXor_2()
    {
        VerifyCudaSupport();

        using var srcDst = MakeByte(Rows, Cols, 0b1100_1100);

        // XORing a matrix with itself should result in 0
        Cv2.Cuda.BitwiseXor(srcDst, srcDst, srcDst);

        Assert.Equal((byte)0, PixelB(srcDst));
    }


    [Fact]
    public void BitwiseXorWithScalar()
    {
        VerifyCudaSupport();

        using var cpuSrc = new Mat(3, 3, MatType.CV_8UC1, new Scalar(10));
        using var gpuSrc = new GpuMat();
        gpuSrc.Upload(cpuSrc);

        using var gpuDst = new GpuMat();
        Cv2.Cuda.BitwiseXor(gpuSrc, new Scalar(12), gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        // 10 ^ 12 = 6
        Assert.Equal(6, cpuDst.At<byte>(0, 0));
    }

    [Fact]
    public void CalcSumTest()
    {
        VerifyCudaSupport();

        using var cpuSrc = new Mat(2, 2, MatType.CV_8UC1, new Scalar(10));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        Cv2.Cuda.CalcSum(gpuSrc, gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // 10 * 4 pixels = 40
        Assert.Equal(40.0, cpuDst.At<double>(0, 0));
    }

    [Fact]
    public void CalcAbsSumTest()
    {
        VerifyCudaSupport();

        // Use 32SC1 for negative numbers
        using var cpuSrc = new Mat(2, 2, MatType.CV_32SC1, new Scalar(-5));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        Cv2.Cuda.CalcAbsSum(gpuSrc, gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // |-5| * 4 pixels = 20
        Assert.Equal(20.0, cpuDst.At<double>(0, 0));
    }

    [Fact]
    public void CalcSqrSumTest()
    {
        VerifyCudaSupport();

        using var cpuSrc = new Mat(2, 2, MatType.CV_8UC1, new Scalar(4));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        Cv2.Cuda.CalcSqrSum(gpuSrc, gpuDst);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // (4^2) * 4 pixels = 16 * 4 = 64
        Assert.Equal(64.0, cpuDst.At<double>(0, 0));
    }

    [Fact]
    public void CalcNormTest()
    {
        VerifyCudaSupport();


        using var cpuSrc = new Mat(2, 2, MatType.CV_8UC1, new Scalar(3));
        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // L1 Norm = sum of absolute values
        Cv2.Cuda.CalcNorm(gpuSrc, gpuDst, NormTypes.L1);

        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        // |3| * 4 pixels = 12
        Assert.Equal(12.0, cpuDst.At<double>(0, 0));
    }

    [Fact]
    public void CalcHistTest()
    {
        VerifyCudaSupport();


        // Create a 10x10 image
        using var cpuSrc = new Mat(10, 10, MatType.CV_8UC1, new Scalar(0));
        // Fill half of it with intensity 50
        cpuSrc[0, 5, 0, 10] = new Mat(5, 10, MatType.CV_8UC1, new Scalar(50));
        // Fill half of it with intensity 100
        cpuSrc[5, 10, 0, 10] = new Mat(5, 10, MatType.CV_8UC1, new Scalar(100));

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuHist = new GpuMat();

        // Act
        Cv2.Cuda.CalcHist(gpuSrc, gpuHist);

        // Assert
        using var cpuHist = new Mat();
        gpuHist.Download(cpuHist);

        Assert.False(cpuHist.Empty());
        // calcHist outputs a 1x256 array of integers (CV_32SC1)
        Assert.Equal(MatType.CV_32SC1, cpuHist.Type());
        Assert.Equal(256, cpuHist.Cols);

        // 50 pixels with intensity 50
        Assert.Equal(50, cpuHist.At<int>(0, 50));
        // 50 pixels with intensity 100
        Assert.Equal(50, cpuHist.At<int>(0, 100));
        // 0 pixels with intensity 99
        Assert.Equal(0, cpuHist.At<int>(0, 99));
    }
    // -----------------------------------------------------------------------
    // cartToPolar
    // -----------------------------------------------------------------------

    [Fact]
    public void CartToPolar_UnitVector_MagnitudeIsOne()
    {
        VerifyCudaSupport();

        // x=1, y=0 → magnitude=1, angle=0°
        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 0f);
        using var mag = new GpuMat();
        using var ang = new GpuMat();

        Cv2.Cuda.CartToPolar(x, y, mag, ang, angleInDegrees: true);

        Assert.Equal(1f, PixelF(mag), 3);
        Assert.Equal(0f, PixelF(ang), 3);
    }

    [Fact]
    public void CartToPolar_DiagonalVector_Magnitude_IsRootTwo()
    {
        VerifyCudaSupport();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 1f);
        using var mag = new GpuMat();
        using var ang = new GpuMat();

        Cv2.Cuda.CartToPolar(x, y, mag, ang);

        Assert.Equal(Math.Sqrt(2f), PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // compare
    // -----------------------------------------------------------------------

    [Fact]
    public void Compare_EqualMatrices_AllPixelsNonZero()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 5f);
        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst = new GpuMat();

        Cv2.Cuda.Compare(src1, src2, dst, CmpTypes.EQ);

        // OpenCV returns 255 for true comparisons
        using var cpu = new Mat();
        dst.Download(cpu);
        Assert.Equal(255, cpu.At<byte>(0, 0));
    }

    [Fact]
    public void Compare_NotEqual_AllPixelsZero()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 5f);
        using var src2 = MakeFloat(Rows, Cols, 6f);
        using var dst = new GpuMat();

        Cv2.Cuda.Compare(src1, src2, dst, CmpTypes.EQ);

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
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 4f);
        using var dst = new GpuMat();

        Cv2.Cuda.Divide(src1, src2, dst);

        Assert.Equal(2.5f, PixelF(dst), 3);
    }

    [Fact]
    public void Divide_WithScale_ReturnsScaledQuotient()
    {
        VerifyCudaSupport();

        // (10 / 2) * 3 = 15
        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 2f);
        using var dst = new GpuMat();

        Cv2.Cuda.Divide(src1, src2, dst, scale: 3.0);

        Assert.Equal(15f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // exp
    // -----------------------------------------------------------------------

    [Fact]
    public void Exp_Zero_ReturnsOne()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 0f);
        using var dst = new GpuMat();

        Cv2.Cuda.Exp(src, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    [Fact]
    public void Exp_One_ReturnsE()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 1f);
        using var dst = new GpuMat();

        Cv2.Cuda.Exp(src, dst);

        Assert.Equal(Math.E, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // log
    // -----------------------------------------------------------------------

    [Fact]
    public void Log_One_ReturnsZero()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 1f);
        using var dst = new GpuMat();

        Cv2.Cuda.Log(src, dst);

        Assert.Equal(0f, PixelF(dst), 3);
    }

    [Fact]
    public void Log_E_ReturnsOne()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, (float) Math.E);
        using var dst = new GpuMat();

        Cv2.Cuda.Log(src, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // lshift / rshift
    // -----------------------------------------------------------------------

    [Fact]
    public void Lshift_ByteMatrix_ShiftsByGivenAmount()
    {
        VerifyCudaSupport();

        // 1 << 3 = 8
        using var src = MakeByte(Rows, Cols, 1);
        using var dst = new GpuMat();

        Cv2.Cuda.Lshift(src, new Vec4i(3, 0, 0, 0), dst);

        Assert.Equal((byte)8, PixelB(dst));
    }

    [Fact]
    public void Rshift_ByteMatrix_ShiftsByGivenAmount()
    {
        VerifyCudaSupport();

        // 16 >> 2 = 4
        using var src = MakeByte(Rows, Cols, 16);
        using var dst = new GpuMat();

        Cv2.Cuda.Rshift(src, new Vec4i(2, 0, 0, 0), dst);

        Assert.Equal((byte)4, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // magnitude
    // -----------------------------------------------------------------------

    [Fact]
    public void Magnitude_SeparatePlanes_Returns345Hypotenuse()
    {
        VerifyCudaSupport();

        // 3-4-5 right triangle
        using var x = MakeFloat(Rows, Cols, 3f);
        using var y = MakeFloat(Rows, Cols, 4f);
        using var mag = new GpuMat();

        Cv2.Cuda.Magnitude(x, y, mag);

        Assert.Equal(5f, PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // magnitudeSqr
    // -----------------------------------------------------------------------

    [Fact]
    public void MagnitudeSqr_SeparatePlanes_ReturnsSquaredMagnitude()
    {
        VerifyCudaSupport();

        // sqrt(3² + 4²) = 5, so sqr = 25
        using var x = MakeFloat(Rows, Cols, 3f);
        using var y = MakeFloat(Rows, Cols, 4f);
        using var mag = new GpuMat();

        Cv2.Cuda.MagnitudeSqr(x, y, mag);

        Assert.Equal(25f, PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // max
    // -----------------------------------------------------------------------

    [Fact]
    public void Max_TwoMatrices_ReturnsLarger()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 7f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.Max(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // min
    // -----------------------------------------------------------------------

    [Fact]
    public void Min_TwoMatrices_ReturnsSmaller()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 7f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.Min(src1, src2, dst);

        Assert.Equal(3f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // multiply
    // -----------------------------------------------------------------------

    [Fact]
    public void Multiply_TwoMatrices_ReturnsProduct()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 6f);
        using var src2 = MakeFloat(Rows, Cols, 7f);
        using var dst = new GpuMat();

        Cv2.Cuda.Multiply(src1, src2, dst);

        Assert.Equal(42f, PixelF(dst), 3);
    }

    [Fact]
    public void Multiply_WithScale_ReturnsScaledProduct()
    {
        VerifyCudaSupport();

        // (2 * 3) * 0.5 = 3
        using var src1 = MakeFloat(Rows, Cols, 2f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.Multiply(src1, src2, dst, scale: 0.5);

        Assert.Equal(3f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // phase
    // -----------------------------------------------------------------------

    [Fact]
    public void Phase_PositiveXZeroY_AngleIsZero()
    {
        VerifyCudaSupport();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 0f);
        using var ang = new GpuMat();

        Cv2.Cuda.Phase(x, y, ang, angleInDegrees: true);

        Assert.Equal(0f, PixelF(ang), 2);
    }

    [Fact]
    public void Phase_EqualXY_AngleIs45Degrees()
    {
        VerifyCudaSupport();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 1f);
        using var ang = new GpuMat();

        Cv2.Cuda.Phase(x, y, ang, angleInDegrees: true);

        Assert.Equal(45f, PixelF(ang), 1);
    }

    // -----------------------------------------------------------------------
    // polarToCart
    // -----------------------------------------------------------------------

    [Fact]
    public void PolarToCart_UnitMagnitudeZeroAngle_XIsOneYIsZero()
    {
        VerifyCudaSupport();

        using var mag = MakeFloat(Rows, Cols, 1f);
        using var ang = MakeFloat(Rows, Cols, 0f);
        using var x = new GpuMat();
        using var y = new GpuMat();

        Cv2.Cuda.PolarToCart(mag, ang, x, y, angleInDegrees: true);

        Assert.Equal(1f, PixelF(x), 3);
        Assert.Equal(0f, PixelF(y), 3);
    }

    // -----------------------------------------------------------------------
    // pow
    // -----------------------------------------------------------------------

    [Fact]
    public void Pow_BaseAndExponent_ReturnsCorrectPower()
    {
        VerifyCudaSupport();

        // 3^4 = 81
        using var src = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.Pow(src, 4.0, dst);

        Assert.Equal(81f, PixelF(dst), 1);
    }

    [Fact]
    public void Pow_ExponentZero_ReturnsOne()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 99f);
        using var dst = new GpuMat();

        Cv2.Cuda.Pow(src, 0.0, dst);

        Assert.Equal(1f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // scaleAdd
    // -----------------------------------------------------------------------

    [Fact]
    public void ScaleAdd_ComputesAlphaSrc1PlusSrc2()
    {
        VerifyCudaSupport();

        // 2.5 * 4 + 3 = 13
        using var src1 = MakeFloat(Rows, Cols, 4f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.ScaleAdd(src1, 2.5, src2, dst);

        Assert.Equal(13f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // sqr
    // -----------------------------------------------------------------------

    [Fact]
    public void Sqr_PositiveValue_ReturnsSquare()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 9f);
        using var dst = new GpuMat();

        Cv2.Cuda.Sqr(src, dst);

        Assert.Equal(81f, PixelF(dst), 2);
    }

    // -----------------------------------------------------------------------
    // sqrt
    // -----------------------------------------------------------------------

    [Fact]
    public void Sqrt_PerfectSquare_ReturnsRoot()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 144f);
        using var dst = new GpuMat();

        Cv2.Cuda.Sqrt(src, dst);

        Assert.Equal(12f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // subtract
    // -----------------------------------------------------------------------

    [Fact]
    public void Subtract_TwoMatrices_ReturnsDifference()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 3f);
        using var dst = new GpuMat();

        Cv2.Cuda.Subtract(src1, src2, dst);

        Assert.Equal(7f, PixelF(dst), 3);
    }

    // -----------------------------------------------------------------------
    // threshold
    // -----------------------------------------------------------------------

    [Fact]
    public void Threshold_BinaryType_PixelsAboveThresholdBecome255()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 200f);
        using var dst = new GpuMat();

        Cv2.Cuda.Threshold(src, dst, thresh: 100.0, maxval: 255.0,
            ThresholdTypes.Binary);

        Assert.Equal(255f, PixelF(dst), 0);
    }

    [Fact]
    public void Threshold_BinaryType_PixelsBelowThresholdBecomeZero()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 50f);
        using var dst = new GpuMat();

        Cv2.Cuda.Threshold(src, dst, thresh: 100.0, maxval: 255.0,
            ThresholdTypes.Binary);

        Assert.Equal(0f, PixelF(dst), 0);
    }

    [Fact]
    public void Threshold_ReturnsThresholdValue()
    {
        VerifyCudaSupport();

        using var src = MakeFloat(Rows, Cols, 50f);
        using var dst = new GpuMat();

        double returned = Cv2.Cuda.Threshold(src, dst,
            thresh: 128.0, maxval: 255.0, ThresholdTypes.Binary);

        Assert.Equal(128.0, returned, 3);
    }

    // -----------------------------------------------------------------------
    // null-argument guard tests (no CUDA needed — pure managed layer)
    // -----------------------------------------------------------------------

    [Fact]
    public void Add_NullSrc1_Throws() =>
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Cuda.Add(null!, new GpuMat(), new GpuMat()));

    [Fact]
    public void Add_NullSrc2_Throws()
    {
        using var src1 = new GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Cuda.Add(src1, null!, new GpuMat()));
    }

    [Fact]
    public void Add_NullDst_Throws()
    {
        using var src1 = new GpuMat();
        using var src2 = new GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Cuda.Add(src1, src2, null!));
    }

    [Fact]
    public void Threshold_NullSrc_Throws() =>
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Cuda.Threshold(null!, new GpuMat(), 0, 255, ThresholdTypes.Binary));

    [Fact]
    public void Threshold_NullDst_Throws()
    {
        using var src = new GpuMat();
        Assert.Throws<ArgumentNullException>(() =>
            Cv2.Cuda.Threshold(src, null!, 0, 255, ThresholdTypes.Binary));
    }

    // -----------------------------------------------------------------------
    // Stream Tests (Verifying actual results, not just execution)
    // -----------------------------------------------------------------------

    [Fact]
    public void Subtract_WithStream_VerifiesResult()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 20f);
        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst = new GpuMat();
        using var stream = new OpenCvSharp.Cuda.Stream();

        Cv2.Cuda.Subtract(src1, src2, dst, stream: stream);

        // Wait for the GPU to finish before checking the result
        stream.WaitForCompletion();

        Assert.Equal(15f, PixelF(dst), 3);
    }

    [Fact]
    public void CartToPolar_WithStream_VerifiesResult()
    {
        VerifyCudaSupport();

        using var x = MakeFloat(Rows, Cols, 3f);
        using var y = MakeFloat(Rows, Cols, 4f);
        using var mag = new GpuMat();
        using var ang = new GpuMat();
        using var stream = new OpenCvSharp.Cuda.Stream();

        Cv2.Cuda.CartToPolar(x, y, mag, ang, stream: stream);
        stream.WaitForCompletion();

        Assert.Equal(5f, PixelF(mag), 2);
    }

    // -----------------------------------------------------------------------
    // Mask Tests
    // -----------------------------------------------------------------------

    [Fact]
    public void Add_WithEmptyMask_LeavesDestinationUntouched()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 4f);
        using var src2 = MakeFloat(Rows, Cols, 6f);
        using var mask = MakeByte(Rows, Cols, 0); // 0 = Do not process
        using var dst = MakeFloat(Rows, Cols, 99f); // Pre-fill with known value

        Cv2.Cuda.Add(src1, src2, dst, mask: mask);

        // Because the mask is 0 everywhere, dst should remain 99f, not 10f
        Assert.Equal(99f, PixelF(dst), 3);
    }

    [Fact]
    public void Add_WithFullMask_UpdatesDestination()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 4f);
        using var src2 = MakeFloat(Rows, Cols, 6f);
        using var mask = MakeByte(Rows, Cols, 255); // 255 = Process everywhere
        using var dst = MakeFloat(Rows, Cols, 99f);

        Cv2.Cuda.Add(src1, src2, dst, mask: mask);

        // Because the mask is 255 everywhere, dst should be updated to 10f
        Assert.Equal(10f, PixelF(dst), 3);
    }

  

    // -----------------------------------------------------------------------
    // In-Place Operation Tests (src == dst)
    // -----------------------------------------------------------------------

    [Fact]
    public void Add_InPlace_UpdatesSourceCorrectly()
    {
        VerifyCudaSupport();

        // We use the same matrix for src1 and dst
        using var srcDst = MakeFloat(Rows, Cols, 5f);
        using var src2 = MakeFloat(Rows, Cols, 2f);

        Cv2.Cuda.Add(srcDst, src2, srcDst);

        Assert.Equal(7f, PixelF(srcDst), 3);
    }

    

    // -----------------------------------------------------------------------
    // Compare Edge Cases
    // -----------------------------------------------------------------------

    [Fact]
    public void Compare_GreaterThan_ReturnsTrueCorrectly()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst = new GpuMat();

        // 10 > 5 is True (255)
        Cv2.Cuda.Compare(src1, src2, dst, CmpTypes.GT);

        Assert.Equal((byte)255, PixelB(dst));
    }

    [Fact]
    public void Compare_LessThan_ReturnsFalseCorrectly()
    {
        VerifyCudaSupport();

        using var src1 = MakeFloat(Rows, Cols, 10f);
        using var src2 = MakeFloat(Rows, Cols, 5f);
        using var dst = new GpuMat();

        // 10 < 5 is False (0)
        Cv2.Cuda.Compare(src1, src2, dst, CmpTypes.LT);

        Assert.Equal((byte)0, PixelB(dst));
    }

    // -----------------------------------------------------------------------
    // Advanced Parameter Tests (dtype, scale, radians vs degrees)
    // -----------------------------------------------------------------------

    [Fact]
    public void Multiply_WithDtype_CastsOutputProperly()
    {
        VerifyCudaSupport();

        // Input is byte (CV_8UC1)
        using var src1 = MakeByte(Rows, Cols, 100);
        using var src2 = MakeByte(Rows, Cols, 2);
        using var dst = new GpuMat();

        // Multiply to 200, but output as Float (MatType.CV_32FC1 == 5)
        Cv2.Cuda.Multiply(src1, src2, dst, dtype: (int)MatType.CV_32FC1);

        Assert.Equal(MatType.CV_32FC1, dst.Type());
        Assert.Equal(200f, PixelF(dst), 2);
    }

    [Fact]
    public void Phase_Radians_ReturnsCorrectAngle()
    {
        VerifyCudaSupport();

        using var x = MakeFloat(Rows, Cols, 1f);
        using var y = MakeFloat(Rows, Cols, 1f);
        using var ang = new GpuMat();

        // angleInDegrees = false -> Should return Pi/4 (approx 0.785)
        Cv2.Cuda.Phase(x, y, ang, angleInDegrees: false);

        Assert.Equal((float)(Math.PI / 4.0), PixelF(ang), 3);
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


