using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using OpenCvSharp.Modules.core.Enum;
using OpenCvSharp.Tests.Cuda;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaImgProcTest : CudaTestBase
{
    [Fact]
    public void AlphaCompTest()
    {
        VerifyCudaSupport();

        // AlphaComp REQUIRES 4-channel images (RGBA/BGRA) (CV_8UC4 or CV_16UC4 or CV_32FC4)
        // Foreground: value 100 on first channel, 255 alpha (fully opaque)
        using var cpuImg1 = new Mat(2, 2, MatType.CV_8UC4, new Scalar(100, 0, 0, 255));
        // Background: value 50 on first channel, 255 alpha
        using var cpuImg2 = new Mat(2, 2, MatType.CV_8UC4, new Scalar(50, 0, 0, 255));

        using var gpuImg1 = new GpuMat();
        using var gpuImg2 = new GpuMat();
        gpuImg1.Upload(cpuImg1);
        gpuImg2.Upload(cpuImg2);

        // Act: Place Foreground OVER Background
        using var gpuDst = new GpuMat();
        Cv2.Cuda.AlphaComp(gpuImg1, gpuImg2, gpuDst, AlphaCompTypes.Over);

        // Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(MatType.CV_8UC4, cpuDst.Type());

        // Because the foreground was fully opaque (alpha=255), ALPHA_OVER 
        // should result in the exact foreground image.
        Vec4b pixel = cpuDst.At<Vec4b>(0, 0);
        Assert.Equal(100, pixel.Item0); // First channel
        Assert.Equal(255, pixel.Item3); // Alpha channel
    }

    [Fact]
    public void BilateralFilterTest()
    {
        VerifyCudaSupport();

        // Bilateral Filter usually operates on CV_8UC1 or CV_8UC3
        // Create a completely flat, gray image.
        using var cpuSrc = new Mat(10, 10, MatType.CV_8UC1, new Scalar(128));
        using var gpuSrc = new GpuMat();
        gpuSrc.Upload(cpuSrc);

        // Act
        using var gpuDst = new GpuMat();
        Cv2.Cuda.BilateralFilter(
            gpuSrc, gpuDst,
            kernelSize: 5,
            sigmaColor: 50.0f,
            sigmaSpatial: 50.0f);

        // Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());
        Assert.Equal(cpuSrc.Rows, cpuDst.Rows);
        Assert.Equal(cpuSrc.Cols, cpuDst.Cols);
        Assert.Equal(MatType.CV_8UC1, cpuDst.Type());

        // Since the input image was a solid color, filtering it should do absolutely 
        // nothing to the color. It should remain 128.
        Assert.Equal(128, cpuDst.At<byte>(5, 5));
    }

    [Fact]
    public void BlendLinearTest()
    {
        VerifyCudaSupport();

        // Arrange: 
        using var cpuImg1 = new Mat(5, 5, MatType.CV_8UC1, new Scalar(100));
        using var cpuImg2 = new Mat(5, 5, MatType.CV_8UC1, new Scalar(50));

        using var cpuW1 = new Mat(5, 5, MatType.CV_32FC1, new Scalar(0.8f));
        using var cpuW2 = new Mat(5, 5, MatType.CV_32FC1, new Scalar(0.2f));

        using var gpuImg1 = new GpuMat(); gpuImg1.Upload(cpuImg1);
        using var gpuImg2 = new GpuMat(); gpuImg2.Upload(cpuImg2);
        using var gpuW1 = new GpuMat(); gpuW1.Upload(cpuW1);
        using var gpuW2 = new GpuMat(); gpuW2.Upload(cpuW2);
        using var gpuDst = new GpuMat();

        // Act
        Cv2.Cuda.BlendLinear(gpuImg1, gpuImg2, gpuW1, gpuW2, gpuDst);

        // Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Assert.False(cpuDst.Empty());

        // We expect ~90, but GPU Fused-Multiply-Add (FMA) might truncate it to 89.
        // We allow a tolerance of +/- 1.
        byte pixelValue = cpuDst.At<byte>(0, 0);
        Assert.InRange(pixelValue, 89, 91);
    }

    [Fact]
    public void CreateCannyEdgeDetector()
    {
        VerifyCudaSupport();

        // 1. Arrange
        // Create a 100x100 completely black image
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));

        // Make the left half completely white.
        // This creates a perfectly sharp vertical edge at column 50.
        Cv2.Rectangle(cpuSrc, new Rect(0, 0, 50, 100), new Scalar(255), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuEdges = new GpuMat();

        // Create Canny detector with standard thresholds
        using var canny = OpenCvSharp.Cuda.CannyEdgeDetector.Create(lowThresh: 50.0, highThresh: 100.0);

        // 2. Act
        canny.Detect(gpuSrc, gpuEdges);

        // 3. Download and Assert
        using var cpuEdges = new Mat();
        gpuEdges.Download(cpuEdges);    

        Assert.False(cpuEdges.Empty());
        Assert.Equal(MatType.CV_8UC1, cpuEdges.Type());

        // A pixel far away from the edge should be 0 (No Edge)
        Assert.Equal(0, cpuEdges.At<byte>(50, 10));
        Assert.Equal(0, cpuEdges.At<byte>(50, 90));

        int edgeCount = 0;

        for (int y = 0; y < 100; y++)
        {
            for (int x = 48; x <= 52; x++)
            {
                if (cpuEdges.At<byte>(y, x) > 0)
                    edgeCount++;
            }
        }

        Assert.True(edgeCount > 50, "Expected a continuous vertical edge");
    }

    [Fact]
    public void CudaCLAHE()
    {
        VerifyCudaSupport();

        // 1. Arrange
        // Create a 100x100 image with terrible contrast.
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(100)); // Dark Gray

        for (int y = 0; y < 100; y++)
        {
            for (int x = 0; x < 100; x++)
            {
                cpuSrc.Set(y, x, (byte)(x)); // horizontal gradient 0–99
            }
        }

        // Draw a tiny square that is barely visibly lighter (150)
        Cv2.Rectangle(cpuSrc, new Rect(40, 40, 20, 20), new Scalar(150), -1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuDst = new GpuMat();

        // Create CLAHE algorithm
        using var clahe = OpenCvSharp.Cuda.CLAHE.Create(clipLimit: 40.0, tileGridSize: new Size(8, 8));

        // 2. Act
        clahe.Apply(gpuSrc, gpuDst);

        // 3. Download and Assert
        using var cpuDst = new Mat();
        gpuDst.Download(cpuDst);

        Cv2.MinMaxLoc(cpuSrc, out double minBefore, out double maxBefore);
        Cv2.MinMaxLoc(cpuDst, out double minAfter, out double maxAfter);

        double rangeBefore = maxBefore - minBefore;
        double rangeAfter = maxAfter - minAfter;

        Assert.True(rangeAfter > rangeBefore,
            $"Expected expanded dynamic range, before={rangeBefore}, after={rangeAfter}");
    }
}



