using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using Xunit;

namespace OpenCvSharp.Tests.Cuda.imgproc;

public class CudaHoughLinesDetectorTest : CudaTestBase
{
    [Fact]
    public void HoughLines_DetectTest()
    {
        VerifyCudaSupport();

        // 1. Arrange: Create a 100x100 black image and draw a white horizontal line
        using var cpuSrc = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
        // Line at y=50
        Cv2.Line(cpuSrc, new Point(0, 50), new Point(100, 50), new Scalar(255), 1);

        using var gpuSrc = new GpuMat(); gpuSrc.Upload(cpuSrc);
        using var gpuLines = new GpuMat();

        // Create detector: rho=1, theta=1 degree (PI/180), threshold=50
        float theta = (float)Math.PI / 180.0f;
        using var detector = OpenCvSharp.Cuda.HoughLinesDetector.Create(1.0f, theta, 50);

        // 2. Act
        detector.Detect(gpuSrc, gpuLines);

        // 3. Download and Assert
        using var cpuLines = new Mat();
        gpuLines.Download(cpuLines);

        Assert.False(cpuLines.Empty());

        // Output for CUDA HoughLines is a 1-row (or 1-col) matrix of CV_32FC2 (rho, theta)
        Assert.Equal(MatType.CV_32FC2, cpuLines.Type());

        var linesIndexer = cpuLines.GetGenericIndexer<Vec2f>();
        Vec2f line = linesIndexer[0];

        // For a horizontal line at y=50:
        // rho (distance from origin) should be ~50
        // theta (angle) should be ~PI/2 (1.57 rad)
        Assert.InRange(line.Item0, 48f, 52f);
        Assert.InRange(line.Item1, 1.5f, 1.6f);
    }
}
