using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Cuda;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class CudaBgsegmTest : CudaTestBase
{
    [Fact]
    public void BackgroundSubtractorMOG2()
    {
        VerifyCudaSupport();

        try
        {
            using var mog2 = OpenCvSharp.Cuda.BackgroundSubtractorMOG2.Create(detectShadows: false);
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // Frame 1: Train background
            gpuFrame.Upload(cpuFrame);
            mog2.Apply(gpuFrame, gpuFgMask, learningRate: 1.0);

            // Frame 2: Introduce foreground
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255), -1);
            gpuFrame.Upload(cpuFrame);
            mog2.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty());
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10)); // Background
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50)); // Foreground
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }

    [Fact]
    public void BackgroundSubtractorMOG()
    {
        VerifyCudaSupport();

        try
        {
            using var mog = OpenCvSharp.Cuda.BackgroundSubtractorMOG.Create();
            using var cpuFrame = new Mat(100, 100, MatType.CV_8UC1, new Scalar(0));
            using var gpuFrame = new GpuMat();
            using var gpuFgMask = new GpuMat();

            // Frame 1: Train background
            gpuFrame.Upload(cpuFrame);
            mog.Apply(gpuFrame, gpuFgMask, learningRate: 1.0);

            // Frame 2: Introduce foreground
            Cv2.Rectangle(cpuFrame, new Rect(40, 40, 20, 20), new Scalar(255), -1);
            gpuFrame.Upload(cpuFrame);
            mog.Apply(gpuFrame, gpuFgMask, learningRate: 0.0);

            // Assert
            using var cpuFgMask = new Mat();
            gpuFgMask.Download(cpuFgMask);

            Assert.False(cpuFgMask.Empty());
            Assert.Equal(0, cpuFgMask.At<byte>(10, 10)); // Background
            Assert.Equal(255, cpuFgMask.At<byte>(50, 50)); // Foreground
        }
        catch (OpenCVException ex) when (ex.Message.Contains("disabled") || ex.Message.Contains("Not Implemented"))
        {
            Assert.Skip("The called functionality is disabled for current build or platform");
        }
    }
}

