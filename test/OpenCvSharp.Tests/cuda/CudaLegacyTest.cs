using Xunit;
using OpenCvSharp.Cuda;

namespace OpenCvSharp.Tests.Cuda;


public class CudaLegacyTest : CudaTestBase
{
    [Fact]
    public void CalcOpticalFlowBMTest()
    {
        VerifyCudaSupport();

        // Optical flow BM needs two CV_8UC1 images
        using var cpuPrev = new Mat(64, 64, MatType.CV_8UC1, new Scalar(0));
        using var cpuCurr = new Mat(64, 64, MatType.CV_8UC1, new Scalar(0));

        // Draw a white square and simulate motion
        Cv2.Rectangle(cpuPrev, new Rect(10, 10, 15, 15), new Scalar(255), -1);
        Cv2.Rectangle(cpuCurr, new Rect(12, 12, 15, 15), new Scalar(255), -1); // Moved +2, +2

        using var gpuPrev = new GpuMat(); gpuPrev.Upload(cpuPrev);
        using var gpuCurr = new GpuMat(); gpuCurr.Upload(cpuCurr);

        using var velx = new GpuMat();
        using var vely = new GpuMat();
        using var buf = new GpuMat();

        // Act
        Cv2.Cuda.CalcOpticalFlowBM(
            gpuPrev, gpuCurr,
            blockSize: new Size(15, 15),
            shiftSize: new Size(1, 1),
            maxRange: new Size(15, 15),
            usePrevious: false,
            velx, vely, buf);

        // Assert
        Assert.False(velx.Empty(), "X velocity matrix should not be empty.");
        Assert.False(vely.Empty(), "Y velocity matrix should not be empty.");

        // The size of the output velocity fields depends on the block size and shift size,
        // but we can confidently assert that execution succeeded and outputs were allocated.
    }
}
