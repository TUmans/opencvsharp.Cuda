using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public class OpenCvCudaConfigTest : CudaTestBase
{
    [Fact]
    public void OpenCVBinaryShouldBeCompiledWithCudaSupport()
    {

        try
        {
            // 1. Get the raw build information from the native library
            string buildInfo = Cv2.GetBuildInformation();
            Console.WriteLine(buildInfo);

            // 2. Define the marker we are looking for. 
            // In OpenCV's output, it looks like: "NVIDIA CUDA:                   YES (ver 11.x)"
            string searchString = "NVIDIA CUDA:";

            // 3. Find the line containing the CUDA status
            string[] lines = buildInfo.Split(["\n", "\r"], StringSplitOptions.RemoveEmptyEntries);
            string? cudaLine = Array.Find(lines, l => l.Contains(searchString));

            bool valid = true;

            if (string.IsNullOrEmpty(cudaLine))
                valid = false;

            if (valid && !cudaLine.ToUpper().Contains("YES"))
                valid = false;

            if (!valid)
                throw new SkipException("OpenCV binary was not compiled with CUDA support.");



            int deviceCount = Cv2.Cuda.GetCudaEnabledDeviceCount();

            if (deviceCount ==0)
                throw new SkipException("OpenCV binary compiled with CUDA support, but no device found");

        }
        catch (Exception ex)
        {
            throw new SkipException($"Could not load OpenCV native library: {ex.Message}");
        }

    }
}
