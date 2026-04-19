using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace OpenCvSharp.Tests.cuda;

public abstract class CudaTestBase : TestBase
{
    // Cached result so we only parse the build info once
    private static readonly bool IsCudaSupported;
    private static readonly string? SkipReason;
    private static readonly bool IsCudaAvailable;
    static CudaTestBase()
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


            IsCudaSupported = valid;
            if (!valid)
            {
                SkipReason = "OpenCV binary was not compiled with CUDA support.";
                return;
            }
                
            
            int deviceCount = Cv2.GetCudaEnabledDeviceCount();

            IsCudaAvailable = deviceCount != 0;
            if (!IsCudaAvailable)
                SkipReason = "OpenCV binary compiled with CUDA support, but no device found";

        }
        catch (Exception ex)
        {
            IsCudaSupported = false;
            SkipReason = $"Could not load OpenCV native library: {ex.Message}";
        }
    }

    protected void EnsureCuda()
    {
        if (!IsCudaSupported || !IsCudaAvailable)
        {
            Console.WriteLine("skipping");
            // Assert.Skip is a feature of xUnit v3 
            // It stops the test and marks it as skipped.
            Assert.Skip(SkipReason);
        }
        Console.WriteLine("cuda ok");
    }
}
