using System;
using System.Collections.Generic;
using System.Text;
using OpenCvSharp.Tests;
using Xunit;

namespace OpenCvSharp.Tests.Cuda;

public abstract class CudaTestBase : TestBase
{
    private static int _cudaSupport = -1;

    protected void VerifyCudaSupport()
    {
       if (_cudaSupport == -1)
        {
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

            _cudaSupport = 2; // default
            if (!valid)
            {
                _cudaSupport = 0; // set no cuda build
            }
            if (Cv2.Cuda.GetCudaEnabledDeviceCount() == 0)
            {
                _cudaSupport = 1; // set no cuda device found
            }
        }
       
        if (_cudaSupport == 0)
            Assert.Skip("OpenCV binary was not compiled with CUDA support.");
        if (_cudaSupport == 1)
            Assert.Skip("No CUDA device available.");
        if (_cudaSupport == 2)
            return;
    }
}
