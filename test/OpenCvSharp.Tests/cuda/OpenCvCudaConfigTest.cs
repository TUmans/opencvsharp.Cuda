using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace OpenCvSharp.Tests.cuda;

public class OpenCvCudaConfigTest
{
    [Fact]
    public void OpenCVBinaryShouldBeCompiledWithCudaSupport()
    {
        // 1. Get the raw build information from the native library
        string buildInfo = Cv2.GetBuildInformation();
        Console.WriteLine(buildInfo);

       // 2. Define the marker we are looking for. 
       // In OpenCV's output, it looks like: "NVIDIA CUDA:                   YES (ver 11.x)"
       string searchString = "NVIDIA CUDA:";

        // 3. Find the line containing the CUDA status
        string[] lines = buildInfo.Split([ "\n", "\r" ], StringSplitOptions.RemoveEmptyEntries);
        string? cudaLine = Array.Find(lines, l => l.Contains(searchString));

        // 4. Assertions
        Assert.NotNull(cudaLine); // Ensure the line exists at all

        // We check if the line contains "YES". 
        // If it says "NO", this test will fail.
        Assert.NotEmpty(cudaLine);
        Assert.Contains("YES", cudaLine.ToUpper());

        // Optional: Print the specific version found to the test output
        Console.WriteLine($"Found CUDA Support: {cudaLine.Trim()}");
    }
}
