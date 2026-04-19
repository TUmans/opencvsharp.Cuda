using System;
using System.Collections.Generic;
using System.Text;
using Xunit;

namespace OpenCvSharp.Tests.cuda;

public class OpenCvCudaConfigTest : CudaTestBase
{
    [Fact]
    public void OpenCVBinaryShouldBeCompiledWithCudaSupport()
    {

        EnsureCuda();
       
    }
}
