using OpenCvSharp.Internal;

namespace OpenCvSharp.Cuda
{
    public sealed class Event : DisposableGpuObject
    {
        public Event()
        {
            NativeMethods.HandleException(NativeMethods.cuda_Event_new(out IntPtr p));
            InitSafeHandle(p);
        }

        private void InitSafeHandle(IntPtr p)
        {
            SetSafeHandle(new OpenCvPtrSafeHandle(p, true,
                h => NativeMethods.cuda_Event_delete(h)));
        }

        public void Record(Stream stream = null)
        {
            NativeMethods.HandleException(
                NativeMethods.cuda_Event_record(CvPtr, stream?.CvPtr ?? IntPtr.Zero));
            GC.KeepAlive(this);
            if (stream != null) GC.KeepAlive(stream);
        }

        public void WaitForCompletion()
        {
            NativeMethods.HandleException(NativeMethods.cuda_Event_waitForCompletion(CvPtr));
            GC.KeepAlive(this);
        }

        public bool QueryIfComplete()
        {
            NativeMethods.HandleException(NativeMethods.cuda_Event_queryIfComplete(CvPtr, out int res));
            GC.KeepAlive(this);
            return res != 0;
        }
    }
}
