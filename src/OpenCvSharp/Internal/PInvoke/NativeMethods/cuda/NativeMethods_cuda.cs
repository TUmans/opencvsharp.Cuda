#if ENABLED_CUDA

using System;
using System.Runtime.InteropServices;
using System.Text;
using OpenCvSharp.Cuda;

#pragma warning disable 1591

namespace OpenCvSharp.Internal
{
    // ReSharper disable InconsistentNaming

    static partial class NativeMethods
    {
        #region Device
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_getCudaEnabledDeviceCount(out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_setDevice(int device);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_getDevice(out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_resetDevice();

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_deviceSupports(int feature_set, out int returnValue);

        // TargetArchs
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_builtWith(int feature_set, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_has(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasPtx(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasBin(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasEqualOrLessPtx(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasEqualOrGreater(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasEqualOrGreaterPtx(int major, int minor, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_TargetArchs_hasEqualOrGreaterBin(int major, int minor, out int returnValue);

        // DeviceInfo
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_new1(out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_new2(int deviceId, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_name(
            IntPtr obj, [MarshalAs(UnmanagedType.LPStr)] StringBuilder buf, int bufLength);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_majorVersion(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_minorVersion(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_multiProcessorCount(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_sharedMemPerBlock(IntPtr obj, out ulong returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_queryMemory(
            IntPtr obj, out ulong totalMemory, out ulong freeMemory);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_freeMemory(IntPtr obj, out ulong returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_totalMemory(IntPtr obj, out ulong returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_supports(IntPtr obj, int featureSet, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_isCompatible(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_deviceID(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_DeviceInfo_canMapHostMemory(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_printCudaDeviceInfo(int device);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_printShortCudaDeviceInfo(int device);
        #endregion

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_registerPageLocked(IntPtr m);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_unregisterPageLocked(IntPtr m);

        #region Stream

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_new1(out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_new2(IntPtr s, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_opAssign(IntPtr left, IntPtr right);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_queryIfComplete(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_waitForCompletion(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueDownload_CudaMem(IntPtr obj, IntPtr src, IntPtr dst);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueDownload_Mat(IntPtr obj, IntPtr src, IntPtr dst);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueUpload_CudaMem(IntPtr obj, IntPtr src, IntPtr dst);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueUpload_Mat(IntPtr obj, IntPtr src, IntPtr dst);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueCopy(IntPtr obj, IntPtr src, IntPtr dst);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueMemSet(IntPtr obj, IntPtr src, Scalar val);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueMemSet_WithMask(IntPtr obj, IntPtr src, Scalar val, IntPtr mask);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueConvert(
            IntPtr obj, IntPtr src, IntPtr dst, int dtype, double a, double b);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_enqueueHostCallback(
            IntPtr obj, IntPtr callback, IntPtr userData);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_Null(out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_Stream_bool(IntPtr obj, out int returnValue);

        #endregion

        #region CascadeClassifier_GPU

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_new1(out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_new2(string filename, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_empty(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_load(IntPtr obj, string filename, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_release(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_detectMultiScale1(IntPtr obj,
            IntPtr image, IntPtr objectsBuf, double scaleFactor, int minNeighbors, Size minSize, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_detectMultiScale2(IntPtr obj,
            IntPtr image, IntPtr objectsBuf, Size maxObjectSize, Size minSize, double scaleFactor,
            int minNeighbors, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_findLargestObject_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_findLargestObject_set(IntPtr obj, int value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_visualizeInPlace_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_visualizeInPlace_set(IntPtr obj, int value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_CascadeClassifier_GPU_getClassifierSize(IntPtr obj, out Size returnValue);

        #endregion

        #region HogDescriptor
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_sizeof(out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_new(Size win_size, Size block_size, Size block_stride, Size cell_size,
            int nbins, double winSigma, double threshold_L2Hys, bool gamma_correction, int nlevels, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_getDescriptorSize(IntPtr obj, out ulong returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_getBlockHistogramSize(IntPtr obj, out ulong returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_checkDetectorSize(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_getWinSigma(IntPtr obj, out double returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_setSVMDetector(IntPtr obj, IntPtr svmdetector);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_detect(IntPtr obj, IntPtr img, IntPtr found_locations, double hit_threshold, Size win_stride, Size padding);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_detectMultiScale(IntPtr obj, IntPtr img, IntPtr found_locations,
                                                   double hit_threshold, Size win_stride, Size padding, double scale, int group_threshold);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_getDescriptors(IntPtr obj, IntPtr img, Size win_stride, IntPtr descriptors, [MarshalAs(UnmanagedType.I4)] DescriptorFormat descr_format);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_win_size_get(IntPtr obj, out Size returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_block_size_get(IntPtr obj, out Size returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_block_stride_get(IntPtr obj, out Size returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_cell_size_get(IntPtr obj, out Size returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_nbins_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_win_sigma_get(IntPtr obj, out double returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_threshold_L2hys_get(IntPtr obj, out double returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_nlevels_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_gamma_correction_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_win_size_set(IntPtr obj, Size value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_block_size_set(IntPtr obj, Size value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_block_stride_set(IntPtr obj, Size value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_cell_size_set(IntPtr obj, Size value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_nbins_set(IntPtr obj, int value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_win_sigma_set(IntPtr obj, double value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_threshold_L2hys_set(IntPtr obj, double value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_nlevels_set(IntPtr obj, int value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus HOGDescriptor_gamma_correction_set(IntPtr obj, int value);
        #endregion

        #region MOG_GPU
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_new(int nmixtures, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_initialize(
            IntPtr obj, Size frameSize, int frameType);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_operator(
            IntPtr obj, IntPtr frame, IntPtr fgmask, float learningRate, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_getBackgroundImage(
            IntPtr obj, IntPtr backgroundImage, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_release(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_history(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_varThreshold(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_backgroundRatio(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG_GPU_noiseSigma(IntPtr obj, out IntPtr returnValue);


        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_new(int nmixtures, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_initialize(
            IntPtr obj, Size frameSize, int frameType);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_operator(
            IntPtr obj, IntPtr frame, IntPtr fgmask, float learningRate, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_getBackgroundImage(
            IntPtr obj, IntPtr backgroundImage, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_release(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_history(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_varThreshold(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_backgroundRatio(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_varThresholdGen(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_fVarInit(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_fVarMin(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_fVarMax(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_fCT(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_bShadowDetection_get(IntPtr obj, out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_bShadowDetection_set(IntPtr obj, int value);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_nShadowDetection(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_MOG2_GPU_fTau(IntPtr obj, out IntPtr returnValue);

        #endregion

        #region StereoBM_GPU
        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_new1(out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_new2(int preset, int ndisparities, int winSize, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_delete(IntPtr obj);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_run1(IntPtr obj, IntPtr left, IntPtr right, IntPtr disparity);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_run2(IntPtr obj, IntPtr left, IntPtr right, IntPtr disparity, IntPtr stream);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_checkIfGpuCallReasonable(out int returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_preset(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_ndisp(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_winSize(IntPtr obj, out IntPtr returnValue);

        [DllImport(DllExtern, CallingConvention = CallingConvention.Cdecl, ExactSpelling = true)]
        public static extern ExceptionStatus cuda_StereoBM_GPU_avergeTexThreshold(IntPtr obj, out IntPtr returnValue);
        #endregion
    }
}

#endif
