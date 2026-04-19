#pragma once

#ifdef ENABLED_CUDA

#include "include_opencv.h"
#include <opencv2/core/cuda.hpp>

#pragma region Init and Disposal

CVAPI(void) cuda_GpuMat_delete(cv::cuda::GpuMat *obj)
{
    delete obj;
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new1()
{
    return new cv::cuda::GpuMat();
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new2(int rows, int cols, int type)
{
    return new cv::cuda::GpuMat(rows, cols, type);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new3(int rows, int cols, int type, void *data, uint64_t step)
{
    return new cv::cuda::GpuMat(rows, cols, type, data, static_cast<size_t>(step));
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new4(cv::Mat *mat)
{
    return new cv::cuda::GpuMat(*mat);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new5(cv::cuda::GpuMat *gpumat)
{
    return new cv::cuda::GpuMat(*gpumat);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new6(CvSize size, int type)
{
    return new cv::cuda::GpuMat(cv::Size(size.width, size.height), type);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new7(CvSize size, int type, void *data, uint64_t step)
{
    return new cv::cuda::GpuMat(cv::Size(size.width, size.height), type, data, static_cast<size_t>(step));
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new8(int rows, int cols, int type, CvScalar s)
{
    return new cv::cuda::GpuMat(rows, cols, type, cv::Scalar(s.val[0], s.val[1], s.val[2], s.val[3]));
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new9(cv::cuda::GpuMat *m, CvSlice rowRange, CvSlice colRange)
{
    return new cv::cuda::GpuMat(*m, cv::Range(rowRange.start_index, rowRange.end_index), cv::Range(colRange.start_index, colRange.end_index));
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new10(cv::cuda::GpuMat *m, CvRect roi)
{
    return new cv::cuda::GpuMat(*m, cv::Rect(roi.x, roi.y, roi.width, roi.height));
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_new11(CvSize size, int type, CvScalar s)
{
    return new cv::cuda::GpuMat(cv::Size(size.width, size.height), type, cv::Scalar(s.val[0], s.val[1], s.val[2], s.val[3]));
}
#pragma endregion

#pragma region Fields
CVAPI(int) cuda_GpuMat_flags(cv::cuda::GpuMat *obj)
{
    return obj->flags;
}
CVAPI(int) cuda_GpuMat_rows(cv::cuda::GpuMat *obj)
{
    return obj->rows;
}
CVAPI(int) cuda_GpuMat_cols(cv::cuda::GpuMat *obj)
{
    return obj->cols;
}
CVAPI(uint64_t) cuda_GpuMat_step(cv::cuda::GpuMat *obj)
{
    return static_cast<uint64_t>(obj->step);
}
CVAPI(uchar *) cuda_GpuMat_data(cv::cuda::GpuMat *obj)
{
    return obj->data;
}
CVAPI(int *) cuda_GpuMat_refcount(cv::cuda::GpuMat *obj)
{
    return obj->refcount;
}
CVAPI(uchar *) cuda_GpuMat_datastart(cv::cuda::GpuMat *obj)
{
    return obj->datastart;
}
CVAPI(const uchar *) cuda_GpuMat_dataend(cv::cuda::GpuMat *obj)
{
    return obj->dataend;
}
#pragma endregion

#pragma region Operators
CVAPI(void) cuda_GpuMat_opAssign(cv::cuda::GpuMat *left, cv::cuda::GpuMat *right)
{
    *left = *right;
}

CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_opRange1(cv::cuda::GpuMat *src, CvRect roi)
{
    cv::cuda::GpuMat gm = (*src)(cv::Rect(roi.x, roi.y, roi.width, roi.height));
    return new cv::cuda::GpuMat(gm);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_opRange2(cv::cuda::GpuMat *src, CvSlice rowRange, CvSlice colRange)
{
    cv::cuda::GpuMat gm = (*src)(cv::Range(rowRange.start_index, rowRange.end_index), cv::Range(colRange.start_index, colRange.end_index));
    return new cv::cuda::GpuMat(gm);
}

CVAPI(cv::Mat *) cuda_GpuMat_opToMat(cv::cuda::GpuMat *src)
{
    cv::Mat m(*src);
    return new cv::Mat(m);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_opToGpuMat(cv::Mat *src)
{
    cv::cuda::GpuMat gm(*src);
    return new cv::cuda::GpuMat(gm);
}
#pragma endregion

#pragma region Methods

CVAPI(void) cuda_GpuMat_upload(cv::cuda::GpuMat *obj, cv::Mat *m)
{
    obj->upload(*m);
}

CVAPI(void) cuda_GpuMat_download(cv::cuda::GpuMat *obj, cv::Mat *m)
{
    obj->download(*m);
}

CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_row(cv::cuda::GpuMat *obj, int y)
{
    cv::cuda::GpuMat ret = obj->row(y);
    return new cv::cuda::GpuMat(ret);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_col(cv::cuda::GpuMat *obj, int x)
{
    cv::cuda::GpuMat ret = obj->col(x);
    return new cv::cuda::GpuMat(ret);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_rowRange(cv::cuda::GpuMat *obj, int startrow, int endrow)
{
    cv::cuda::GpuMat ret = obj->rowRange(startrow, endrow);
    return new cv::cuda::GpuMat(ret);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_colRange(cv::cuda::GpuMat *obj, int startcol, int endcol)
{
    cv::cuda::GpuMat ret = obj->colRange(startcol, endcol);
    return new cv::cuda::GpuMat(ret);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_clone(cv::cuda::GpuMat *obj)
{
    cv::cuda::GpuMat ret = obj->clone();
    return new cv::cuda::GpuMat(ret);
}
CVAPI(void) cuda_GpuMat_copyTo1(cv::cuda::GpuMat *obj, cv::cuda::GpuMat *m)
{
    obj->copyTo(*m);
}
CVAPI(void) cuda_GpuMat_copyTo2(cv::cuda::GpuMat *obj, cv::cuda::GpuMat *m, cv::cuda::GpuMat *mask)
{
    obj->copyTo(*m, *mask);
}
CVAPI(void) cuda_GpuMat_convertTo(cv::cuda::GpuMat *obj, cv::cuda::GpuMat *m, int rtype, double alpha, double beta)
{
    obj->convertTo(*m, rtype, alpha, beta);
}
CVAPI(void) cuda_GpuMat_assignTo(cv::cuda::GpuMat *obj, cv::cuda::GpuMat *m, int type)
{
    obj->assignTo(*m, type);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_setTo(cv::cuda::GpuMat *obj, CvScalar s, cv::cuda::GpuMat *mask)
{
    cv::Scalar scalar(s.val[0], s.val[1], s.val[2], s.val[3]);
    cv::cuda::GpuMat gm = (mask == nullptr) ? obj->setTo(scalar) : obj->setTo(scalar, *mask);
    return new cv::cuda::GpuMat(gm);
}
CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_reshape(cv::cuda::GpuMat *obj, int cn, int rows)
{
    cv::cuda::GpuMat gm = obj->reshape(cn, rows);
    return new cv::cuda::GpuMat(gm);
}

CVAPI(void) cuda_GpuMat_create1(cv::cuda::GpuMat *obj, int rows, int cols, int type)
{
    obj->create(rows, cols, type);
}
CVAPI(void) cuda_GpuMat_create2(cv::cuda::GpuMat *obj, CvSize size, int type)
{
    obj->create(cv::Size(size.width, size.height), type);
}
CVAPI(void) cuda_GpuMat_release(cv::cuda::GpuMat *obj)
{
    obj->release();
}
CVAPI(void) cuda_GpuMat_swap(cv::cuda::GpuMat *obj, cv::cuda::GpuMat *mat)
{
    obj->swap(*mat);
}

CVAPI(void) cuda_GpuMat_locateROI(cv::cuda::GpuMat *obj, CvSize *wholeSize, CvPoint *ofs)
{
    cv::Size _wholeSize;
    cv::Point _ofs;
    obj->locateROI(_wholeSize, _ofs);
    wholeSize->width = _wholeSize.width;
    wholeSize->height = _wholeSize.height;
    ofs->x = _ofs.x;
    ofs->y = _ofs.y;
}

CVAPI(cv::cuda::GpuMat *) cuda_GpuMat_adjustROI(cv::cuda::GpuMat *obj, int dtop, int dbottom, int dleft, int dright)
{
    cv::cuda::GpuMat gm = obj->adjustROI(dtop, dbottom, dleft, dright);
    return new cv::cuda::GpuMat(gm);
}

CVAPI(int) cuda_GpuMat_isContinuous(cv::cuda::GpuMat *obj)
{
    return obj->isContinuous() ? 1 : 0;
}

CVAPI(uint64_t) cuda_GpuMat_elemSize(cv::cuda::GpuMat *obj)
{
    return static_cast<uint64_t>(obj->elemSize());
}
CVAPI(uint64_t) cuda_GpuMat_elemSize1(cv::cuda::GpuMat *obj)
{
    return static_cast<uint64_t>(obj->elemSize1());
}

CVAPI(int) cuda_GpuMat_type(cv::cuda::GpuMat *obj)
{
    return obj->type();
}
CVAPI(int) cuda_GpuMat_depth(cv::cuda::GpuMat *obj)
{
    return obj->depth();
}
CVAPI(int) cuda_GpuMat_channels(cv::cuda::GpuMat *obj)
{
    return obj->channels();
}
CVAPI(uint64_t) cuda_GpuMat_step1(cv::cuda::GpuMat *obj)
{
    return static_cast<uint64_t>(obj->step1());
}
CVAPI(MyCvSize) cuda_GpuMat_size(cv::cuda::GpuMat *obj)
{
    return c(obj->size());
}
CVAPI(int) cuda_GpuMat_empty(cv::cuda::GpuMat *obj)
{
    return obj->empty() ? 1 : 0;
}

CVAPI(const uchar *) cuda_GpuMat_ptr(const cv::cuda::GpuMat *obj, int y)
{
    return obj->ptr(y);
}

#pragma endregion

//! Creates continuous GPU matrix
CVAPI(void) cuda_createContinuous1(int rows, int cols, int type, cv::cuda::GpuMat *gm)
{
    cv::cuda::createContinuous(rows, cols, type, *gm);
}
CVAPI(cv::cuda::GpuMat *) cuda_createContinuous2(int rows, int cols, int type)
{
    cv::cuda::GpuMat gm = cv::cuda::createContinuous(rows, cols, type);
    return new cv::cuda::GpuMat(gm);
}

//! Ensures that size of the given matrix is not less than (rows, cols) size
//! and matrix type is match specified one too
CVAPI(void) cuda_ensureSizeIsEnough(int rows, int cols, int type, cv::cuda::GpuMat *m)
{
    cv::cuda::ensureSizeIsEnough(rows, cols, type, *m);
}


#endif
