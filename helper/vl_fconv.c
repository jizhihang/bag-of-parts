/** file: spconv.c */

#include <toolbox/mexutils.h>
#include <vl/generic.h>

#include "blas.h"
#include <math.h>
#include <string.h>


enum {
  IN_IMAGE = 0,
  IN_FILTER
} ;

enum {
  OUT_RESULT = 0
} ;

VL_INLINE void shiftdim_kij(float * B, float const * A, vl_size M, vl_size N, vl_size K)
{
  int i, j, k ;
  for (i = 0 ; i < M ; i++) {
    for (j = 0 ; j < N ; j++) {
      for (k = 0 ; k < K ; k++) {
        B[k + i*K + j*(K*M)] = A[i + j*M + k*(M*N)] ;
      }
    }
  }
}

VL_INLINE void shiftdim_ikj(float * B, float const * A, vl_size M, vl_size N, vl_size K)
{
  int i, j, k ;
  for (i = 0 ; i < M ; i++) {
    for (j = 0 ; j < N ; j++) {
      for (k = 0 ; k < K ; k++) {
        B[i + k*M + j*(K*M)] = A[i + j*M + k*(M*N)] ;
      }
    }
  }
}

VL_INLINE void shiftdim_kji(float * B, float const * A, vl_size M, vl_size N, vl_size K)
{
  int i, j, k ;
  for (i = 0 ; i < M ; i++) {
    for (j = 0 ; j < N ; j++) {
      for (k = 0 ; k < K ; k++) {
        B[k + j*K + i*(K*N)] = A[i + j*M + k*(M*N)] ;
      }
    }
  }
}

void
mexFunction (int nout, mxArray * out [],
             int nin, const mxArray * in [])
{
  mxArray const * image_array = IN(IMAGE) ;
  mxArray const * filter_array = IN(FILTER) ;
  mwSize dimension ;
  mwSize filterDimension ;
  mwSize imageWidth, imageHeight, filterWidth, filterHeight ;
  int i,j ;

  if (mxIsSparse(image_array)) {
    mexErrMsgTxt("IMAGE is not dense.") ;
  }
  if (mxGetNumberOfDimensions(image_array) > 3) {
    mexErrMsgTxt("IMAGE is a 2-D or 3-D array.") ;
  }
  if (mxGetClassID(image_array) != mxSINGLE_CLASS) {
    mexErrMsgTxt("IMAGE is not SINGLE.") ;
  }

  if (mxIsSparse(filter_array)) {
    mexErrMsgTxt("FILTER is not dense.") ;
  }
  if (mxGetNumberOfDimensions(filter_array) > 3) {
    mexErrMsgTxt("FILTER is a 2-D or 3-D array.") ;
  }
  if (mxGetClassID(filter_array) != mxSINGLE_CLASS) {
    mexErrMsgTxt("FILTER is not SINGLE.") ;
  }

  imageHeight = mxGetDimensions(image_array)[0] ;
  imageWidth = mxGetDimensions(image_array)[1] ;
  if (mxGetNumberOfDimensions(image_array) == 2) {
    dimension = 1 ;
  } else {
    dimension = mxGetDimensions(image_array)[2] ;
  }

  if (mxGetNumberOfDimensions(filter_array) == 2) {
    filterDimension = 1 ;
  } else {
      filterDimension = mxGetDimensions(filter_array)[2] ;
  }

  filterHeight = mxGetDimensions(filter_array)[0] ;
  filterWidth = mxGetDimensions(filter_array)[1] ;
  if (dimension != filterDimension ||
      imageHeight < filterHeight ||
      imageWidth < filterWidth) {
    vlmxError(vlmxErrInvalidArgument, "FILTER dimensions do not match IMAGE") ;
  }

  OUT(RESULT) = mxCreateNumericMatrix(imageHeight-filterHeight+1,
                                      imageWidth-filterWidth+1,
                                      mxSINGLE_CLASS,
                                      mxREAL) ;

  float * A = vl_malloc(sizeof(float) * imageHeight*imageWidth*dimension) ;
  float * B = vl_malloc(sizeof(float) * filterHeight*filterWidth*dimension) ;
  float * C = mxGetData(OUT(RESULT)) ;

#if 0
  /*
   A = [imageHeigth*dimension x imageWidth]
   B = [filterHeight*dimension x filterWidth]
   C = [imageHeight-filterHeight+1 x imageWidth-filterWidth+1]
   */
  shiftdim_kij(A, mxGetData(image_array), imageHeight, imageWidth, dimension) ;
  shiftdim_kij(B, mxGetData(filter_array), filterHeight, filterWidth, dimension) ;
  for (i = 0 ; i < imageHeight - filterHeight + 1 ; ++i) {
    for (j = 0 ; j < filterWidth ; ++j) {
      char trans = 't';
      float one = 1;
      ptrdiff_t lda = dimension * imageHeight ;
      ptrdiff_t m = dimension * filterHeight ;
      ptrdiff_t n = imageWidth - filterWidth + 1 ;
      ptrdiff_t incx = 1 ;
      ptrdiff_t incy = imageHeight - filterHeight + 1 ;
      sgemv (&trans,
             &m, &n, /* A dimension */
             &one, /* alpha = 1 */
             A + dimension * (i + j * imageHeight), /* A with offset */
             &lda, /* column stride of A (we use subset) */
             B + j * m, /* B with offset */
             &incx, /* B stride = 1 */
             &one, /* beta = 1 */
             C + i, /* C with offset */
             &incy /* C stride */
             ) ;
    }
  }
#else
  /*
   A = [imageHeigth x dimension*imageWidth]
   B = [dimension*filterWidth x filterHeight]
   C = [imageHeight-filterHeight+1 x imageWidth-filterWidth+1]
   */
  shiftdim_ikj(A, mxGetData(image_array), imageHeight, imageWidth, dimension) ;
  shiftdim_kji(B, mxGetData(filter_array), filterHeight, filterWidth, dimension) ;
  for (j = 0 ; j < imageWidth - filterWidth + 1 ; ++j) {
    for (i = 0 ; i < filterHeight ; ++i) {
      char trans = 'n' ;
      float one = 1 ;
      ptrdiff_t lda = imageHeight ;
      ptrdiff_t m = imageHeight - filterHeight + 1 ;
      ptrdiff_t n = dimension * filterWidth ;
      ptrdiff_t incx = 1 ;
      ptrdiff_t incy = 1 ;
      sgemv (&trans,
             &m, &n, /* A dimension */
             &one, /* alpha = 1 */
             A + (i + j * lda * dimension), /* A with offset */
             &lda, /* column stride of A (we use subset) */
             B + i * n, /* B with offset */
             &incx, /* B stride = 1 */
             &one, /* beta = 1 */
             C + j * m, /* C with offset */
             &incy /* C stride */
             ) ;
    }
  }
#endif
  vl_free(A) ;
  vl_free(B) ;
}
