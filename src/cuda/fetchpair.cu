
// #include "linalg.cu"






/*!
 * Compute the initial labels for a gene pair in an expression matrix. Samples
 * with missing values and samples that fall below the expression threshold are
 * labeled as such, all other samples are labeled as cluster 0. The number of
 * clean samples is returned.
 *
 * @param numPairs
 * @param expressions
 * @param sampleSize
 * @param in_index
 * @param minExpression
 * @param out_N
 * @param out_labels
 */
__global__
void fetchPair(
   int numPairs,
   const float *expressions,
   int sampleSize,
   const int2 *in_index,
   int minExpression,
   int *out_N,
   char *out_labels)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   if ( i >= numPairs )
   {
      return;
   }

   // initialize variables
   int2 index = in_index[i];
   char *labels = &out_labels[i];
   int *p_N = &out_N[i];

   // index into gene expressions
   const float *x = &expressions[index.x * sampleSize];
   const float *y = &expressions[index.y * sampleSize];

   // label the pairwise samples
   int N = 0;

   for ( int i = 0, j = 0; i < sampleSize; i += 1, j += stride )
   {
      // label samples with missing values
      if ( isnan(x[i]) || isnan(y[i]) )
      {
         labels[j] = -9;
      }

      // label samples which fall below the expression threshold
      else if ( x[i] < minExpression || y[i] < minExpression )
      {
         labels[j] = -6;
      }

      // label any remaining samples as cluster 0
      else
      {
         N++;
         labels[j] = 0;
      }
   }

   // save number of clean samples
   *p_N = N;
}
