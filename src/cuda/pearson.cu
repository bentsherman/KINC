





/*!
 * Compute the Pearson correlation of a cluster in a pairwise data array.
 *
 * @param x
 * @param y
 * @param labels
 * @param sampleSize
 * @param stride
 * @param cluster
 * @param minSamples
 */
__device__
float Pearson_computeCluster(
   const float *x,
   const float *y,
   const char *labels,
   int sampleSize,
   int stride,
   char cluster,
   int minSamples)
{
   // compute intermediate sums
   int n = 0;
   float sumx = 0;
   float sumy = 0;
   float sumx2 = 0;
   float sumy2 = 0;
   float sumxy = 0;

   for ( int i = 0; i < sampleSize; ++i )
   {
      if ( labels[i * stride] == cluster )
      {
         float x_i = x[i];
         float y_i = y[i];

         sumx += x_i;
         sumy += y_i;
         sumx2 += x_i * x_i;
         sumy2 += y_i * y_i;
         sumxy += x_i * y_i;

         ++n;
      }
   }

   // compute correlation only if there are enough samples
   float result = NAN;

   if ( n >= minSamples )
   {
      result = (n*sumxy - sumx*sumy) / sqrt((n*sumx2 - sumx*sumx) * (n*sumy2 - sumy*sumy));
   }

   return result;
}






/*!
 * Compute the correlation of each cluster in a pairwise data array. The data array
 * should only contain the clean samples that were extracted from the expression
 * matrix, while the labels should contain all samples.
 *
 * @param numPairs
 * @param expressions
 * @param sampleSize
 * @param in_index
 * @param in_argsort
 * @param clusterSize
 * @param in_labels
 * @param minSamples
 * @param out_correlations
 */
__global__
void Pearson_compute(
   int numPairs,
   const float *expressions,
   int sampleSize,
   const int2 *in_index,
   const int *in_argsort,
   char clusterSize,
   const char *in_labels,
   int minSamples,
   float *out_correlations)
{
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = gridDim.x * blockDim.x;

   if ( i >= numPairs )
   {
      return;
   }

   // initialize workspace variables
   int i_argsort = in_argsort[i];
   int2 index = in_index[i_argsort];
   const float *x = &expressions[index.x * sampleSize];
   const float *y = &expressions[index.y * sampleSize];
   const char *labels = &in_labels[i_argsort];
   float *correlations = &out_correlations[i_argsort];

   for ( char k = 0; k < clusterSize; ++k )
   {
      correlations[k * stride] = Pearson_computeCluster(x, y, labels, sampleSize, stride, k, minSamples);
   }
}
