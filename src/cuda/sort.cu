





/*!
 * Compute the next power of 2 which occurs after a number.
 *
 * @param n
 */
__device__
int nextPower2(int n)
{
   int pow2 = 2;
   while ( pow2 < n )
   {
      pow2 *= 2;
   }

   return pow2;
}






/*!
 * Swap two values
 *
 * @param a
 * @param b
 */
__device__
void swap(float *a, float *b)
{
   float c = *a;
   *a = *b;
   *b = c;
}






/*!
 * Sort an array using bitonic sort. The array should have a size which is a
 * power of two.
 *
 * @param array
 * @param size
 * @param stride
 */
__device__
void bitonicSort(float *array, int size, int stride)
{
   int bsize = size / 2;
   int dir, a, b, t;

   for ( int ob = 2; ob <= size; ob *= 2 )
   {
      for ( int ib = ob; ib >= 2; ib /= 2 )
      {
         t = ib/2;
         for ( int i = 0; i < bsize; ++i )
         {
            dir = -((i/(ob/2)) & 0x1);
            a = (i/t) * ib + (i%t);
            b = a + t;

            a *= stride;
            b *= stride;

            if ( (!dir && (array[a] > array[b])) || (dir && (array[a] < array[b])) )
            {
               swap(&array[a], &array[b]);
            }
         }
      }
   }
}






/*!
 * Sort an array using bitonic sort, while also applying the same swap operations
 * to a second array of the same size. The arrays should have a size which is a
 * power of two.
 *
 * @param array
 * @param extra
 * @param size
 * @param stride
 */
__device__
void bitonicSortFF(float *array, float *extra, int size, int stride)
{
   int bsize = size / 2;
   int dir, a, b, t;

   for ( int ob = 2; ob <= size; ob *= 2 )
   {
      for ( int ib = ob; ib >= 2; ib /= 2 )
      {
         t = ib/2;
         for ( int i = 0; i < bsize; ++i )
         {
            dir = -((i/(ob/2)) & 0x1);
            a = (i/t) * ib + (i%t);
            b = a + t;

            a *= stride;
            b *= stride;

            if ( (!dir && (array[a] > array[b])) || (dir && (array[a] < array[b])) )
            {
               swap(&array[a], &array[b]);
               swap(&extra[a], &extra[b]);
            }
         }
      }
   }
}






/*!
 * Compute the rank of a sorted vector in place. In the event of ties,
 * the ranks are corrected using fractional ranking.
 *
 * @param array
 * @param n
 * @param stride
 */
__device__
void computeRank(float *array, int n, int stride)
{
   int i = 0;

   while ( i < n - 1 )
   {
      float a_i = array[i * stride];

      if ( a_i == array[(i + 1) * stride] )
      {
         int j = i + 2;
         int k;
         float rank = 0;

         // we have detected a tie, find number of equal elements
         while ( j < n && a_i == array[j * stride] )
         {
            ++j;
         }

         // compute rank
         for ( k = i; k < j; ++k )
         {
            rank += k;
         }

         // divide by number of ties
         rank /= (float) (j - i);

         for ( k = i; k < j; ++k )
         {
            array[k * stride] = rank;
         }

         i = j;
      }
      else
      {
         // no tie - set rank to natural ordered position
         array[i * stride] = i;
         ++i;
      }
   }

   if ( i == n - 1 )
   {
      array[(n - 1) * stride] = (float) (n - 1);
   }
}
