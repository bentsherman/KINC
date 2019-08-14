#ifndef SIMILARITY_RESULTBLOCK_H
#define SIMILARITY_RESULTBLOCK_H
#include "similarity.h"



/*!
 * This class implements the result block of the similarity analytic.
 */
class Similarity::ResultBlock : public EAbstractAnalyticBlock
{
   Q_OBJECT
public:
   /*!
    * Construct a new result block in an uninitialized null state.
    */
   explicit ResultBlock() = default;
   explicit ResultBlock(int index, qint64 start);
   template<class T> static QVector<T> makeVector(const T* data, int size, int stride=1);
   qint64 start() const { return _start; }
   const QVector<Pair>& pairs() const { return _pairs; }
   QVector<Pair>& pairs() { return _pairs; }
   void append(const Pair& pair);
protected:
   virtual void write(QDataStream& stream) const override final;
   virtual void read(QDataStream& stream) override final;
private:
   /*!
    * The pairwise index of the first pair in the result block.
    */
   qint64 _start;
   /*!
    * The list of pairs that were processed.
    */
   QVector<Pair> _pairs;
};






/*!
 * Create a vector from the given pointer and size. The contents of the
 * pointer are copied into the vector.
 *
 * @param data
 * @param size
 * @param stride
 */
template<class T>
QVector<T> Similarity::ResultBlock::makeVector(const T* data, int size, int stride)
{
   QVector<T> v(size);

   if ( stride == 1 )
   {
      memcpy(v.data(), data, size * sizeof(T));
   }
   else
   {
      for ( int i = 0; i < size; i++ )
      {
         v[i] = data[i * stride];
      }
   }
   return v;
}



#endif
