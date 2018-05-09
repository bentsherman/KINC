#ifndef PAIRWISE_BASE_H
#define PAIRWISE_BASE_H
#include <ace/core/AceCore.h>
#include <ace/core/metadata.h>

#include "pairwise_index.h"



namespace Pairwise
{
   class Base : public EAbstractData
   {
   public:
      class Pair
      {
      public:
         Pair(Base* matrix):
            _matrix(matrix),
            _cMatrix(matrix),
            _index({matrix->_geneSize,0})
            {}
         Pair(const Base* matrix):
            _cMatrix(matrix),
            _index({matrix->_geneSize,0})
            {}
         Pair() {}
         Pair(const Pair&) = default;
         Pair(Pair&&) = default;
         virtual void clearClusters() const = 0;
         virtual void addCluster(int amount = 1) const = 0;
         virtual int clusterSize() const = 0;
         virtual bool isEmpty() const = 0;
         void write(Index index);
         void read(Index index) const;
         void reset() const { _rawIndex = 0; };
         void readNext() const;
         bool hasNext() const { return _rawIndex != _cMatrix->_clusterSize; }
         const Index& index() const { return _index; }
         Pair& operator=(const Pair&) = default;
         Pair& operator=(Pair&&) = default;
      protected:
         virtual void writeCluster(EDataStream& stream, int cluster) = 0;
         virtual void readCluster(const EDataStream& stream, int cluster) const = 0;
      private:
         Base* _matrix {nullptr};
         const Base* _cMatrix;
         mutable qint64 _rawIndex {0};
         mutable Index _index;
      };
      virtual void readData() override final;
      virtual quint64 getDataEnd() const override final
         { return _headerSize + _offset + _clusterSize*(_dataSize + _itemHeaderSize); }
      virtual void newData() override final;
      virtual void prepare(bool) override final {}
      virtual void finish() override final { newData(); }
      int geneSize() const { return _geneSize; }
      qint64 size() const { return _pairSize; }
      const EMetadata& geneNames() const;
   protected:
      virtual void writeHeader() = 0;
      virtual void readHeader() = 0;
      void initialize(const EMetadata& geneNames, int dataSize, int offset);
   private:
      void write(Index index, qint8 cluster);
      Index getPair(qint64 index, qint8* cluster) const;
      qint64 findPair(qint64 indent, qint64 first, qint64 last) const;
      void seekPair(qint64 index) const;
      constexpr static int _headerSize {26};
      constexpr static int _itemHeaderSize {9};
      qint32 _geneSize {0};
      qint32 _dataSize {0};
      qint64 _pairSize {0};
      qint64 _clusterSize {0};
      qint16 _offset {0};
      qint64 _lastWrite {-2};
   };
}



#endif