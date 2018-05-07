#ifndef EXPORTCORRELATIONMATRIX_H
#define EXPORTCORRELATIONMATRIX_H
#include <ace/core/AceCore.h>

#include "ccmatrix.h"
#include "correlationmatrix.h"



class ExportCorrelationMatrix : public EAbstractAnalytic
{
   Q_OBJECT

public:
   enum Arguments
   {
      ClusterData = 0
      ,CorrelationData
      ,OutputFile
      ,Total
   };

   virtual int getArgumentCount() override final { return Total; }
   virtual ArgumentType getArgumentData(int argument) override final;
   virtual QVariant getArgumentData(int argument, Role role) override final;
   virtual void setArgument(int argument, EAbstractData* data) override final;
   virtual void setArgument(int argument, QFile* file) override final;
   quint32 getCapabilities() const override final { return Capabilities::Serial; }
   virtual bool initialize() override final;
   virtual void runSerial() override final;

private:
   CCMatrix* _ccm {nullptr};
   CorrelationMatrix* _cmx {nullptr};
   QFile* _output {nullptr};
};



#endif
