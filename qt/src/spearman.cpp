#include <ace/core/metadata.h>

#include "spearman.h"
#include "datafactory.h"
#include "expressionmatrix.h"
#include "correlationmatrix.h"



using namespace std;






Spearman::~Spearman()
{
   if ( _blocks )
   {
      for (int i = 0; i < _blockSize ;++i)
      {
         delete _blocks[i];
      }
      delete[] _blocks;
   }
   delete _kernel;
   delete _program;
}






EAbstractAnalytic::ArgumentType Spearman::getArgumentData(int argument)
{
   using Type = EAbstractAnalytic::ArgumentType;
   switch (argument)
   {
   case InputData: return Type::DataIn;
   case OutputData: return Type::DataOut;
   case Minimum: return Type::Integer;
   case BlockSize: return Type::Integer;
   case KernelSize: return Type::Integer;
   case PreAllocate: return Type::Bool;
   default: return Type::Bool;
   }
}






QVariant Spearman::getArgumentData(int argument, Role role)
{
   using Role = EAbstractAnalytic::Role;
   switch (role)
   {
   case Role::CommandLineName:
      switch (argument)
      {
      case InputData: return QString("input");
      case OutputData: return QString("output");
      case Minimum: return QString("min");
      case BlockSize: return QString("bsize");
      case KernelSize: return QString("ksize");
      case PreAllocate: return QString("prealloc");
      default: return QVariant();
      }
   case Role::Title:
      switch (argument)
      {
      case InputData: return tr("Input:");
      case OutputData: return tr("Output:");
      case Minimum: return tr("Minimum Sample Size:");
      case BlockSize: return tr("Block Size:");
      case KernelSize: return tr("Kernel Size:");
      case PreAllocate: return tr("Pre-Allocate Output?");
      default: return QVariant();
      }
   case Role::WhatsThis:
      switch (argument)
      {
      case InputData: return tr("Input expression matrix that will be used to compute spearman"
                                " coefficients.");
      case OutputData: return tr("Output correlation matrixx that will store spearman coefficient"
                                 " results.");
      case Minimum: return tr("Minimum size of samples two genes must share to generate a spearman"
                              " coefficient.");
      case BlockSize: return tr("This option only applies if OpenCL is used. Total number of blocks"
                                " to run for execution.");
      case KernelSize: return tr("This option only applies if OpenCL is used. Total number of"
                                 " kernels to run per block of execution.");
      case PreAllocate: return tr("Should the output correlation matrix have file space"
                                  " pre-allocated? WARNING this only works in linux systems.");
      default: return QVariant();
      }
   case Role::DefaultValue:
      switch (argument)
      {
      case Minimum: return 30;
      case BlockSize: return 4;
      case KernelSize: return 4096;
      default: return QVariant();
      }
   case Role::Minimum:
      switch (argument)
      {
      case Minimum: return 1;
      case BlockSize: return 1;
      case KernelSize: return 1;
      default: return QVariant();
      }
   case Role::Maximum:
      switch (argument)
      {
      case Minimum: return INT_MAX;
      case BlockSize: return INT_MAX;
      case KernelSize: return INT_MAX;
      default: return QVariant();
      }
   case Role::DataType:
      switch (argument)
      {
      case InputData: return DataFactory::ExpressionMatrixType;
      case OutputData: return DataFactory::CorrelationMatrixType;
      default: return QVariant();
      }
   default:
      return QVariant();
   }
}






void Spearman::setArgument(int argument, QVariant value)
{
   switch (argument)
   {
   case Minimum:
      _minimum = value.toInt();
      break;
   case BlockSize:
      _blockSize = value.toInt();
      break;
   case KernelSize:
      _kernelSize = value.toInt();
      break;
   case PreAllocate:
      _allocate = value.toBool();
      break;
   }
}






void Spearman::setArgument(int argument, EAbstractData *data)
{
   switch (argument)
   {
   case InputData:
      _input = dynamic_cast<ExpressionMatrix*>(data);
      break;
   case OutputData:
      _output = dynamic_cast<CorrelationMatrix*>(data);
      break;
   }
}






bool Spearman::initialize()
{
   if ( !_input || !_output )
   {
      ;//ERROR!
   }
   _output->initialize(_input->getGeneNames(),_input->getSampleSize(),1,1);
   return _allocate;
}






void Spearman::runSerial()
{
}






int Spearman::getBlockSize()
{
   if ( _blockSize < 1 || _kernelSize < 1 )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("Invalid Argument"));
      e.setDetails((tr("Block size and/or kernel size are set to values less than 1.")));
      throw e;
   }
   EOpenCLDevice& device {EOpenCLDevice::getInstance()};
   _program = device.makeProgram().release();
   if ( !device )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   _program->addFile(":/opencl/spearman.cl");
   if ( !_program->compile() )
   {
      E_MAKE_EXCEPTION(e);
      e.setTitle(tr("OpenCL Compile Error"));
      e.setDetails(tr("OpenCL program failed to compile:\n\n%1").arg(_program->getBuildError()));
      throw e;
   }
   _kernel = _program->makeKernel("calculateSpearmanBlock").release();
   if ( !*_program )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   _expressions = device.makeBuffer<cl_float>(_input->getRawSize()).release();
   if ( !device )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   unique_ptr<ExpressionMatrix::Expression> rawData(_input->dumpRawData());
   ExpressionMatrix::Expression* rawDataRef {rawData.get()};
   for (int i = 0; i < _input->getRawSize() ;++i)
   {
      (*_expressions)[i] = rawDataRef[i];
   }
   EOpenCLEvent event = _expressions->write();
   if ( !*_expressions )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   event.wait();
   if ( !event )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   int pow2Size {2};
   while ( pow2Size < _output->getSampleSize() )
   {
      pow2Size *= 2;
   }
   int pow2 {2};
   while ( pow2 < _kernelSize )
   {
      pow2 *= 2;
   }
   _kernelSize = pow2;
   _blocks = new Block*[_blockSize];
   for (int i = 0; i < _blockSize ;++i)
   {
      _blocks[i] = new Block(device,pow2Size,_kernelSize);
   }
   int workgroupSize {2};
   while ( workgroupSize*2 <= _kernelSize && workgroupSize < (int)_kernel->getMaxWorkgroupSize() )
   {
      workgroupSize *= 2;
   }
   _kernel->setArgument(0,(cl_int)_output->getSampleSize());
   _kernel->setArgument(1,(cl_int)pow2Size);
   _kernel->setArgument(2,(cl_int)_minimum);
   _kernel->setBuffer(4,_expressions);
   _kernel->setDimensionCount(1);
   _kernel->setGlobalSize(0,_kernelSize);
   _kernel->setWorkgroupSize(0,workgroupSize);
   if ( !*_kernel )
   {
      E_MAKE_EXCEPTION(e);
      throw e;
   }
   qint64 geneSize {_output->getGeneSize()};
   _totalPairs = geneSize*(geneSize-1)/2;
   return _blockSize;
}






bool Spearman::runBlock(int index)
{
   Block& block {*_blocks[index]};
   switch (block.state)
   {
   case Block::Start:
      if ( _x < _output->getGeneSize() )
      {
         block.x = _x;
         block.y = _y;
         int index {0};
         while ( _x < _output->getGeneSize() && index < _kernelSize )
         {
            (*block.references)[index*2] = _x;
            (*block.references)[(index*2)+1] = _y;
            CorrelationMatrix::increment(_x,_y);
            ++index;
         }
         while ( index < _kernelSize )
         {
            (*block.references)[index*2] = 0;
            (*block.references)[(index*2)+1] = 0;
            ++index;
         }
         block.event = block.references->write();
         if ( !*block.references )
         {
            E_MAKE_EXCEPTION(e);
            throw e;
         }
         block.state = Block::Load;
      }
      else
      {
         block.state = Block::Done;
      }
      break;
   case Block::Load:
      if ( block.event.isDone() )
      {
         if ( !block.event )
         {
            E_MAKE_EXCEPTION(e);
            throw e;
         }
         _kernel->setBuffer(3,block.references);
         _kernel->setBuffer(5,block.workBuffer);
         _kernel->setBuffer(6,block.rankBuffer);
         _kernel->setBuffer(7,block.answers);
         block.event = _kernel->execute();
         if ( !*_kernel )
         {
            E_MAKE_EXCEPTION(e);
            throw e;
         }
         block.state = Block::Execute;
      }
      break;
   case Block::Execute:
      if ( block.event.isDone() )
      {
         if ( !block.event )
         {
            E_MAKE_EXCEPTION(e);
            e.setTitle(block.event.getErrorFunction());
            e.setDetails(block.event.getErrorCode());
            throw e;
         }
         block.event = block.answers->read();
         if ( !*block.answers )
         {
            E_MAKE_EXCEPTION(e);
            throw e;
         }
         block.state = Block::Read;
      }
      break;
   case Block::Read:
      if ( block.event.isDone() )
      {
         if ( !block.event )
         {
            E_MAKE_EXCEPTION(e);
            throw e;
         }
         CorrelationMatrix::Pair pair(_output);
         pair.setModeSize(1);
         for (int i = 0; i < _output->getSampleSize() ;++i)
         {
            pair.mode(0,i) = 1;
         }
         int index {0};
         while ( block.x < _output->getGeneSize() && index < _kernelSize )
         {
            pair.at(0,0) = (*block.answers)[index];
            pair.write(block.x,block.y);
            CorrelationMatrix::increment(block.x,block.y);
            ++index;
            ++_pairsComplete;
         }
         block.state = Block::Start;
      }
      break;
   case Block::Done:
   default:
      return false;
   }
   int newPercent = _pairsComplete*100/_totalPairs;
   if ( newPercent != _lastPercent )
   {
      _lastPercent = newPercent;
      emit progressed(_lastPercent);
   }
   return true;
}