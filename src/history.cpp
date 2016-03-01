#include "history.h"



void History::add_child(const History& child)
{
   HistItem nChild(_mem);
   nChild.copy_from(child._head);
   if (_head.childHead()==FileMem::nullPtr)
   {
      _head.childHead(nChild.addr());
      _head.sync();
   }
   else
   {
      HistItem tmp(_mem,_head.childHead());
      while (tmp.next()!=FileMem::nullPtr)
      {
         tmp = tmp.next();
      }
      tmp.next(nChild.addr());
      tmp.sync();
   }
}
