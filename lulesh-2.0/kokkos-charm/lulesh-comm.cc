#include "lulesh.h"


#include <string.h>
#include <algorithm>

/* Comm Routines */

#define ALLOW_UNPACKED_PLANE false
#define ALLOW_UNPACKED_ROW   false
#define ALLOW_UNPACKED_COL   false

/*
   There are coherence issues for packing and unpacking message
   buffers.  Ideally, you would like a lot of threads to 
   cooperate in the assembly/dissassembly of each message.
   To do that, each thread should really be operating in a
   different coherence zone.

   Let's assume we have three fields, f1 through f3, defined on
   a 61x61x61 cube.  If we want to send the block boundary
   information for each field to each neighbor processor across
   each cube face, then we have three cases for the
   memory layout/coherence of data on each of the six cube
   boundaries:

      (a) Two of the faces will be in contiguous memory blocks
      (b) Two of the faces will be comprised of pencils of
          contiguous memory.
      (c) Two of the faces will have large strides between
          every value living on the face.

   How do you pack and unpack this data in buffers to
   simultaneous achieve the best memory efficiency and
   the most thread independence?

   Do do you pack field f1 through f3 tighly to reduce message
   size?  Do you align each field on a cache coherence boundary
   within the message so that threads can pack and unpack each
   field independently?  For case (b), do you align each
   boundary pencil of each field separately?  This increases
   the message size, but could improve cache coherence so
   each pencil could be processed independently by a separate
   thread with no conflicts.

   Also, memory access for case (c) would best be done without
   going through the cache (the stride is so large it just causes
   a lot of useless cache evictions).  Is it worth creating
   a special case version of the packing algorithm that uses
   non-coherent load/store opcodes?
*/


/******************************************/

// extern "C" void packingDoneCallback(void* param, void* msg) {
//    PackingDoneMsg* m = (PackingDoneMsg*) msg;
//    m->domain->packingDone(m->msgType, m->x, m->y, m->z, m->xferFields, m->sendCount, m->offset);
// }

/******************************************/

void Copy1D(Kokkos::View<Real_t*> &src, int src_offset,
   int src_stride,
    Kokkos::View<Real_t*> &dest, int dst_offset, int dst_stride,
   int size, ExecSpace execSpace)
{
   //Copy1D
   Kokkos::parallel_for(Kokkos::Experimental::require(RangePolicy(execSpace, 0, size), Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       KOKKOS_LAMBDA(const int i) {
      dest[dst_offset + i * dst_stride] = src[src_offset + i * src_stride];
   });
}

void Add1D(Kokkos::View<Real_t*> &src, int src_offset, int src_stride,
   Kokkos::View<Real_t*> &dest, int dst_offset, int dst_stride,
   int size, ExecSpace execSpace)
{
   Kokkos::parallel_for("Add1D", Kokkos::Experimental::require(RangePolicy(execSpace, 0, size), Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       KOKKOS_LAMBDA(const int i) {
      dest[dst_offset + i * dst_stride] += src[src_offset + i * src_stride];
   });
}

/******************************************/

void Copy2D(Kokkos::View<Real_t*> &src, 
   int src_offset,
   int src_stride_x, int src_stride_y,
   Kokkos::View<Real_t*> &dest, 
   int dst_offset,
   int dst_stride_x, int dst_stride_y,
   int dim_x, int dim_y, ExecSpace execSpace)
{
   Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(execSpace, {0, 0}, {dim_x, dim_y});
   
   //"Copy2D"
   Kokkos::parallel_for(Kokkos::Experimental::require(policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       KOKKOS_LAMBDA(const int i, const int j) {
      dest[dst_offset + j * dst_stride_y + i * dst_stride_x] = 
         src[src_offset + j * src_stride_y + i * src_stride_x];
   });
}

void Add2D(Kokkos::View<Real_t*> &src, 
   int src_offset,
   int src_stride_x, int src_stride_y,
   Kokkos::View<Real_t*> &dest, 
   int dst_offset,
   int dst_stride_x, int dst_stride_y,
   int dim_x, int dim_y, ExecSpace execSpace)
{
   Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(execSpace, {0, 0}, {dim_x, dim_y});
   Kokkos::parallel_for("Add2D", Kokkos::Experimental::require(policy, Kokkos::Experimental::WorkItemProperty::HintLightWeight),
                       KOKKOS_LAMBDA(const int i, const int j) {
      dest[dst_offset + j * dst_stride_y + i * dst_stride_x] += 
         src[src_offset + j * src_stride_y + i * src_stride_x];
   });
}

/******************************************/

// Helper to create a CommData and allocate its per-neighbor buffer
static CommData MakeCommData(int offset,
   int src_stride_0, int src_stride_1,
   int dst_stride_0, int dst_stride_1,
   int size_0, int size_1)
{
   CommData cd(offset, src_stride_0, src_stride_1, dst_stride_0, dst_stride_1, size_0, size_1);
   int bufSize = MAX_FIELDS_PER_MPI_COMM * size_0 * size_1;
   // For corners, ensure at least CACHE_COHERENCE_PAD_REAL per field
   if (size_0 == 1 && size_1 == 1)
      bufSize = MAX_FIELDS_PER_MPI_COMM * CACHE_COHERENCE_PAD_REAL;
   Kokkos::resize(cd.buffer, bufSize);
   Kokkos::deep_copy(cd.buffer, 0);
   cd.bufSize = bufSize;
   return cd;
}

void DomainChare::CommDataSendInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
                               bool doSend, bool planeOnly, CommDataMap_t &commDataMap)
{
   if (domain.numRanks() == 1)
      return ;

   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;
   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;
   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   if (planeMin | planeMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dy ;

      if (planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z-1}] =
            MakeCommData(0, 1, 0, 1, 0, sendCount, 1);
      }
      if (planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z+1}] =
            MakeCommData(dx*dy*(dz - 1), 1, 0, 1, 0, sendCount, 1);
      }
   }
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */

      if (rowMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z}] =
            MakeCommData(0, 1, dx*dy, 1, dx, dx, dz);
      }
      if (rowMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z}] =
            MakeCommData(dx*(dy - 1), 1, dx*dy, 1, dx, dx, dz);
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */

      if (colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z}] =
            MakeCommData(0, dx, dx*dy, 1, dy, dy, dz);
      }
      if (colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z}] =
            MakeCommData(dx - 1, dx, dx*dy, 1, dy, dy, dz);
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z}] = 
            MakeCommData(0, dx*dy, 0, 1, 0, dz, 1);
      }

      if (rowMin && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(0, 1, 0, 1, 0, dx, 1);
      }

      if (colMin && planeMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z-1}] = 
            MakeCommData(0, dx, 0, 1, 0, dy, 1);
      }

      if (rowMax && colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z}] = 
            MakeCommData(dx*dy - 1, dx*dy, 0, 1, 0, dz, 1);
      }

      if (rowMax && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(dx*(dy-1) + dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
      }

      if (colMax && planeMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz-1) + dx-1, dx, 0, 1, 0, dy, 1);
      }

      if (rowMax && colMin && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z}] = 
            MakeCommData(dx*(dy - 1), dx*dy, 0, 1, 0, dz, 1);
      }

      if (rowMin && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
      }

      if (colMin && planeMax && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz - 1), dx, 0, 1, 0, dy, 1);
      }

      if (rowMin && colMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z}] = 
            MakeCommData(dx-1, dx*dy, 0, 1, 0, dz, 1);
      }

      if (rowMax && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(dx*(dy - 1), 1, 0, 1, 0, dx, 1);
      }

      if (colMax && planeMin) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z-1}] = 
            MakeCommData(dx - 1, dx, 0, 1, 0, dy, 1);
      }

      if (rowMin && colMin && planeMin) {
         /* corner at domain logical coord (0, 0, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(0, 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMax && planeMin) {
         /* corner at domain logical coord (1, 0, 0) */
         Index_t idx = dx - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMin && planeMin) {
         /* corner at domain logical coord (0, 1, 0) */
         Index_t idx = dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 1, 1) */
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMax && planeMin) {
         /* corner at domain logical coord (1, 1, 0) */
         Index_t idx = dx*dy - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 1, 1) */
         Index_t idx = dx*dy*dz - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(idx, 1, 0, 1, 0, 1, 1);
      }
   }
}

void DomainChare::CommDataRecvInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
                      bool doRecv, bool planeOnly, CommDataMap_t &commDataMap) {

   if (domain.numRanks() == 1)
      return ;

   bool rowMin, rowMax, colMin, colMax, planeMin, planeMax ;

   /* assume communication to 6 neighbors by default */
   rowMin = rowMax = colMin = colMax = planeMin = planeMax = true ;

   if (domain.rowLoc() == 0) {
      rowMin = false ;
   }
   if (domain.rowLoc() == (domain.tp()-1)) {
      rowMax = false ;
   }
   if (domain.colLoc() == 0) {
      colMin = false ;
   }
   if (domain.colLoc() == (domain.tp()-1)) {
      colMax = false ;
   }
   if (domain.planeLoc() == 0) {
      planeMin = false ;
   }
   if (domain.planeLoc() == (domain.tp()-1)) {
      planeMax = false ;
   }

   /* receive data from neighboring domain faces */
   int planeIdx = 0; // sequential index for ghost region destination (used by MonoQ)
   if (planeMin && doRecv) {
      /* contiguous memory */
      auto cd = MakeCommData(0, 1, 0, 1, 0, dx * dy, 1);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z-1}] = cd;
   }
   if (planeMax) {
      /* contiguous memory */
      auto cd = MakeCommData(dx*dy*(dz - 1), 1, 0, 1, 0, dx * dy, 1);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z+1}] = cd;
   }
   if (rowMin && doRecv) {
      /* semi-contiguous memory */
      auto cd = MakeCommData(0, 1, dx, 1, dx*dy, dx, dz);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z}] = cd;
   }
   if (rowMax) {
      /* semi-contiguous memory */
      auto cd = MakeCommData(dx*(dy - 1), 1, dx, 1, dx*dy, dx, dz);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z}] = cd;
   }
   if (colMin && doRecv) {
      /* scattered memory */
      auto cd = MakeCommData(0, 1, dy, dx, dx*dy, dy, dz);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z}] = cd;
   }
   if (colMax) {
      /* scattered memory */
      auto cd = MakeCommData(dx - 1, 1, dy, dx, dx*dy, dy, dz);
      cd.ghostOffset = planeIdx++;
      commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z}] = cd;
   }

   if (!planeOnly) {
      /* receive data from domains connected only by an edge */
      if (rowMin && colMin && doRecv) {
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z}] = 
            MakeCommData(0, 1, 0, dx*dy, 0, dz, 1);
      }

      if (rowMin && planeMin && doRecv) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(0, 1, 0, 1, 0, dx, 1);
      }

      if (colMin && planeMin && doRecv) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z-1}] =
            MakeCommData(0, 1, 0, dx, 0, dy, 1);
      }

      if (rowMax && colMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z}] =
            MakeCommData(dx*dy - 1, 1, 0, dx*dy, 0, dz, 1);
      }

      if (rowMax && planeMax) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(dx*(dy-1) + dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
      }

      if (colMax && planeMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z+1}] =
            MakeCommData(dx*dy*(dz-1) + dx-1, 1, 0, dx, 0, dy, 1);
      }

      if (rowMax && colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z}] =
            MakeCommData(dx*(dy - 1), 1, 0, dx*dy, 0, dz, 1);
      }

      if (rowMin && planeMax) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
      }

      if (colMin && planeMax) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z+1}] =
            MakeCommData(dx*dy*(dz - 1), 1, 0, dx, 0, dy, 1);
      }

      if (rowMin && colMax && doRecv) {
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z}] =
            MakeCommData(dx-1, 1, 0, dx*dy, 0, dz, 1);
      }

      if (rowMax && planeMin && doRecv) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(dx*(dy - 1), 1, 0, 1, 0, dx, 1);
      }

      if (colMax && planeMin && doRecv) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z-1}] =
            MakeCommData(dx - 1, 1, 0, dx, 0, dy, 1);
      }

      /* receive data from domains connected only by a corner */
      if (rowMin && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 0, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(0, 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMin && planeMax) {
         /* corner at domain logical coord (0, 0, 1) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz - 1), 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 0, 0) */
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z-1}] = 
            MakeCommData(dx - 1, 1, 0, 1, 0, 1, 1);
      }
      if (rowMin && colMax && planeMax) {
         /* corner at domain logical coord (1, 0, 1) */
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz - 1) + (dx - 1), 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 1, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(dx*(dy - 1), 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMin && planeMax) {
         /* corner at domain logical coord (0, 1, 1) */
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*(dz - 1) + dx*(dy - 1), 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 1, 0) */
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z-1}] = 
            MakeCommData(dx*dy - 1, 1, 0, 1, 0, 1, 1);
      }
      if (rowMax && colMax && planeMax) {
         /* corner at domain logical coord (1, 1, 1) */
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z+1}] = 
            MakeCommData(dx*dy*dz - 1, 1, 0, 1, 0, 1, 1);
      }
   }
}

void DomainChare::CommSend(Domain& domain, int msgType,
                           Index_t xferFields, Kokkos::View<Real_t*> *fieldData,
                           Index_t dx, Index_t dy, Index_t dz, bool doSend, bool planeOnly,
                           CommDataMap_t& commDataMap)
{
   if (domain.numRanks() == 1) {
      if (msgType == MSG_SYNC_POS_VEL) {
         thisProxy[thisIndex].PosVelSendDone();
      }
      else if (msgType == MSG_MONOQ) {
         thisProxy[thisIndex].MonoQSendDone();
      }
      else if (msgType == MSG_COMM_SBN) {
         thisProxy[thisIndex].SBNSendDone();
      }
      return ;
   }

   CommDataMapIter_t it;

   DBG_PRINTF("[DEBUG CommSend] (%d,%d,%d) iter=%u msgType=0x%x xferFields=%d commDataMap.size()=%lu\n",
      thisIndex.x, thisIndex.y, thisIndex.z, iter, msgType, xferFields, (unsigned long)commDataMap.size());

   if (commDataMap.size() == 0) {
      if (msgType == MSG_SYNC_POS_VEL) {
         thisProxy[thisIndex].PosVelSendDone();
      }
      else if (msgType == MSG_MONOQ) {
         thisProxy[thisIndex].MonoQSendDone();
      }
      else if (msgType == MSG_COMM_SBN) {
         thisProxy[thisIndex].SBNSendDone();
      }
   }

   for (it = commDataMap.begin(); it != commDataMap.end(); ++it) {
      std::tuple<int, int, int> idx = it->first ;
      CommData& cdata = it->second ;
      int offsetX = std::get<0>(idx) - thisIndex.x ;
      int offsetY = std::get<1>(idx) - thisIndex.y ;
      int offsetZ = std::get<2>(idx) - thisIndex.z ;

      if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) ||
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy2D(src, cdata.offset, cdata.src_stride[0], cdata.src_stride[1],
                   cdata.buffer, 
                   fi * cdata.size[0] * cdata.size[1], 
                   cdata.dst_stride[0], cdata.dst_stride[1], 
                   cdata.size[0], cdata.size[1], commSpace);
         }
      } else {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy1D(src, cdata.offset, cdata.src_stride[0],
                   cdata.buffer, 
                   fi * cdata.size[0], 
                   cdata.dst_stride[0], cdata.size[0], commSpace);
         }
      }

      int sendCount = xferFields * cdata.size[0] * cdata.size[1];
      PackingDoneMsg* msg = new PackingDoneMsg(msgType,
         iter,
         std::get<0>(idx), std::get<1>(idx), std::get<2>(idx),
         xferFields, sendCount);
      
      uint32_t ref = MAKE_REF(msg->msgType, msg->sendIter);
      CkCallback* cb;
      CkArrayIndex3D myIndex = CkArrayIndex3D(thisIndex);

      if (msg->msgType == MSG_SYNC_POS_VEL) {
         cb = new CkCallback(CkIndex_DomainChare::PosVelSendCallback(), myIndex, thisArrayID);
      }
      else if (msg->msgType == MSG_MONOQ) {
         cb = new CkCallback(CkIndex_DomainChare::MonoQSendCallback(), myIndex, thisArrayID);
      }
      else if (msg->msgType == MSG_COMM_SBN) {
         cb = new CkCallback(CkIndex_DomainChare::SBNSendCallback(), myIndex, thisArrayID);
      }
      else
         CkAbort("DomainChare::packingDone: Unknown msgType") ;

      Real_t* sendPtr = cdata.buffer.data();

      thisProxy(msg->x, msg->y, msg->z).CommRecv(ref, thisIndex.x, thisIndex.y, thisIndex.z, 
      msg->xferFields, msg->sendCount, CkDeviceBuffer(sendPtr, *cb, commStream));
      
      DBG_PRINTF("[DEBUG CommSend packing] (%d,%d,%d)->(%d,%d,%d) iter=%u msgType=0x%x xferFields=%d sendCount=%d\n",
         thisIndex.x, thisIndex.y, thisIndex.z,
         std::get<0>(idx), std::get<1>(idx), std::get<2>(idx),
         iter, msgType, xferFields, sendCount);

      // hapiAddCallback(commStream, cb, msg);
   }
}

void DomainChare::PosVelSendCallback() {
   if (++posVelSendsDone == commDataSendPosVel.size()) {
      thisProxy[thisIndex].PosVelSendDone();
      posVelSendsDone = 0 ;
   }
}

void DomainChare::MonoQSendCallback() {
   if (++monoQSendsDone == commDataSendMonoQ.size()) {
      thisProxy[thisIndex].MonoQSendDone();
      monoQSendsDone = 0 ;
   }
}

void DomainChare::SBNSendCallback() {
   if (++sbnSendsDone == commDataSendSBN.size()) {
      thisProxy[thisIndex].SBNSendDone();
      sbnSendsDone = 0 ;
   }
}
   
void DomainChare::packingDone(PackingDoneMsg* msg) {
   std::ostringstream os;
   os<<" [start] packingDone ";
   NVTXTracer(os.str(), NVTXColor::PeterRiver);
   os.clear();
   uint32_t ref = MAKE_REF(msg->msgType, msg->sendIter);
   CkCallback* cb;
   CkArrayIndex3D myIndex = CkArrayIndex3D(thisIndex);

   if (msg->msgType == MSG_SYNC_POS_VEL) {
      cb = new CkCallback(CkIndex_DomainChare::PosVelSendCallback(), myIndex, thisArrayID);
   }
   else if (msg->msgType == MSG_MONOQ) {
      cb = new CkCallback(CkIndex_DomainChare::MonoQSendCallback(), myIndex, thisArrayID);
   }
   else if (msg->msgType == MSG_COMM_SBN) {
      cb = new CkCallback(CkIndex_DomainChare::SBNSendCallback(), myIndex, thisArrayID);
   }
   else
      CkAbort("DomainChare::packingDone: Unknown msgType") ;

   // Look up the send map to find the per-neighbor buffer
   CommDataMap_t* sendMap;
   if (msg->msgType == MSG_SYNC_POS_VEL)
      sendMap = &commDataSendPosVel;
   else if (msg->msgType == MSG_MONOQ)
      sendMap = &commDataSendMonoQ;
   else
      sendMap = &commDataSendSBN;

   CommData& cdata = (*sendMap)[{msg->x, msg->y, msg->z}];
   Real_t* sendPtr = cdata.buffer.data();

   #if DEBUG_COMM
   // Sender-side debug: print source field values at packed positions for PosVel z-face
   if(msg->msgType == MSG_SYNC_POS_VEL)
   {
      int offsetX = msg->x - thisIndex.x ;
      int offsetY = msg->y - thisIndex.y ;
      int offsetZ = msg->z - thisIndex.z ;
   
      if ( offsetZ != 0 && offsetX == 0 && offsetY == 0) {
         commSpace.fence(); // ensure packing kernels done before host read
         auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cdata.buffer);
         int countPerField = cdata.size[0] * cdata.size[1];
         for (Index_t fi = 0; fi < 3 && fi < msg->xferFields; ++fi) {
            int base = fi * countPerField;
            printf("[PACKED chare(%d,%d,%d)->(%d,%d,%d) fi=%d] base=%d countPerField=%d",
               thisIndex.x, thisIndex.y, thisIndex.z,
               msg->x, msg->y, msg->z,
               fi, base, countPerField);
            // count zeros
            int nzero = 0;
            for (int k = 0; k < countPerField; ++k) {
               if (h_send(base + k) == 0.0 || h_send(base + k) == -0.0) nzero++;
            }
            printf(" zeros=%d/%d", nzero, countPerField);
            printf(" | last5:");
            for (int k = countPerField - 5; k < countPerField; ++k) {
               if (k >= 0) printf(" [%d]=%.10e", base + k, h_send(base + k));
            }
            printf("\n");
         }
      }
   }
   #endif

   thisProxy(msg->x, msg->y, msg->z).CommRecv(ref, thisIndex.x, thisIndex.y, thisIndex.z, 
      msg->xferFields, msg->sendCount, CkDeviceBuffer(sendPtr, *cb, commStream));
   // std::ostringstream os;
   os<<" [end] packingDone ";
   NVTXTracer(os.str(), NVTXColor::PeterRiver);
   os.clear();
}

/******************************************/

void DomainChare::CommRecv(uint32_t ref, int x, int y, int z, int xferFields, int& size, Real_t* &buf, CkDeviceBufferPost* post) {
   std::ostringstream os;
   os<<" [start] CommRecv ";
   NVTXTracer(os.str(), NVTXColor::PeterRiver);
   os.clear();
   uint32_t msgType = REF_MSGTYPE(ref);
   CommDataMap_t* commDataMap;
   if (msgType == MSG_SYNC_POS_VEL) {
      commDataMap = &commDataRecvPosVel;
   }
   else if (msgType == MSG_MONOQ) {
      commDataMap = &commDataRecvMonoQ;
   }
   else if (msgType == MSG_COMM_SBN) {
      commDataMap = &commDataRecvSBN;
   }
   else
      CkAbort("DomainChare::CommRecv: Unknown msgType") ;

   CommDataMapIter_t it = commDataMap->find({x, y, z});
   if (it == commDataMap->end()) {
      CkAbort("DomainChare::CommRecv: Invalid comm data map key") ;
   }

   buf = it->second.buffer.data();
   post[0].hapi_stream = commStream;
   // std::ostringstream os;
   os<<" [end] CommRecv ";
   NVTXTracer(os.str(), NVTXColor::PeterRiver);
   os.clear();
}

/******************************************/

void DomainChare::processRemotePosVel(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   Kokkos::View<Real_t*> fieldData[6];
   fieldData[0] = domain.m_x;
   fieldData[1] = domain.m_y;
   fieldData[2] = domain.m_z;
   fieldData[3] = domain.m_xd;
   fieldData[4] = domain.m_yd;
   fieldData[5] = domain.m_zd;

   CommData& cdata = commDataRecvPosVel[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) ||
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         Kokkos::View<Real_t*>& dest = fieldData[fi];
         Copy2D(cdata.buffer,
            fi * cdata.size[0] * cdata.size[1],
            1, cdata.size[0],
            dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
            cdata.size[0], cdata.size[1], commSpace);
         }
   } else {
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         Kokkos::View<Real_t*>& dest = fieldData[fi];
         Copy1D(cdata.buffer,
            fi * cdata.size[0],
            1,
            dest, cdata.offset, cdata.dst_stride[0],
            cdata.size[0], commSpace);
      }
   }
   #if DEBUG_COMM
   // Diagnostic: check recv data immediately at DMA completion for z-face neighbors
   int ox = x - thisIndex.x;
   int oy = y - thisIndex.y;
   int oz = z - thisIndex.z;
   if (oz != 0 && ox == 0 && oy == 0) {
      // z-face: check if data at offset 0 in recvView is correct
      commSpace.fence();
      auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), cdata.buffer);
      int count = cdata.size[0] * cdata.size[1];
      int nzero = 0;
      for (int k = 0; k < count; ++k) {
         if (h_recv(k) == 0.0 || h_recv(k) == -0.0) nzero++;
      }
      if (nzero > count * 9 / 10) {
         printf("[DMA-LAND ZERO chare(%d,%d,%d) from(%d,%d,%d)] fi=0 has %d zeros out of %d\n",
            thisIndex.x, thisIndex.y, thisIndex.z, x, y, z, nzero, count);
         for (int fi = 0; fi < 6 && fi < xferFields; ++fi) {
            printf("  fi=%d last5:", fi);
            int base = fi * count;
            for (int k = count - 5; k < count; ++k) {
               if (k >= 0) printf(" [%d]=%.10e", base + k, h_recv(base + k));
            }
            printf("\n");
         }
      } else {
         printf("[DMA-LAND OK chare(%d,%d,%d) from(%d,%d,%d)] fi=0 has %d zeros out of %d\n",
            thisIndex.x, thisIndex.y, thisIndex.z, x, y, z, nzero, count);
      }
   }
   fflush(stdout);
   #endif
}
/******************************************/

void DomainChare::processRemoteQ(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   CommData& cdata = commDataRecvMonoQ[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[3];
   Index_t fieldOffset[3];
   fieldData[0] = domain.m_delv_xi ;
   fieldData[1] = domain.m_delv_eta ;
   fieldData[2] = domain.m_delv_zeta ;
   fieldOffset[0] = domain.numElem() ;
   fieldOffset[1] = domain.numElem() ;
   fieldOffset[2] = domain.numElem() ;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      int copyLen = cdata.size[0] * cdata.size[1];
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         int srcOff = fi * copyLen;
         int dstOff = fieldOffset[fi] + cdata.ghostOffset * copyLen;
         Copy1D(cdata.buffer, srcOff, 1,
            dest, dstOff, 1, copyLen, commSpace);
      }
   } else {
      int copyLen = cdata.size[0];
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi];
         int srcOff = fi * copyLen;
         int dstOff = fieldOffset[fi] + cdata.ghostOffset * copyLen;
         Copy1D(cdata.buffer, srcOff, 1,
            dest, dstOff, 1, copyLen, commSpace);
      }
   }
}

/******************************************/

void DomainChare::processRemoteMass(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;

   CommData& cdata = commDataRecvSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[1];
   fieldData[0] = domain.m_nodalMass;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add2D(cdata.buffer,
               fi * cdata.size[0] * cdata.size[1],
               1, cdata.size[0],
               dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
               cdata.size[0], cdata.size[1], commSpace);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add1D(cdata.buffer,
               fi * cdata.size[0],
               1,
               dest, cdata.offset, cdata.dst_stride[0],
               cdata.size[0], commSpace);
      }
   }
}

/******************************************/

void DomainChare::processRemoteForce(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
      Domain& domain = *locDom;

   CommData& cdata = commDataRecvSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[3];
   fieldData[0] = domain.m_fx;
   fieldData[1] = domain.m_fy;
   fieldData[2] = domain.m_fz;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         Add2D(cdata.buffer, fi * cdata.size[0] * cdata.size[1],
               1, cdata.size[0],
               dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
               cdata.size[0], cdata.size[1], commSpace);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         Add1D(cdata.buffer, fi * cdata.size[0],
               1,
               dest, cdata.offset, cdata.dst_stride[0],
               cdata.size[0], commSpace);
      }
   }
}