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


void DomainChare::CommDataSendInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
                               bool doSend, bool planeOnly, CommDataMap_t &commDataMap)
{
   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   int myRank ;
   Index_t maxPlaneComm = domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
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
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z-1}] = CommData(
            pmsg, emsg, cmsg, 0, 1, 0, 1, 0, sendCount, 1);
         ++pmsg ;
      }
      if (planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z+1}] = CommData(
            pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, sendCount, 1);
         ++pmsg ;
      }
   }
   if (rowMin | rowMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dx * dz ;

      if (rowMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, 0, 1, dx*dy, 1, dx, dx, dz);
         ++pmsg ;
      }
      if (rowMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, dx*(dy - 1), 1, dx*dy, 1, dx, dx, dz);
         ++pmsg ;
      }
   }
   if (colMin | colMax) {
      /* ASSUMING ONE DOMAIN PER RANK, CONSTANT BLOCK SIZE HERE */
      int sendCount = dy * dz ;

      if (colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, 0, dx, dx*dy, 1, dy, dy, dz);
         ++pmsg ;
      }
      if (colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z}] = CommData(
            pmsg, emsg, cmsg, dx - 1, dx, dx*dy, 1, dy, dy, dz);
         ++pmsg ;
      }
   }

   if (!planeOnly) {
      if (rowMin && colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, 0, dx*dy, 0, 1, 0, dz, 1);
         ++emsg ;
      }

      if (rowMin && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMin && planeMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, dx, 0, 1, 0, dy, 1);
         ++emsg ;
      }

      if (rowMax && colMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx*dy - 1, dx*dy, 0, 1, 0, dz, 1);
         ++emsg ;
      }

      if (rowMax && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy-1) + dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMax && planeMax && doSend) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz-1) + dx-1, dx, 0, 1, 0, dy, 1);
         ++emsg ;
      }

      if (rowMax && colMin && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), dx*dy, 0, 1, 0, dz, 1);
         ++emsg ;
      }

      if (rowMin && planeMax && doSend) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMin && planeMax && doSend) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), dx, 0, 1, 0, dy, 1);
         ++emsg ;
      }

      if (rowMin && colMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, dx-1, dx*dy, 0, 1, 0, dz, 1);
         ++emsg ;
      }

      if (rowMax && planeMin) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMax && planeMin) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx - 1, dx, 0, 1, 0, dy, 1);
         ++emsg ;
      }

      if (rowMin && colMin && planeMin) {
         /* corner at domain logical coord (0, 0, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin) {
         /* corner at domain logical coord (1, 0, 0) */
         Index_t idx = dx - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 0, 1) */
         Index_t idx = dx*dy*(dz - 1) + (dx - 1) ;
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin) {
         /* corner at domain logical coord (0, 1, 0) */
         Index_t idx = dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax && doSend) {
         /* corner at domain logical coord (0, 1, 1) */
         Index_t idx = dx*dy*(dz - 1) + dx*(dy - 1) ;
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin) {
         /* corner at domain logical coord (1, 1, 0) */
         Index_t idx = dx*dy - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax && doSend) {
         /* corner at domain logical coord (1, 1, 1) */
         Index_t idx = dx*dy*dz - 1 ;
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, idx, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
   }
}

void DomainChare::CommDataRecvInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
                      bool doRecv, bool planeOnly, CommDataMap_t &commDataMap) {

   if (domain.numRanks() == 1)
      return ;

   /* post recieve buffers for all incoming messages */
   Index_t maxPlaneComm =  domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = domain.maxEdgeSize() ;
   Index_t pmsg = 0 ; /* plane comm msg */
   Index_t emsg = 0 ; /* edge comm msg */
   Index_t cmsg = 0 ; /* corner comm msg */
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
   if (planeMin && doRecv) {
      /* contiguous memory */
      commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z-1}] = CommData(
         pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dx * dy, 1);
      ++pmsg ;
   }
   if (planeMax) {
      /* contiguous memory */
      commDataMap[{thisIndex.x, thisIndex.y, thisIndex.z+1}] = CommData(
         pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, dx * dy, 1);
      ++pmsg ;
   }
   if (rowMin && doRecv) {
      /* semi-contiguous memory */
      commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z}] = CommData(
         pmsg, emsg, cmsg, 0, 1, dx, 1, dx*dy, dx, dz);
      ++pmsg ;
   }
   if (rowMax) {
      /* semi-contiguous memory */
      commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z}] = CommData(
         pmsg, emsg, cmsg, dx*(dy - 1), 1, dx, 1, dx*dy, dx, dz);
      ++pmsg ;
   }
   if (colMin && doRecv) {
      /* scattered memory */
      commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z}] = CommData(
         pmsg, emsg, cmsg, 0, 1, dy, dx, dx*dy, dy, dz);
      ++pmsg ;
   }
   if (colMax) {
      /* scattered memory */
      commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z}] = CommData(
         pmsg, emsg, cmsg, dx - 1, 1, dy, dx, dx*dy, dy, dz);
      ++pmsg ;
   }

   if (!planeOnly) {
      /* receive data from domains connected only by an edge */
      if (rowMin && colMin && doRecv) {
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, dx*dy, 0, dz, 1);
         ++emsg ;
      }

      if (rowMin && planeMin && doRecv) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMin && planeMin && doRecv) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z-1}] =
            CommData(pmsg, emsg, cmsg, 0, 1, 0, dx, 0, dy, 1);
         ++emsg ;
      }

      if (rowMax && colMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z}] =
            CommData(pmsg, emsg, cmsg, dx*dy - 1, 1, 0, dx*dy, 0, dz, 1);
         ++emsg ;
      }

      if (rowMax && planeMax) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy-1) + dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMax && planeMax) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z+1}] =
            CommData(pmsg, emsg, cmsg, dx*dy*(dz-1) + dx-1, 1, 0, dx, 0, dy, 1);
         ++emsg ;
      }

      if (rowMax && colMin) {
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z}] =
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), 1, 0, dx*dy, 0, dz, 1);
         ++emsg ;
      }

      if (rowMin && planeMax) {
         commDataMap[{thisIndex.x, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz-1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMin && planeMax) {
         commDataMap[{thisIndex.x-1, thisIndex.y, thisIndex.z+1}] =
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, dx, 0, dy, 1);
         ++emsg ;
      }

      if (rowMin && colMax && doRecv) {
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z}] =
            CommData(pmsg, emsg, cmsg, dx-1, 1, 0, dx*dy, 0, dz, 1);
         ++emsg ;
      }

      if (rowMax && planeMin && doRecv) {
         commDataMap[{thisIndex.x, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), 1, 0, 1, 0, dx, 1);
         ++emsg ;
      }

      if (colMax && planeMin && doRecv) {
         commDataMap[{thisIndex.x+1, thisIndex.y, thisIndex.z-1}] =
            CommData(pmsg, emsg, cmsg, dx - 1, 1, 0, dx, 0, dy, 1);
         ++emsg ;
      }

      /* receive data from domains connected only by a corner */
      if (rowMin && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 0, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, 0, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMin && planeMax) {
         /* corner at domain logical coord (0, 0, 1) */
         commDataMap[{thisIndex.x-1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1), 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 0, 0) */
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx - 1, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMin && colMax && planeMax) {
         /* corner at domain logical coord (1, 0, 1) */
         commDataMap[{thisIndex.x+1, thisIndex.y-1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1) + (dx - 1), 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMin && doRecv) {
         /* corner at domain logical coord (0, 1, 0) */
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx*(dy - 1), 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMin && planeMax) {
         /* corner at domain logical coord (0, 1, 1) */
         commDataMap[{thisIndex.x-1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*(dz - 1) + dx*(dy - 1), 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMin && doRecv) {
         /* corner at domain logical coord (1, 1, 0) */
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z-1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy - 1, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
      }
      if (rowMax && colMax && planeMax) {
         /* corner at domain logical coord (1, 1, 1) */
         commDataMap[{thisIndex.x+1, thisIndex.y+1, thisIndex.z+1}] = 
            CommData(pmsg, emsg, cmsg, dx*dy*dz - 1, 1, 0, 1, 0, 1, 1);
         ++cmsg ;
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

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommDataMapIter_t it;

   // DBG_PRINTF("[DEBUG CommSend] (%d,%d,%d) iter=%u msgType=0x%x xferFields=%d commDataMap.size()=%lu\n",
   //    thisIndex.x, thisIndex.y, thisIndex.z, iter, msgType, xferFields, (unsigned long)commDataMap.size());

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

   std::string debug_string = "";

   for (it = commDataMap.begin(); it != commDataMap.end(); ++it) {
      std::tuple<int, int, int> idx = it->first ;
      CommData cdata = it->second ;
      int offsetX = std::get<0>(idx) - thisIndex.x ;
      int offsetY = std::get<1>(idx) - thisIndex.y ;
      int offsetZ = std::get<2>(idx) - thisIndex.z ;

      int offset = cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL; 

      if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) ||
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy2D(src, cdata.offset, cdata.src_stride[0], cdata.src_stride[1],
                   domain.commDataSendView, 
                   offset + fi * cdata.size[0] * cdata.size[1], 
                   cdata.dst_stride[0], cdata.dst_stride[1], 
                   cdata.size[0], cdata.size[1], commSpace);//TODO:: sync.use event for correctness, i think work has been on computeSpace before this(current this is okay as they are the same)

         if(false && thisIndex.x==1 && thisIndex.y==0 && thisIndex.z==0 && msgType == MSG_SYNC_POS_VEL && std::get<0>(idx)==0 && std::get<1>(idx)==0 && std::get<2>(idx)==0) {
            commSpace.fence();
            if(fi==0||fi==1)
            {
               for(int i=0;i<cdata.size[0]; ++i){
                  for(int j=0;j<cdata.size[1];j++){
                     auto h_src = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
                     int dst_idx = offset + fi * cdata.size[0] * cdata.size[1] + j*cdata.dst_stride[1] + i*cdata.dst_stride[0];
                     int src_idx = cdata.offset + j*cdata.src_stride[1] + i*cdata.src_stride[0];
                     int offset_dst = dst_idx - offset;
                     if(offset_dst==30||offset_dst==31||offset_dst==66||offset_dst==67){
                        printf("dst_idx: %d, src_idx: %d, value: %f ,,,,,,,,,,,", dst_idx, src_idx, h_src[src_idx]);
                     }
                  }
               }
               auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataSendView);
               //look at idx 30,31,66,67
               printf("[1,0,0] copying into h_send:");
               for(int i=0;i<xferFields * cdata.size[0] * cdata.size[1]; ++i) {
                  if(i==30||i==31||i==66||i==67)
                     printf(",,,,,,,,,, %d: %f ",i, h_send[offset + i]);
               }
               printf("\n");
            }
         }
         }
      } else {
         for (Index_t fi=0 ; fi<xferFields; ++fi) {
            Kokkos::View<Real_t*> src = fieldData[fi] ;
            Copy1D(src, cdata.offset, cdata.src_stride[0],
                   domain.commDataSendView, 
                   offset + fi * cdata.size[0], 
                   cdata.dst_stride[0], cdata.size[0], commSpace);
                   if(false && thisIndex.x==1 && thisIndex.y==0 && thisIndex.z==0 && msgType == MSG_SYNC_POS_VEL && std::get<0>(idx)==0 && std::get<1>(idx)==0 && std::get<2>(idx)==0) {
                      commSpace.fence();
                      if(fi==0||fi==1)
                      {
                        for(int i=0;i<cdata.size[0]; ++i){
                           auto h_src = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), src);
                           int dst_idx = offset + fi * cdata.size[0] + i*cdata.dst_stride[0];
                           int src_idx = cdata.offset + i*cdata.src_stride[0];
                           int offset_dst = dst_idx - offset;
                           if(offset_dst==30||offset_dst==31||offset_dst==66||offset_dst==67){
                              printf("dst_idx: %d, src_idx: %d, value: %f ,,,,,,,, ", dst_idx, src_idx, h_src[src_idx]);
                           }
                        }
                         auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataSendView);
                         //look at idx 30,31,66,67
                         printf("[1,0,0] copying into h_send:");
                         for(int i=0;i<xferFields * cdata.size[0] * cdata.size[1]; ++i) {
                            if(i==30||i==31||i==66||i==67)
                               printf(",,,,,,,,,, %d: %f ",i, h_send[offset + i]);
                         }
                         printf("\n");
                      }
                   }
         }
      }

      // commSpace.fence();

      // CkCallback* cb = new CkCallback(CkIndex_DomainChare::packingDone(NULL), thisProxy[thisIndex]);
      int sendCount = xferFields * cdata.size[0] * cdata.size[1];
      PackingDoneMsg* msg = new PackingDoneMsg(msgType,
         iter,
         std::get<0>(idx), std::get<1>(idx), std::get<2>(idx),
         xferFields, sendCount, offset);
      
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

      Real_t* sendPtr = locDom->commDataSendView.data() + msg->offset;

      if(false && thisIndex.x==0 && thisIndex.y==0 && thisIndex.z==1 && msgType == MSG_SYNC_POS_VEL && std::get<0>(idx)==1 && std::get<1>(idx)==0 && std::get<2>(idx)==0) {
         commSpace.fence();
         auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataSendView);
         //look at idx 0,1,2,6,7,8
         printf("[0,0,1] to [1,0,0]: ptr %p ", sendPtr);
         for(int i=0;i<msg->sendCount; ++i) {
            // if(i==0||i==1||i==2||i==6||i==7||i==8){
               printf(" %f ", h_send[msg->offset + i]);
               //print the corresponding source values as well
            // }
         }
         printf("\n");
      }

         #if DEBUG_COMM
       // Sender-side debug: print source field values at packed positions for PosVel z-face
      // if(msg->msgType == MSG_SYNC_POS_VEL)
      // {
      //    auto commDataMap = commDataSendPosVel;
      
      //    CommData cdata = commDataMap[{msg->x, msg->y, msg->z}];
      
      //    if ( msg->x== 0 && msg->y== 0 && msg->z == 0) {
      //       commSpace.fence(); // ensure packing kernels done before host read
      //       auto h_send = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), locDom->commDataSendView);
      //       printf(" [Send to (0,0,0) from (%d,%d,%d)]: ", thisIndex.x, thisIndex.y, thisIndex.z);
      //       for(int i=0; i<msg->sendCount; ++i) {
      //          printf("%f ", h_send[msg->offset + i]);
      //       }
      //       printf("\n");
      //    }
      // }
      #endif

      thisProxy(msg->x, msg->y, msg->z).CommRecv(ref, thisIndex.x, thisIndex.y, thisIndex.z, 
      msg->xferFields, msg->sendCount, CkDeviceBuffer(sendPtr, *cb, commStream));
      
      // DBG_PRINTF("[DEBUG CommSend packing] (%d,%d,%d)->(%d,%d,%d) iter=%u msgType=0x%x xferFields=%d sendCount=%d offset=%d\n",
      //    thisIndex.x, thisIndex.y, thisIndex.z,
      //    std::get<0>(idx), std::get<1>(idx), std::get<2>(idx),
      //    iter, msgType, xferFields, sendCount, offset);

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

   Real_t* sendPtr = locDom->commDataSendView.data() + msg->offset;

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
   Kokkos::View<Real_t*> recvView;
   if (msgType == MSG_SYNC_POS_VEL) {
      commDataMap = &commDataRecvPosVel;
      recvView = locDom->commDataRecvViewPosVel;
   }
   else if (msgType == MSG_MONOQ) {
      commDataMap = &commDataRecvMonoQ;
      recvView = locDom->commDataRecvViewMonoQ;
   }
   else if (msgType == MSG_COMM_SBN) {
      commDataMap = &commDataRecvSBN;
      recvView = locDom->commDataRecvViewSBN;
   }
   else
      CkAbort("DomainChare::CommRecv: Unknown msgType") ;

   CommDataMapIter_t it = commDataMap->find({x, y, z});
   if (it == commDataMap->end()) {
      CkAbort("DomainChare::CommRecv: Invalid comm data map key") ;
   }

   Index_t maxPlaneComm = xferFields * locDom->maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * locDom->maxEdgeSize() ;

   int pmsg = it->second.pmsg ;
   int emsg = it->second.emsg ;
   int cmsg = it->second.cmsg ;

   int offset = pmsg * maxPlaneComm + emsg * maxEdgeComm + cmsg * CACHE_COHERENCE_PAD_REAL;
   // if(flatIndex==1 && msgType == MSG_SYNC_POS_VEL)
   // {
      // auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), recvView);
      // if(x==0 && y==0 && z==1){
      // printf(" prise be ");
      //    for(int i=0; i<36; ++i) {
      //          printf(" %f ", h_recv[768 + i]);
      //    }
      // printf("\n");
      // }
      // printf(" [start] CommRecv at (%d,%d,%d) from (%d,%d,%d) with offset %d with size %d\n", thisIndex.x, thisIndex.y, thisIndex.z, x, y, z, offset, size);
   // }

   buf = recvView.data() + offset;
   post[0].hapi_stream = commStream;
   // std::ostringstream os;
   os<<" [end] CommRecv ";
   NVTXTracer(os.str(), NVTXColor::PeterRiver);
   os.clear();
}

/******************************************/

void DomainChare::processRemotePosVel(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;
   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize();
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize();

   Kokkos::View<Real_t*> fieldData[6];
   fieldData[0] = domain.m_x;
   fieldData[1] = domain.m_y;
   fieldData[2] = domain.m_z;
   fieldData[3] = domain.m_xd;
   fieldData[4] = domain.m_yd;
   fieldData[5] = domain.m_zd;

   Real_t* h_recv_buf;
   cudaMallocHost(&h_recv_buf, size * sizeof(Real_t));
   cudaMemcpy(h_recv_buf, buf, size * sizeof(Real_t), cudaMemcpyDeviceToHost);
   cudaDeviceSynchronize();
   auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataRecvViewPosVel);

   int sx = x;
   int sy = y;
   int sz = z;

   CommData& cdata = commDataRecvPosVel[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   int offset = cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL;

   if(false && flatIndex==1 && x==0 && y==0 && z==1)
   {
      printf("[1,0,0] from [0,0,1]: ptr %p ", domain.commDataRecvViewPosVel.data() + offset);
      for(int i=0; i<size; ++i) {
            printf(" %f ", h_recv[offset + i]);
      }
      printf("\n");
   }

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) ||
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         Kokkos::View<Real_t*>& dest = fieldData[fi];
         Copy2D(domain.commDataRecvViewPosVel,
            offset + fi * cdata.size[0] * cdata.size[1],
            1, cdata.size[0],
            dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
            cdata.size[0], cdata.size[1], commSpace);
            if(false && flatIndex==1 && (fi==0 || fi==1))
            {
               for(int i=0;i<cdata.size[0];i++)
                  for(int j=0;j<cdata.size[1];j++)
               {
                  int idx = cdata.offset + i * cdata.dst_stride[0] + j * cdata.dst_stride[1];
                  if(idx==180 || idx==186 || idx==192)
                  {
                     Kokkos::fence();
                     printf("[Copy2D dest idx %d from (%d,%d,%d)] from offset %d\n", idx, x, y, z, offset);
                     //incoming values
                     auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataRecvViewPosVel);
                     int srcIdx = offset + fi * cdata.size[0] * cdata.size[1] + i*1 + j*cdata.size[0];
                     printf(" [copy2D copying into idx %d from srcIdx %d(value %.6f)]\n", idx, srcIdx - offset, h_recv[srcIdx]);
                     //field values after copy2D
                     auto h_dest = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dest);
                     printf(" [copy2D copied into idx %d value %.6f]\n", idx, h_dest[idx]);
                  }
               }
            }
         }
   } else {
      for (Index_t fi = 0; fi < xferFields; ++fi) {
         Kokkos::View<Real_t*>& dest = fieldData[fi];
         Copy1D(domain.commDataRecvViewPosVel,
            offset + fi * cdata.size[0],
            1,
            dest, cdata.offset, cdata.dst_stride[0],
            cdata.size[0], commSpace);
         if(false && flatIndex==1 && (fi==0 || fi==1))
            for(int i=0; i<cdata.size[0]; ++i) {
               int idx = cdata.offset + i * cdata.dst_stride[0];
               if(idx==180 || idx==186 || idx==192)
               {
                  Kokkos::fence();
                  printf("[Copy1D dest idx %d from (%d,%d,%d)] from offset %d\n", idx, x, y, z, offset);
                  //incoming values
                  auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.commDataRecvViewPosVel);
                  int srcIdx = offset + fi * cdata.size[0] + i;
                  printf(" [copy1D copying into idx %d from srcIdx %d(value %.6f)]\n", idx, srcIdx - offset, h_recv[srcIdx]);
                  //field values after copy1D
                  auto h_dest = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dest);
                  printf(" [copy1D copied into idx %d value %.6f]\n", idx, h_dest[idx]);
               }
            }
      }
   }
   #if DEBUG_COMM
   // Diagnostic: check recv buffer and domain field values after unpack, for flatIndex 0
   if (false && flatIndex==1) {
      commSpace.fence();
      auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.m_x);
      auto h_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), domain.m_y);
      
      //pick 180, 186, 192, 198, 204, 204, 210

      printf("[180 from (%d,%d,%d)] x=%.6f y=%.6f\n", x, y, z, h_x[180], h_y[180]);
      printf("[186 from (%d,%d,%d)] x=%.6f y=%.6f\n", x, y, z, h_x[186], h_y[186]);
      printf("[192 from (%d,%d,%d)] x=%.6f y=%.6f\n", x, y, z, h_x[192], h_y[192]);
      // auto h_recv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), locDom->commDataRecvViewPosVel);
      // CommData& cd = commDataRecvPosVel[{x, y, z}];
      // int maxPlaneComm = xferFields * locDom->maxPlaneSize();
      // int maxEdgeComm = xferFields * locDom->maxEdgeSize();
      // int off = cd.pmsg * maxPlaneComm + cd.emsg * maxEdgeComm + cd.cmsg * CACHE_COHERENCE_PAD_REAL;
      // int count = size;
      // printf("[RECV chare(%d,%d,%d) from(%d,%d,%d)] ",
      //    thisIndex.x, thisIndex.y, thisIndex.z, x, y, z,
      //    off, cd.offset, cd.dst_stride[0], count);
      // for(int i=0;i<count;i++)
      // {
      //    printf("%.6f ",h_recv(off + i));
      // }
      // printf("\n");
   }
   #endif
}
/******************************************/

void DomainChare::processRemoteQ(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;
   // commSpace.fence();

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommData cdata = commDataRecvMonoQ[{x, y, z}];
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

   int offset = cdata.pmsg * maxPlaneComm;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      int copyLen = cdata.size[0] * cdata.size[1];
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         int srcOff = offset + fi * copyLen;
         int dstOff = fieldOffset[fi] + cdata.pmsg * copyLen;
         Copy1D(domain.commDataRecvViewMonoQ, srcOff, 1,
            dest, dstOff, 1, copyLen, commSpace);
      }
   } else {
      int copyLen = cdata.size[0];
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi];
         int srcOff = offset + fi * copyLen;
         int dstOff = fieldOffset[fi] + cdata.pmsg * copyLen;
         Copy1D(domain.commDataRecvViewMonoQ, srcOff, 1,
            dest, dstOff, 1, copyLen, commSpace);
      }
   }
   // commSpace.fence();
}

/******************************************/

void DomainChare::processRemoteMass(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
   Domain& domain = *locDom;
   // commSpace.fence();

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommData cdata = commDataRecvSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[1];
   fieldData[0] = domain.m_nodalMass;

   int offset = cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add2D(domain.commDataRecvViewSBN,
               offset + fi * cdata.size[0] * cdata.size[1],
               1, cdata.size[0],
               dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
               cdata.size[0], cdata.size[1], commSpace);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> dest = fieldData[fi] ;
         Add1D(domain.commDataRecvViewSBN,
               offset + fi * cdata.size[0],
               1,
               dest, cdata.offset, cdata.dst_stride[0],
               cdata.size[0], commSpace);
      }
   }
   // commSpace.fence();
}

/******************************************/

void DomainChare::processRemoteForce(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf) {
      Domain& domain = *locDom;

   Index_t maxPlaneComm = xferFields * domain.maxPlaneSize() ;
   Index_t maxEdgeComm  = xferFields * domain.maxEdgeSize() ;

   CommData cdata = commDataRecvSBN[{x, y, z}];
   Index_t offsetX = x - thisIndex.x;
   Index_t offsetY = y - thisIndex.y;
   Index_t offsetZ = z - thisIndex.z;

   Kokkos::View<Real_t*> fieldData[3];
   fieldData[0] = domain.m_fx;
   fieldData[1] = domain.m_fy;
   fieldData[2] = domain.m_fz;

   int offset = cdata.pmsg * maxPlaneComm + cdata.emsg * maxEdgeComm + cdata.cmsg * CACHE_COHERENCE_PAD_REAL;

   if (((offsetX == -1 || offsetX == 1) && offsetY == 0 && offsetZ == 0) || 
         offsetX == 0 && ((offsetY == -1 || offsetY == 1) && offsetZ == 0)) {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         Add2D(domain.commDataRecvView, offset + fi * cdata.size[0] * cdata.size[1],
               1, cdata.size[0],
               dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
               cdata.size[0], cdata.size[1], commSpace);
         // Add2D(dest, cdata.offset, cdata.dst_stride[0], cdata.dst_stride[1],
         //       domain.commDataRecvView, 
         //       offset + fi * cdata.size[0] * cdata.size[1],
         //       cdata.src_stride[0], cdata.src_stride[1],
         //       cdata.size[0], cdata.size[1], commSpace);
      }
   } else {
      for (Index_t fi=0 ; fi<xferFields; ++fi) {
         Kokkos::View<Real_t*> &dest = fieldData[fi] ;
         Add1D(domain.commDataRecvView, offset + fi * cdata.size[0],
               1,
               dest, cdata.offset, cdata.dst_stride[0],
               cdata.size[0], commSpace);
         // Add1D(dest, cdata.offset, cdata.dst_stride[0],
         //       domain.commDataRecvView, 
         //       offset + fi * cdata.size[0],
         //       cdata.src_stride[0],
         //       cdata.size[0], commSpace);
      }
   }
}