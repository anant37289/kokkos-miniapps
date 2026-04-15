#ifndef LULESH_H
#define LULESH_H

#define DEBUG_COMM 0     // verbose per-message prints (changes timing!)
#define DEBUG_COMM_LITE 0  // lightweight phase-transition prints only
#if DEBUG_COMM
#define DBG_PRINTF(...) CkPrintf(__VA_ARGS__)
#else
#define DBG_PRINTF(...) do {} while(0)
#endif
#if DEBUG_COMM_LITE
#define DBG_LITE(...) CkPrintf(__VA_ARGS__)
#else
#define DBG_LITE(...) do {} while(0)
#endif

#include <unordered_map>
#include <tuple>
#include <functional>

// OpenMP will be compiled in if this flag is set to 1 AND the compiler beging
// used supports it (i.e. the _OPENMP symbol is defined)

//#include "lulesh-domain.h"
#include "lulesh.decl.h"
#include "hapi.h"
#include "hapi_nvtx.h"

extern CProxy_Main mainProxy;

struct TupleHash {
    template <class T1, class T2, class T3>
    std::size_t operator()(const std::tuple<T1, T2, T3>& v) const {
        // Use a basic hash-combining strategy
        auto h1 = std::hash<T1>{}(std::get<0>(v));
        auto h2 = std::hash<T2>{}(std::get<1>(v));
        auto h3 = std::hash<T3>{}(std::get<2>(v));
        
        // A common way to combine hashes to avoid collisions
        return h1 ^ (h2 << 1) ^ (h3 << 2); 
    }
};

using CommDataMap_t = std::unordered_map<std::tuple<int, int, int>, CommData, TupleHash>;
using CommDataMapIter_t = CommDataMap_t::iterator;

class KokkosManager : public CBase_KokkosManager {
public:
  KokkosManager() : CBase_KokkosManager() {
    Kokkos::initialize();
  }

  KokkosManager(CkMigrateMessage *msg) : CBase_KokkosManager(msg) {
    Kokkos::initialize();
  }

  ~KokkosManager() {
  }

  void finalize();
};

class DomainChare : public CBase_DomainChare {
  DomainChare_SDAG_CODE

public:
  DomainChare(CkMigrateMessage *msg);

  DomainChare(int numRanks, Index_t nx_, int nr_,
              int balance_, int cost_, int showProg_, int quiet_,
              int its_, int viz_, int do_atomic_, int lb_every,
              int numChares_);

  ~DomainChare() { delete locDom; }

  void CommDataSendInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t &commDataMap);

  void CommDataRecvInit(Domain& domain, Index_t dx, Index_t dy, Index_t dz, 
    bool doRecv, bool planeOnly, CommDataMap_t &commDataMap);

  void CommSend(Domain& domain, int msgType, Index_t xferFields, 
    Kokkos::View<Real_t*> *fieldData, Index_t dx, Index_t dy, Index_t dz, 
    bool doSend, bool planeOnly, CommDataMap_t& commDataMap);

  //void CommRecv(int& ref, int& x, int& y, int& z, int& xferFields, int& size, 
  //  Real_t* &buf, CkDeviceBufferPost* post);

  Real_t TimeStepCalculateLocal(Domain &domain);

  void TimeIncrement(Real_t newdt);

  void CommRecv(uint32_t ref, int x, int y, int z, int xferFields, int& size, Real_t* &buf, CkDeviceBufferPost* post);

  // void PosVelSendDone();
  // void MonoQSendDone();
  // void SBNSendDone();

  void PosVelSendCallback();
  void MonoQSendCallback();
  void SBNSendCallback();

  void packingDone(PackingDoneMsg* msg);

  void processRemotePosVel(uint32_t ref, int x, int y, int z, int xferFields, 
    int size, Real_t* buf);

  void processRemoteQ(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  void processRemoteMass(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  void processRemoteForce(uint32_t ref, int x, int y, int z, int xferFields, int size, Real_t* buf);

  void ResumeFromSync()
  {
    CkCallback cb_done(CkReductionTarget(Main, endLB), mainProxy);
    contribute(cb_done);
    // thisProxy[thisIndex].run(); //the above calls 
  }

  void pup(PUP::er &p)
  {
    p| locDom;

    p| iter;
    p| flatIndex;
    p| commNbrs;
    p| remoteCount;
    p| numChares;
    p| recvRef;
    p| opts;
    p| startTime;
    p| posVelSendsDone;
    p| monoQSendsDone;
    p| sbnSendsDone;

    p| commDataSendPosVel;
    p| commDataSendMonoQ;
    p| commDataSendSBN;

    p| commDataRecvPosVel;
    p| commDataRecvMonoQ;
    p| commDataRecvSBN;
  }


  Domain *locDom;
  uint32_t iter;
  int flatIndex;
  int commNbrs, remoteCount;
  int numChares;
  uint32_t recvRef;
  struct cmdLineOpts opts;
  double startTime;

  int posVelSendsDone ;
  int monoQSendsDone ;
  int sbnSendsDone ;

  CommDataMap_t commDataSendPosVel;
  CommDataMap_t commDataSendMonoQ;
  CommDataMap_t commDataSendSBN;

  CommDataMap_t commDataRecvPosVel;
  CommDataMap_t commDataRecvMonoQ;
  CommDataMap_t commDataRecvSBN;

  hapiStream_t commStream, computeStream;
  ExecSpace commSpace, computeSpace;
};

class PackingDoneMsg : public CMessage_PackingDoneMsg {
public:
  uint32_t msgType;
  uint32_t sendIter;  // iter value captured at CommSend time (not when packingDone fires)
  int x, y, z;
  int xferFields, sendCount, offset;

  PackingDoneMsg(uint32_t msgType_, uint32_t sendIter_, int x_, int y_, int z_,
                 int xferFields_, int sendCount_, int offset_)
      : msgType(msgType_), sendIter(sendIter_), x(x_), y(y_), z(z_),
        xferFields(xferFields_), sendCount(sendCount_), offset(offset_) {}
};

class Main : public CBase_Main {
  Main_SDAG_CODE
  struct cmdLineOpts opts;
  double last_lb_start = 0;

public:
  Main(CkArgMsg *m);

  double start;
};

#endif // LULESH_H
