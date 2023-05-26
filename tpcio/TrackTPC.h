// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_TRACKTPC
#define ALICEO2_TPC_TRACKTPC

#include "Track.h"
#include "RangeReference.h"
#include "dEdxInfo.h"

namespace o2
{
namespace tpc
{
/// \class TrackTPC
/// This is the definition of the TPC Track Object

using TPCClRefElem = uint32_t;
  
class TrackTPC : public o2::track::TrackParCov
{
  using ClusRef = o2::dataformats::RangeReference<uint32_t, uint16_t>;

 public:
  enum Flags : unsigned short {
    HasASideClusters = 0x1 << 0,                                ///< track has clusters on A side
    HasCSideClusters = 0x1 << 1,                                ///< track has clusters on C side
    HasBothSidesClusters = HasASideClusters | HasCSideClusters, // track has clusters on both sides
    FullMask = 0xffff
  };

  using o2::track::TrackParCov::TrackParCov; // inherit

  /// Default constructor
  TrackTPC() = default;

  /// Destructor
  ~TrackTPC() = default;

  unsigned short getFlags() const { return mFlags; }
  unsigned short getClustersSideInfo() const { return mFlags & HasBothSidesClusters; }
  bool hasASideClusters() const { return mFlags & HasASideClusters; }
  bool hasCSideClusters() const { return mFlags & HasCSideClusters; }
  bool hasBothSidesClusters() const { return (mFlags & (HasASideClusters | HasCSideClusters)) == (HasASideClusters | HasCSideClusters); }
  bool hasASideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasASideClusters; }
  bool hasCSideClustersOnly() const { return (mFlags & HasBothSidesClusters) == HasCSideClusters; }

  void setHasASideClusters() { mFlags |= HasASideClusters; }
  void setHasCSideClusters() { mFlags |= HasCSideClusters; }

  float getTime0() const { return mTime0; }         ///< Reference time of the track, i.e. t-bins of a primary track with eta=0.
  float getDeltaTBwd() const { return mDeltaTBwd; } ///< max possible decrement to getTimeVertex
  float getDeltaTFwd() const { return mDeltaTFwd; } ///< max possible increment to getTimeVertex
  void setDeltaTBwd(float t) { mDeltaTBwd = t; }    ///< set max possible decrement to getTimeVertex
  void setDeltaTFwd(float t) { mDeltaTFwd = t; }    ///< set max possible increment to getTimeVertex

  float getChi2() const { return mChi2; }
  const o2::track::TrackParCov& getOuterParam() const { return mOuterParam; }
  const o2::track::TrackParCov& getParamOut() const { return mOuterParam; } // to have method with same name as other tracks
  o2::track::TrackParCov& getParamOut() { return mOuterParam; }             // to have method with same name as other tracks
  void setTime0(float v) { mTime0 = v; }
  void setChi2(float v) { mChi2 = v; }
  void setOuterParam(o2::track::TrackParCov&& v) { mOuterParam = v; }
  void setParamOut(o2::track::TrackParCov&& v) { mOuterParam = v; } // to have method with same name as other tracks
  const ClusRef& getClusterRef() const { return mClustersReference; }
  void shiftFirstClusterRef(int dif) { mClustersReference.setFirstEntry(dif + mClustersReference.getFirstEntry()); }
  int getNClusters() const { return mClustersReference.getEntries(); }
  int getNClusterReferences() const { return getNClusters(); }
  void setClusterRef(uint32_t entry, uint16_t ncl) { mClustersReference.set(entry, ncl); }

  const dEdxInfo& getdEdx() const { return mdEdx; }
  void setdEdx(const dEdxInfo& v) { mdEdx = v; }

 private:
  float mTime0 = 0.f;                 ///< Assumed time of the vertex that created the track in TPC time bins, 0 for triggered data
  float mDeltaTFwd = 0;               ///< max possible increment to mTime0
  float mDeltaTBwd = 0;               ///< max possible decrement to mTime0
  short mFlags = 0;                   ///< various flags, see Flags enum
  float mChi2 = 0.f;                  // Chi2 of the track
  o2::track::TrackParCov mOuterParam; // Track parameters at outer end of TPC.
  dEdxInfo mdEdx;                     // dEdx Information
  ClusRef mClustersReference;         // reference to externale cluster indices

  ClassDefNV(TrackTPC, 4);
};

} // namespace tpc
} // namespace o2

#endif
