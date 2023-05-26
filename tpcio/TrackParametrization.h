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

/// @file   TrackParametrization.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

/*
  24/09/2020: Added new data member for abs. charge. This is needed for uniform treatment of tracks with non-standard
  charge: 0 (for V0s) and e.g. 2 for hypernuclei.
  In the aliroot AliExternalTrackParam this was treated by derived classes using virtual methods, which we don't use in O2.
  The meaning of mP[kQ2Pt] remains exactly the same, except for q=0 case: in this case the mP[kQ2Pt] is just an alias to
  1/pT, regardless of its sign, and the getCurvature() will 0 (because mAbsCharge is 0).
  The methods returning lab momentum or its combination account for eventual q>1.
 */

#ifndef INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_
#define INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_

#include <Rtypes.h>
#include <TMath.h>
#include "PID.h"
#include "MathConstants.h"

namespace o2
{

namespace track
{
// aliases for track elements
enum ParLabels : int { kY,
                       kZ,
                       kSnp,
                       kTgl,
                       kQ2Pt };
enum CovLabels : int {
  kSigY2,
  kSigZY,
  kSigZ2,
  kSigSnpY,
  kSigSnpZ,
  kSigSnp2,
  kSigTglY,
  kSigTglZ,
  kSigTglSnp,
  kSigTgl2,
  kSigQ2PtY,
  kSigQ2PtZ,
  kSigQ2PtSnp,
  kSigQ2PtTgl,
  kSigQ2Pt2
};

enum DirType : int { DirInward = -1,
                     DirAuto = 0,
                     DirOutward = 1 };

constexpr int kNParams = 5, kCovMatSize = 15, kLabCovMatSize = 21;

constexpr float kCY2max = 100 * 100, // SigmaY<=100cm
  kCZ2max = 100 * 100,               // SigmaZ<=100cm
  kCSnp2max = 1 * 1,                 // SigmaSin<=1
  kCTgl2max = 1 * 1,                 // SigmaTan<=1
  kC1Pt2max = 100 * 100,             // Sigma1/Pt<=100 1/GeV
  kMostProbablePt = 0.6f,            // Most Probable Pt (GeV), for running with Bz=0
  kCalcdEdxAuto = -999.f;            // value indicating request for dedx calculation

// access to covariance matrix by row and column
constexpr int CovarMap[kNParams][kNParams] = {{0, 1, 3, 6, 10},
                                                   {1, 2, 4, 7, 11},
                                                   {3, 4, 5, 8, 12},
                                                   {6, 7, 8, 9, 13},
                                                   {10, 11, 12, 13, 14}};

// access to covariance matrix diagonal elements
constexpr int DiagMap[kNParams] = {0, 2, 5, 9, 14};

constexpr float HugeF = o2::constants::math::VeryBig;

template <typename value_T = float>
class TrackParametrization
{ // track parameterization, kinematics only.

 public:
  using value_t = value_T;
  using dim2_t = std::array<value_t, 2>;
  using dim3_t = std::array<value_t, 3>;
  using params_t = std::array<value_t, kNParams>;

  struct yzerr_t { // 2 measurement with error
    dim2_t yz;
    dim3_t yzerr;
  };

#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_t>);
#endif

  TrackParametrization() = default;
  TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge = 1, const PID pid = PID::Pion);
  TrackParametrization(const dim3_t& xyz, const dim3_t& pxpypz, int charge, bool sectorAlpha = true, const PID pid = PID::Pion);
  TrackParametrization(const TrackParametrization&) = default;
  TrackParametrization(TrackParametrization&&) = default;
  TrackParametrization& operator=(const TrackParametrization& src) = default;
  TrackParametrization& operator=(TrackParametrization&& src) = default;
  ~TrackParametrization() = default;

  void set(value_t x, value_t alpha, const params_t& par, int charge = 1, const PID pid = PID::Pion);
  void set(value_t x, value_t alpha, const value_t* par, int charge = 1, const PID pid = PID::Pion);
  const value_t* getParams() const;
  value_t getParam(int i) const;
  value_t getX() const;
  value_t getAlpha() const;
  value_t getY() const;
  value_t getZ() const;
  value_t getSnp() const;
  value_t getTgl() const;
  value_t getQ2Pt() const;
  value_t getCharge2Pt() const;
  int getAbsCharge() const;
  PID getPID() const;
  void setPID(const PID pid);

  /// calculate cos^2 and cos of track direction in rphi-tracking
  value_t getCsp2() const;
  value_t getCsp() const;

  void setX(value_t v);
  void setParam(value_t v, int i);
  void setAlpha(value_t v);
  void setY(value_t v);
  void setZ(value_t v);
  void setSnp(value_t v);
  void setTgl(value_t v);
  void setQ2Pt(value_t v);
  void setAbsCharge(int q);

  value_t getCurvature(value_t b) const;
  int getCharge() const;
  int getSign() const;
  value_t getPhi() const;
  value_t getPhiPos() const;

  value_t getPtInv() const;
  value_t getP2Inv() const;
  value_t getP2() const;
  value_t getPInv() const;
  value_t getP() const;
  value_t getPt() const;

  value_t getTheta() const;
  value_t getEta() const;

  uint16_t getUserField() const;
  void setUserField(uint16_t v);

  void printParam() const;
  std::string asString() const;

 private:
  //
  static constexpr value_t InvalidX = -99999.f;
  value_t mX = 0.f;             /// X of track evaluation
  value_t mAlpha = 0.f;         /// track frame angle
  value_t mP[kNParams] = {0.f}; /// 5 parameters: Y,Z,sin(phi),tg(lambda),q/pT
  char mAbsCharge = 1;          /// Extra info about the abs charge, to be taken into account only if not 1
  PID mPID{PID::Pion};          /// 8 bit PID
  uint16_t mUserField = 0;      /// field provided to user

  ClassDefNV(TrackParametrization, 3);
};

//____________________________________________________________
template <typename value_T>
TrackParametrization<value_T>::TrackParametrization(value_t x, value_t alpha, const params_t& par, int charge, const PID pid)
  : mX{x}, mAlpha{alpha}, mAbsCharge{char(std::abs(charge))}, mPID{pid}
{
  // explicit constructor
  for (int i = 0; i < kNParams; i++) {
    mP[i] = par[i];
  }
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::set(value_t x, value_t alpha, const params_t& par, int charge, const PID pid)
{
  set(x, alpha, par.data(), charge, pid);
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::set(value_t x, value_t alpha, const value_t* par, int charge, const PID pid)
{
  mX = x;
  mAlpha = alpha;
  mAbsCharge = char(std::abs(charge));
  for (int i = 0; i < kNParams; i++) {
    mP[i] = par[i];
  }
  mPID = pid;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getParams() const -> const value_t*
{
  return mP;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getParam(int i) const -> value_t
{
  return mP[i];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getX() const -> value_t
{
  return mX;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getAlpha() const -> value_t
{
  return mAlpha;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getY() const -> value_t
{
  return mP[kY];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getZ() const -> value_t
{
  return mP[kZ];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getSnp() const -> value_t
{
  return mP[kSnp];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getTgl() const -> value_t
{
  return mP[kTgl];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getQ2Pt() const -> value_t
{
  return mP[kQ2Pt];
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getCharge2Pt() const -> value_t
{
  return mAbsCharge ? mP[kQ2Pt] : 0.f;
}

//____________________________________________________________
template <typename value_T>
int TrackParametrization<value_T>::getAbsCharge() const
{
  return mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
PID TrackParametrization<value_T>::getPID() const
{
  return mPID;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setPID(const PID pid)
{
  mPID = pid;
  //  setAbsCharge(pid.getCharge()); // If needed, user should change the charge via corr. setter
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getCsp2() const -> value_t
{
  const value_t csp2 = (1.f - mP[kSnp]) * (1.f + mP[kSnp]);
  return csp2 > o2::constants::math::Almost0 ? csp2 : o2::constants::math::Almost0;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getCsp() const -> value_t
{
  return TMath::Sqrt(getCsp2());
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setX(value_t v)
{
  mX = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setParam(value_t v, int i)
{
  mP[i] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setAlpha(value_t v)
{
  mAlpha = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setY(value_t v)
{
  mP[kY] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setZ(value_t v)
{
  mP[kZ] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setSnp(value_t v)
{
  mP[kSnp] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setTgl(value_t v)
{
  mP[kTgl] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setQ2Pt(value_t v)
{
  mP[kQ2Pt] = v;
}

//____________________________________________________________
template <typename value_T>
void TrackParametrization<value_T>::setAbsCharge(int q)
{
  mAbsCharge = std::abs(q);
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getCurvature(value_t b) const -> value_t
{
  return mAbsCharge ? mP[kQ2Pt] * b * o2::constants::math::B2C : 0.;
}

//____________________________________________________________
template <typename value_T>
int TrackParametrization<value_T>::getCharge() const
{
  return getSign() > 0 ? mAbsCharge : -mAbsCharge;
}

//____________________________________________________________
template <typename value_T>
int TrackParametrization<value_T>::getSign() const
{
  return mAbsCharge ? (mP[kQ2Pt] > 0.f ? 1 : -1) : 0;
}

//_______________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getPhi() const -> value_t
{
  // track pt direction phi (in 0:2pi range)
  value_t phi = TMath::ASin(getSnp()) + getAlpha();
  return phi;
}

//_______________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getPhiPos() const -> value_t
{
  // angle of track position (in -pi:pi range)
  value_t phi = TMath::ATan2(getY(), getX()) + getAlpha();
  return phi;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getPtInv() const -> value_t
{
  // return the inverted track pT
  const value_t ptInv = std::abs(mP[kQ2Pt]);
  return (mAbsCharge > 1) ? ptInv / mAbsCharge : ptInv;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getP2Inv() const -> value_t
{
  // return the inverted track momentum^2
  const value_t p2 = mP[kQ2Pt] * mP[kQ2Pt] / (1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? p2 / (mAbsCharge * mAbsCharge) : p2;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getP2() const -> value_t
{
  // return the track momentum^2
  const value_t p2inv = getP2Inv();
  return (p2inv > o2::constants::math::Almost0) ? 1.f / p2inv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getPInv() const -> value_t
{
  // return the inverted track momentum^2
  const value_t pInv = std::abs(mP[kQ2Pt]) / TMath::Sqrt(1.f + getTgl() * getTgl());
  return (mAbsCharge > 1) ? pInv / mAbsCharge : pInv;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getP() const -> value_t
{
  // return the track momentum
  const value_t pInv = getPInv();
  return (pInv > o2::constants::math::Almost0) ? 1.f / pInv : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getPt() const -> value_t
{
  // return the track transverse momentum
  value_t ptI = std::abs(mP[kQ2Pt]);
  if (mAbsCharge > 1) {
    ptI /= mAbsCharge;
  }
  return (ptI > o2::constants::math::Almost0) ? 1.f / ptI : o2::constants::math::VeryBig;
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getTheta() const -> value_t
{
  return constants::math::PIHalf - TMath::ATan(mP[3]);
}

//____________________________________________________________
template <typename value_T>
auto TrackParametrization<value_T>::getEta() const -> value_t
{
  return -TMath::Log(TMath::Tan(0.5f * getTheta()));
}

} // namespace track
} // namespace o2

#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATION_H_ */
