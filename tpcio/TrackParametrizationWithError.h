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

/// @file   TrackParametrizationWithError.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  Oct 1, 2020
/// @brief

#ifndef INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_
#define INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_

#include "TrackParametrization.h"

namespace o2
{
namespace track
{

template <typename value_T = float>
class TrackParametrizationWithError : public TrackParametrization<value_T>
{ // track+error parameterization
 public:
  using typename TrackParametrization<value_T>::value_t;
  using typename TrackParametrization<value_T>::dim3_t;
  using typename TrackParametrization<value_T>::dim2_t;
  using typename TrackParametrization<value_T>::params_t;

#ifndef GPUCA_GPUCODE_DEVICE
  static_assert(std::is_floating_point_v<value_t>);
#endif

  using covMat_t = std::array<value_t, kCovMatSize>;

  TrackParametrizationWithError();
  TrackParametrizationWithError(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1, const PID pid = PID::Pion);
  TrackParametrizationWithError(const dim3_t& xyz, const dim3_t& pxpypz,
                                       const std::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true, const PID pid = PID::Pion);

  TrackParametrizationWithError(const TrackParametrizationWithError& src) = default;
  TrackParametrizationWithError(TrackParametrizationWithError&& src) = default;
  TrackParametrizationWithError& operator=(const TrackParametrizationWithError& src) = default;
  TrackParametrizationWithError& operator=(TrackParametrizationWithError&& src) = default;
  ~TrackParametrizationWithError() = default;
  using TrackParametrization<value_T>::TrackParametrization;

  using TrackParametrization<value_T>::set;
  void set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge = 1, const PID pid = PID::Pion);
  void set(value_t x, value_t alpha, const value_t* par, const value_t* cov, int charge = 1, const PID pid = PID::Pion);
  void set(const dim3_t& xyz, const dim3_t& pxpypz, const std::array<value_t, kLabCovMatSize>& cv, int sign, bool sectorAlpha = true, const PID pid = PID::Pion);
  const covMat_t& getCov() const;
  value_t getSigmaY2() const;
  value_t getSigmaZY() const;
  value_t getSigmaZ2() const;
  value_t getSigmaSnpY() const;
  value_t getSigmaSnpZ() const;
  value_t getSigmaSnp2() const;
  value_t getSigmaTglY() const;
  value_t getSigmaTglZ() const;
  value_t getSigmaTglSnp() const;
  value_t getSigmaTgl2() const;
  value_t getSigma1PtY() const;
  value_t getSigma1PtZ() const;
  value_t getSigma1PtSnp() const;
  value_t getSigma1PtTgl() const;
  value_t getSigma1Pt2() const;
  value_t getCovarElem(int i, int j) const;
  value_t getDiagError2(int i) const;

  void print() const;
  std::string asString() const;


 protected:
  covMat_t mC{0.f}; // 15 covariance matrix elements

  ClassDefNV(TrackParametrizationWithError, 2);
};

//__________________________________________________________________________
template <typename value_T>
TrackParametrizationWithError<value_T>::TrackParametrizationWithError() : TrackParametrization<value_T>{}
{
}

//__________________________________________________________________________
template <typename value_T>
TrackParametrizationWithError<value_T>::TrackParametrizationWithError(value_t x, value_t alpha, const params_t& par,
                                                                              const covMat_t& cov, int charge, const PID pid)
  : TrackParametrization<value_T>{x, alpha, par, charge, pid}
{
  // explicit constructor
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
void TrackParametrizationWithError<value_T>::set(value_t x, value_t alpha, const params_t& par, const covMat_t& cov, int charge, const PID pid)
{
  set(x, alpha, par.data(), cov.data(), charge, pid);
}

//__________________________________________________________________________
template <typename value_T>
void TrackParametrizationWithError<value_T>::set(value_t x, value_t alpha, const value_t* par, const value_t* cov, int charge, const PID pid)
{
  TrackParametrization<value_T>::set(x, alpha, par, charge, pid);
  for (int i = 0; i < kCovMatSize; i++) {
    mC[i] = cov[i];
  }
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getCov() const -> const covMat_t&
{
  return mC;
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaY2() const -> value_t
{
  return mC[kSigY2];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaZY() const -> value_t
{
  return mC[kSigZY];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaZ2() const -> value_t
{
  return mC[kSigZ2];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaSnpY() const -> value_t
{
  return mC[kSigSnpY];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaSnpZ() const -> value_t
{
  return mC[kSigSnpZ];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaSnp2() const -> value_t
{
  return mC[kSigSnp2];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaTglY() const -> value_t
{
  return mC[kSigTglY];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaTglZ() const -> value_t
{
  return mC[kSigTglZ];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaTglSnp() const -> value_t
{
  return mC[kSigTglSnp];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigmaTgl2() const -> value_t
{
  return mC[kSigTgl2];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigma1PtY() const -> value_t
{
  return mC[kSigQ2PtY];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigma1PtZ() const -> value_t
{
  return mC[kSigQ2PtZ];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigma1PtSnp() const -> value_t
{
  return mC[kSigQ2PtSnp];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigma1PtTgl() const -> value_t
{
  return mC[kSigQ2PtTgl];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getSigma1Pt2() const -> value_t
{
  return mC[kSigQ2Pt2];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getCovarElem(int i, int j) const -> value_t
{
  return mC[CovarMap[i][j]];
}

//__________________________________________________________________________
template <typename value_T>
auto TrackParametrizationWithError<value_T>::getDiagError2(int i) const -> value_t
{
  return mC[DiagMap[i]];
}

} // namespace track
} // namespace o2
#endif /* INCLUDE_RECONSTRUCTIONDATAFORMATS_TRACKPARAMETRIZATIONWITHERROR_H_ */
