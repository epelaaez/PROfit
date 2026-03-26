#ifndef NUFASTLBL_H
#define NUFASTLBL_H

namespace NuFastLBL {

// Probability_Matter_LBL calculates all nine oscillation probabilities including
// the matter effect in an optimized, fast, and efficient way. The precision can
// be controlled with N_Newton. For many applications N_Newton=0 may be enough,
// but many years of DUNE or HK-LBL may require N_Newton=1. This code may be
// suitable for atmospheric neutrinos. The code is standalone.
//
// Inputs:
//   mixing angles (usual parameterization)
//   phase (usual parameterization) make Dmsq31 positive/negative for the NO/IO
//   Delta msq's (eV^2)
//   L (km)
//   E (GeV) positive for neutrinos, negative for antineutrinos
//   rho (g/cc)
//   Ye: electron fraction, typically around 0.5
//   N_Newton: number of Newton's method iterations to do. should be zero, one, two (or higher)
// Outputs:
//   probs_returned is all nine oscillation probabilities: e.g. probs_returned[1][0] is mu->e
void Probability_Matter_LBL(double s12sq, double s13sq, double s23sq, double delta, double Dmsq21, double Dmsq31, double L, double E, double rho, double Ye, int N_Newton, double (*probs_returned)[3][3]);

void Probability_Vacuum_LBL(double s12sq, double s13sq, double s23sq, double delta, double Dmsq21, double Dmsq31, double L, double E, double (*probs_returned)[3][3]);

}

#endif

