#ifndef CUBIC_STRING_PROCESSOR_H
#define CUBIC_STRING_PROCESSOR_H

#include <cmath>
#include <vector>
#include <tuple>

#include <Eigen/Dense>

#include "Bow.h"

template <class T>
class StringProcessor
{
private:
  bool controlTerm{true};
  // Physical parameters
  T eta_0{0}, eta_1{0}, rho{0}, mu{0}, E{0}, I{0}, R{0}, A{0}, T0{0}, l0{0};

  // nonlinear mode  0: linear, 1: KC, 2: cubic geom, 3: contact, 4: Geometric + Contact
  int nl_mode{2};

  // Contact parameter (only needed for nl == 3)
  T kc{1e9}, alphac{1.4};
  Eigen::Vector<T, -1> boundary;

  // Discretization parameters
  T dt{0}, sr{0}, h{0};
  int N;

  // Update coefficients
  Eigen::Vector<T, -1> Current0, D40, righthand, Rbow, term0V;
  T Current1, Current2, Last0, Last1, D41, D42, vrel, phinow, dxq2{0};

  // Nonlinear part
  Eigen::Vector<T, -1> g, dxq, dxq3, Vprime;
  T psi{0}, epsilon{0}, V{0};

  // State
  Eigen::Vector<T, -1> qlast, qnow, qnext;

  // Excitation and listening positions
  T posex{0}, poslistL{0}, poslistR{0};
  T vl{0}, vr{0};
  // Pitch bend
  T bend{0}, fbend{0};

  // Bow
  Bow bow{SMOOTHEDMATUSIAK};

public:
  StringProcessor(float sampleRate, bool controlTerm = true);

  void updateDerivedConstants();

  void setTandLFromf0Beta();

  void setDissFromDecays();

  void modifyhFromBend();

  T zeta(T omega, T gamma2, T kappa2);

  void setBoundary(std::vector<T> in);

  void reinitDsp(float sampleRate);

  void updateCoefficients();

  void computeVAndVprime();

  void computeV();

  std::vector<float> getState() const
  {
    std::vector<float> state(qnow.size());
    for (int i = 0; i < qnow.size(); ++i)
      state[i] = static_cast<float>(qnow[i]);
    return state;
  }

  std::tuple<T, T, T> process(T input, T bend = 0, T posex = 0.9, T poslistL = 0.3, T poslistR = 0.3, T t60_0 = 0);

  std::tuple<T, T, T> processBowed(T vbow, T Fbow, T bend = 0, T posex = 0.9, T poslistL = 0.3, T poslistR = 0.3);

  // Return physical parameters
  std::tuple<T, T, T, T, T, T, T, T, T, T> getPhysicalParameters()
  {
    return {eta_0, eta_1, rho, mu, E, I, R, A, T0, l0};
  }

  // Bow characteristic
  T phi(T vrel);

  void vout();

  // Higher level perceptive parameters
  T t60_0{0}, t60_1{0}, fd0{0}, fd1{0}, f0{0}, beta{0};
  // Control SAV parameter
  T lambda0{0};
  // Discretization parameters
  T alpha{0};

  int getN() { return N; };

  // Nonlinear mode
  int nonlinear_mode{2};
};

#endif
