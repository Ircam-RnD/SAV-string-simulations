#include "StringProcessor.h"
#include "vecMath.h"
#include "iostream"

template <class T>
StringProcessor<T>::StringProcessor(float sampleRate, bool controlTerm)
{
  this->controlTerm = controlTerm;
  // Default physical constants
  l0 = 1;
  T0 = 40;
  rho = 8000;
  E = 2e11;
  R = 2.9e-4;
  eta_0 = 0.5;
  eta_1 = 0;

  // Derived physical constants
  updateDerivedConstants();

  // Higher level parameters
  t60_0 = 12;
  t60_1 = 6;
  fd0 = 0;
  fd1 = 1000;
  setDissFromDecays();

  f0 = 100;
  beta = 1e-4;
  setTandLFromf0Beta();

  // Sav control setting
  alpha = 0.9;
  lambda0 = 100;

  // Excitation/Listening position
  posex = 0.72;
  poslistL = 0.3;
  poslistR = 0.3;

  // State variables
  reinitDsp(sampleRate);
};

template <class T>
void StringProcessor<T>::updateDerivedConstants()
{
  A = M_PI * R * R;
  I = M_PI * R * R * R * R / 4;
  mu = rho * A;
}

template <class T>
void StringProcessor<T>::setTandLFromf0Beta()
{
  T0 = sqrt((pow(M_PI, 4) * E * pow(f0, 2) * pow(R, 6) * rho) / (beta * (1 + beta)));
  l0 = sqrt((pow(M_PI, 3) * E * pow(R, 4)) / (4 * beta * T0));
}

template <class T>
void StringProcessor<T>::setDissFromDecays()
{
  double gamma2 = T0 / mu;
  double kappa2 = E * I / mu;
  double beta2_0 = zeta(2 * M_PI * fd0, gamma2, kappa2);
  double beta2_1 = zeta(2 * M_PI * fd1, gamma2, kappa2);

  eta_0 = fmax(3 * log(10) / (beta2_1 - beta2_0) * (beta2_1 / t60_0 - beta2_0 / t60_1), 0);
  eta_1 = fmax(3 * log(10) / (beta2_1 - beta2_0) * (-1 / t60_0 + 1 / t60_1), 0);
}

template <class T>
T StringProcessor<T>::zeta(T omega, T gamma2, T kappa2)
{
  return (-gamma2 + sqrt(gamma2 * gamma2 + 4 * kappa2 * omega * omega)) / (2 * kappa2);
}

template <class T>
void StringProcessor<T>::modifyhFromBend()
{
  // Only the length is modified for simplicity.
  // Inharmonicity is not taken into account.
  fbend = f0 * pow(2, bend / 1200);
  // T lbend = 1/(2*fbend) * sqrt(T0/mu) * sqrt(1 + pow(M_PI, 3) * E * pow(R, 4) / (4 * l0 * l0));
  T a = fbend * fbend;
  T b = -T0 / (4 * mu);
  T c = -pow(M_PI, 2) * E * I / mu;
  T delta = pow(b, 2) - 4 * a * c;
  T lbend = sqrt((-b + sqrt(delta)) / (2 * a));
  T gamma = dt * dt * T0 + 4 * dt * mu * eta_1;
  h = fmax(lbend / N, sqrt(
                          (gamma + sqrt(gamma * gamma + 16 * mu * E * I * pow(dt, 2))) / (2 * mu)));
}

template <class T>
void StringProcessor<T>::setBoundary(std::vector<T> in)
{
  if (in.size() == N - 1)
  {
    boundary = Eigen::Map<Eigen::VectorX<T>>(in.data(), in.size());
  }
}

template <class T>
void StringProcessor<T>::reinitDsp(float sampleRate)
{
  fbend = f0;
  // Recompute physical coefficients from high level parameters
  setDissFromDecays();
  setTandLFromf0Beta();

  // // To test with specific physical parameters (Overrides lines before)
  // R = 0.4e-3;
  // rho = 2.34e-3 / (M_PI * R * R);
  // T0 = 39.15;
  // eta_0 = 3.12;
  // eta_1 = 0;
  // l0 = 0.33;
  // E = 4e9;

  updateDerivedConstants();

  modifyhFromBend();

  // Copying the nonlinear mode in the private (effective) variable
  nl_mode = nonlinear_mode;

  // Stability condition
  sr = sampleRate;
  dt = 1 / sr;
  T gamma = dt * dt * T0 + 4 * dt * mu * eta_1;
  h = sqrt(
      (gamma + sqrt(gamma * gamma + 16 * mu * E * I * pow(dt, 2))) / (2 * mu));
  N = static_cast<int>(floor(fmax(3, fmin(1000, alpha * floor(l0 / h)))));
  h = l0 / N;

  updateCoefficients();

  // Initialisation of vectors to right size
  qlast = Eigen::Vector<T, -1>::Zero(N - 1);
  qnow = Eigen::Vector<T, -1>::Zero(N - 1);
  qnext = Eigen::Vector<T, -1>::Zero(N - 1);
  g = Eigen::Vector<T, -1>::Zero(N - 1);

  dxq = Eigen::Vector<T, -1>::Zero(N);
  dxq3 = Eigen::Vector<T, -1>::Zero(N);

  Vprime = Eigen::Vector<T, -1>::Zero(N - 1);

  righthand = Eigen::Vector<T, -1>::Zero(N - 1);

  Rbow = Eigen::Vector<T, -1>::Zero(N - 1);

  boundary = -Eigen::Vector<T, -1>::Ones(N - 1) * 1e-3;

  psi = 0;
}

template <class T>
void StringProcessor<T>::updateCoefficients()
{
  D40 = Eigen::Vector<T, -1>::Ones(N - 1) * 6 / pow(h, 4);
  D40[0] = 5 / pow(h, 4);
  D40[N - 2] = 5 / pow(h, 4);
  D41 = -4 / (pow(h, 4));
  D42 = 1 / (pow(h, 4));
  Current0 = (2 * mu - 2 * pow(dt / h, 2) * T0 - 4 * mu * eta_1 * dt / pow(h, 2)) * Eigen::Vector<T, -1>::Ones(N - 1) - pow(dt, 2) * E * I * D40;
  Last0 = -(1 - dt * eta_0) * mu + 4 * mu * dt * eta_1 / pow(h, 2);
  Current1 = pow(dt / h, 2) * T0 - pow(dt, 2) * E * I * D41 + 2 * mu * eta_1 * dt / pow(h, 2);
  Current2 = -pow(dt, 2) * E * I * D42;
  Last1 = -2 * mu * eta_1 * dt / pow(h, 2);
}

template <class T>
void StringProcessor<T>::computeVAndVprime()
{
  switch (nl_mode)
  {
  case 0:
    V = 0;
    Vprime.setZero();
    break;
  case 1:
    dxq.setZero();
    dxq.head(N - 1) = qnow;
    dxq.tail(N - 1) -= qnow;
    dxq /= h;
    dxq2 = dxq.dot(dxq);

    V = E * A / (8 * l0) * h * h * pow(dxq2, 2);
    Vprime = -E * A / (2 * l0) * h * dxq2 * (dxq.tail(N - 1) - dxq.head(N - 1));
    break;
  case 2:
    dxq.setZero();
    dxq.head(N - 1) = qnow;
    dxq.tail(N - 1) -= qnow;
    dxq /= h;
    dxq3 = dxq.array().cube();

    Vprime = -(E * A - T0) / 2 * (dxq3.tail(N - 1) - dxq3.head(N - 1));
    V = (E * A - T0) / 8 * h * (dxq3.cwiseProduct(dxq)).sum();
    break;
  case 3:
    V = ((-qnow + boundary).cwiseMax(0).array().pow(alphac + 1)).sum() * kc / (alphac + 1) * h;
    Vprime = -((-qnow + boundary).cwiseMax(0).array().pow(alphac)).matrix() * kc * h;
    break;
  case 4:
    // Geometric + Contact
    dxq.setZero();
    dxq.head(N - 1) = qnow;
    dxq.tail(N - 1) -= qnow;
    dxq /= h;
    dxq3 = dxq.array().cube();

    V = (E * A - T0) / 8 * h * (dxq3.cwiseProduct(dxq)).sum() + ((-qnow + boundary).cwiseMax(0).array().pow(alphac + 1)).sum() * kc / (alphac + 1) * h;
    Vprime = -(E * A - T0) / 2 * (dxq3.tail(N - 1) - dxq3.head(N - 1)) - ((-qnow + boundary).cwiseMax(0).array().pow(alphac)).matrix() * kc * h;
    break;
  default:
    V = 0;
    Vprime.setZero();
    break;
  }
}

template <class T>
void StringProcessor<T>::computeV()
{
  switch (nl_mode)
  {
  case 0:
    V = 0;
    break;
  case 1:
    dxq.setZero();
    dxq.head(N - 1) = (qnow + qlast) / 2;
    dxq.tail(N - 1) -= (qnow + qlast) / 2;
    dxq /= h;

    V = E * A / (8 * l0) * h * h * pow(dxq.dot(dxq), 2);
    break;
  case 2:
    dxq.setZero();
    dxq.head(N - 1) = (qnow + qlast) / 2;
    dxq.tail(N - 1) -= (qnow + qlast) / 2;
    dxq /= h;
    dxq3 = dxq.array().cube();

    V = (E * A - T0) / 8 * h * (dxq3.cwiseProduct(dxq)).sum();
    break;
  case 3:
    V = ((-(qnow + qlast) / 2 + boundary).cwiseMax(0).array().pow(alphac + 1)).sum() * kc / (alphac + 1) * h;
    break;
  case 4:
    dxq.setZero();
    dxq.head(N - 1) = (qnow + qlast) / 2;
    dxq.tail(N - 1) -= (qnow + qlast) / 2;
    dxq /= h;
    dxq3 = dxq.array().cube();

    V = (E * A - T0) / 8 * h * (dxq3.cwiseProduct(dxq)).sum() + ((-(qnow + qlast) / 2 + boundary).cwiseMax(0).array().pow(alphac + 1)).sum() * kc / (alphac + 1) * h;
    break;
  default:
    V = 0;
    break;
  }
}

template <class T>
std::tuple<T, T, T> StringProcessor<T>::process(T input, T bend, T posex, T poslistL, T poslistR, T t60_0)
{
  // Eigen::internal::set_is_malloc_allowed(false);
  //  Pitch bend
  if (bend != this->bend)
  {
    this->bend = bend;
    modifyhFromBend();
    updateCoefficients();
  }
  // Excitation and listening positions
  if (posex != this->posex || poslistL != this->poslistL || poslistR != this->poslistR)
  {
    this->posex = std::clamp(posex, T(0), T(1));
    this->poslistL = std::clamp(poslistL, T(0), T(1));
    this->poslistR = std::clamp(poslistR, T(0), T(1));
  }
  // Modify damping from t60_0. Only eta_0 is modified to preserve stability.
  // F60_0 is considered to be zero here.
  if (t60_0 != this->t60_0 && t60_0 > 1e-4)
  {
    this->t60_0 = t60_0;
    this->eta_0 = 3 * log(10) / (t60_0);

    updateCoefficients();
  }

  // Compute g (dependant on the nonlinear mode => value of nl)
  computeVAndVprime();
  g = Vprime / (sqrt(2 * V) + 1e-12);

  if (controlTerm)
  {
    computeV();
    epsilon = psi - sqrt(2 * V);
    g += -lambda0 * epsilon * dt * ((qnow - qlast).array() > 0).select(Eigen::Vector<T, -1>::Ones(N - 1), -Eigen::Vector<T, -1>::Ones(N - 1)) / ((qnow - qlast).template lpNorm<1>() + 1e-12);
  }

  // Linear part
  righthand = Current0.cwiseProduct(qnow) + Last0 * qlast;
  righthand.head(N - 2) += Current1 * qnow.tail(N - 2) + Last1 * qlast.tail(N - 2);
  righthand.tail(N - 2) += Current1 * qnow.head(N - 2) + Last1 * qlast.head(N - 2);

  righthand.tail(N - 3) += Current2 * qnow.head(N - 3);
  righthand.head(N - 3) += Current2 * qnow.tail(N - 3);

  // External force
  righthand(static_cast<int>(floor(this->posex * (N - 2)))) += pow(dt, 2) * input / h * (1 - (this->posex * (N - 2) - floor(this->posex * (N - 2))));
  righthand(static_cast<int>(ceil(this->posex * (N - 2)))) += pow(dt, 2) * input / h * ((this->posex * (N - 2) - floor(this->posex * (N - 2))));

  // Nonlinear part
  righthand += pow(dt / 2, 2) * 1 / h * g * g.dot(qlast) - pow(dt, 2) * 1 / h * g * psi;

  // Solving using shermann morrisson
  double term0 = 1 / (mu * (1 + dt * eta_0));
  qnext = term0 * righthand - pow(dt / 2, 2) * pow(term0, 2) * 1 / h * g * g.dot(righthand) / (1 + term0 * pow(dt / 2, 2) * 1 / h * g.dot(g));

  psi = psi + 0.5 * g.dot(qnext - qlast);

  qlast = qnow;
  qnow = qnext;
  // Eigen::internal::set_is_malloc_allowed(true);
  vout();
  return {vl, vr, epsilon / (sqrt(2 * V) + 1e-12)};
}

template <class T>
std::tuple<T, T, T> StringProcessor<T>::processBowed(T vbow, T Fbow, T bend, T posex, T poslistL, T poslistR)
{

  // Compute relative bow velocity
  vrel = (qnow(static_cast<int>(posex * (N - 2))) - qlast(static_cast<int>(posex * (N - 2)))) / dt - vbow;
  // Compute bow force
  phinow = bow.phi(vrel);
  Rbow.setZero();
  Rbow(static_cast<int>(posex * (N - 2))) = abs(Fbow) * phinow / (vrel + std::copysign(1e-12, vrel));

  // Eigen::internal::set_is_malloc_allowed(false);
  //  Pitch bend
  if (bend != this->bend)
  {
    this->bend = bend;
    modifyhFromBend();
    updateCoefficients();
  }
  // Excitation and listening positions
  if (posex != this->posex || poslistL != this->poslistL || poslistR != this->poslistR)
  {
    this->posex = std::clamp(posex, T(0), T(1));
    this->poslistL = std::clamp(poslistL, T(0), T(1));
    this->poslistR = std::clamp(poslistR, T(0), T(1));
  }

  // Compute g (dependant on the nonlinear mode => value of nl)
  computeVAndVprime();
  g = Vprime / (sqrt(2 * V) + 1e-12);

  if (controlTerm)
  {
    computeV();
    epsilon = psi - sqrt(2 * V);
    g += -lambda0 * epsilon * dt * ((qnow - qlast).array() > 0).select(Eigen::Vector<T, -1>::Ones(N - 1), -Eigen::Vector<T, -1>::Ones(N - 1)) / ((qnow - qlast).template lpNorm<1>() + 1e-12);
  }
  // Linear part
  righthand = Current0.cwiseProduct(qnow) + (Last0 * Eigen::Vector<T, -1>::Ones(N - 1) + dt * Rbow / h * 0.5).cwiseProduct(qlast) + dt * dt / h * Rbow * vbow;
  righthand.head(N - 2) += Current1 * qnow.tail(N - 2) + Last1 * qlast.tail(N - 2);
  righthand.tail(N - 2) += Current1 * qnow.head(N - 2) + Last1 * qlast.head(N - 2);

  righthand.tail(N - 3) += Current2 * qnow.head(N - 3);
  righthand.head(N - 3) += Current2 * qnow.tail(N - 3);

  // Nonlinear part
  righthand += pow(dt / 2, 2) * 1 / h * g * g.dot(qlast) - pow(dt, 2) * 1 / h * g * psi;

  // Solving using shermann morrisson
  term0V = (mu * (1 + dt * eta_0) * Eigen::Vector<T, -1>::Ones(N - 1) + dt * Rbow / h * 0.5).cwiseInverse();
  qnext = term0V.cwiseProduct(righthand) - pow(dt / 2, 2) * 1 / h * term0V.cwiseProduct(term0V).cwiseProduct(g) * g.dot(righthand) / (1 + pow(dt / 2, 2) * 1 / h * term0V.cwiseProduct(g).dot(g));

  psi = psi + 0.5 * g.dot(qnext - qlast);

  qlast = qnow;
  qnow = qnext;
  // Eigen::internal::set_is_malloc_allowed(true);
  vout();
  return {vl, vr, epsilon / (sqrt(2 * V) + 1e-12)};
}

template <class T>
void StringProcessor<T>::vout()
{
  vl = ((qnow(static_cast<int>(floor(poslistL * (N - 2)))) - qlast(static_cast<int>(floor(poslistL * (N - 2))))) * (1 - (poslistL * (N - 2) - floor(poslistL * (N - 2)))) + (qnow(static_cast<int>(ceil(poslistL * (N - 2)))) - qlast(static_cast<int>(ceil(poslistL * (N - 2))))) * (poslistL * (N - 2) - floor(poslistL * (N - 2)))) / (dt);
  vr = ((qnow(static_cast<int>(floor(poslistR * (N - 2)))) - qlast(static_cast<int>(floor(poslistR * (N - 2))))) * (1 - (poslistR * (N - 2) - floor(poslistR * (N - 2)))) + (qnow(static_cast<int>(ceil(poslistR * (N - 2)))) - qlast(static_cast<int>(ceil(poslistR * (N - 2))))) * (poslistR * (N - 2) - floor(poslistR * (N - 2)))) / (dt);
}

template class StringProcessor<double>;
template class StringProcessor<float>;
