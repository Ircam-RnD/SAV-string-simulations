#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "CubicStringProcessor.h"

using namespace std::chrono;

int main(int argc, char const *argv[])
{
  // Result file creation
  std::ofstream resultFile;
  resultFile.open("../timeCubicString.txt");

#ifdef EIGENPROC
  resultFile << "Eigen Processor" << std::endl;
  std::cout << "Eigen Processor" << std::endl;
#else
  resultFile << "Std Processor" << std::endl;
  std::cout << "Std Processor" << std::endl;
#endif

  // Samplerate
  float sr = 44100;
  float simDuration = 20;
  // Parameters
  float alpha = 0.9;
  float beta = 1e-4;
  float t60_0 = 12;
  float t60_1 = 6;
  std::array<float, 5> f0s{20, 50, 100, 200, 500};

  CubicStringProcessor<double> csp(sr);
  csp.alpha = alpha;
  csp.beta = beta;
  csp.t60_0 = t60_0;
  csp.t60_1 = t60_1;
  csp.reinitDsp(sr);

  // Without control term
  CubicStringProcessor<double> cspVanilla(sr, false);
  cspVanilla.alpha = alpha;
  cspVanilla.beta = beta;
  cspVanilla.t60_0 = t60_0;
  cspVanilla.t60_1 = t60_1;
  cspVanilla.reinitDsp(sr);

  CubicStringProcessor<float> cspFloat(sr);
  cspFloat.alpha = alpha;
  cspFloat.beta = beta;
  cspFloat.t60_0 = t60_0;
  cspFloat.t60_1 = t60_1;
  cspFloat.reinitDsp(sr);

  int width = 14;
  resultFile << std::setw(6) << "f0" << std::setw(6) << "N"
             << std::setw(width + 1) << "Double" << std::setw(width + 1)
             << "Double Vanilla" << std::setw(width + 1) << "Float"
             << std::endl;

  auto start = high_resolution_clock::now();
  auto stop = high_resolution_clock::now();
  float rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6;

  std::cout.precision(2);
  resultFile.precision(2);
  for (int i = 0; i < f0s.size(); i++)
  {
    csp.f0 = f0s[i];
    csp.reinitDsp(sr);
    cspVanilla.f0 = f0s[i];
    cspVanilla.reinitDsp(sr);
    cspFloat.f0 = f0s[i];
    cspFloat.reinitDsp(sr);
    std::cout << "f0 = " << f0s[i] << ", N = " << csp.getN() << std::endl;
    resultFile << std::setw(6) << f0s[i] << std::setw(6) << csp.getN();

    // Time CubicStringProcessor
    start = high_resolution_clock::now();
    for (int i = 0; i < sr * simDuration; i++)
    {
      csp.process(0.1, 0, 0.9, 0.3, 0.3);
    }
    stop = high_resolution_clock::now();
    rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6 /
              simDuration;

    std::cout << "Time CubicStringProcessor: " << rtRatio * 100 << "%"
              << std::endl;
    resultFile << std::setw(width) << rtRatio * 100 << "%";

    // Time CubicStringProcessor Vanilla
    start = high_resolution_clock::now();
    for (int i = 0; i < sr * simDuration; i++)
    {
      cspVanilla.process(0.1, 0, 0.9, 0.3, 0.3);
    }
    stop = high_resolution_clock::now();
    rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6 /
              simDuration;

    std::cout << "Time CubicStringProcessor: " << rtRatio * 100 << "%"
              << std::endl;
    resultFile << std::setw(width) << rtRatio * 100 << "%";

    // Time CubicStringProcessor float
    start = high_resolution_clock::now();
    for (int i = 0; i < sr * simDuration; i++)
    {
      cspFloat.process(0.1, 0, 0.9, 0.3, 0.3);
    }
    stop = high_resolution_clock::now();
    rtRatio = (duration_cast<microseconds>(stop - start)).count() * 1e-6 /
              simDuration;

    std::cout << "Time CubicStringProcessor Float: " << rtRatio * 100 << "%"
              << std::endl;
    resultFile << std::setw(width) << rtRatio * 100 << "%" << std::endl;
  }

  return 0;
}
