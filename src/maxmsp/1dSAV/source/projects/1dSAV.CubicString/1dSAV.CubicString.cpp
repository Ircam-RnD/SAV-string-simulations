#include "StringProcessor.h"
#include "c74_min.h"
#include <atomic>
#include <memory>
using namespace c74::min;

class CubicString
    : public object<CubicString>,
      public sample_operator<6, 3>
{
private:
  std::shared_ptr<StringProcessor<double>> processor;
  bool pIntialised{false};
  std::shared_ptr<StringProcessor<double>> newProcessor;
  float sr{0};
  float pbend{0}, posex{0.9}, poslistL{0.3}, poslistR{0.3},
      t60_0mod{4};
  std::atomic<bool> reinitFlag{false};

public:
  MIN_DESCRIPTION{"String model with cubic nonlinearity"};
  MIN_TAGS{"audio"};
  MIN_AUTHOR{"Thomas Risse"};

  inlet<> input{this, "(signal) excitation"};
  inlet<> pbendInlet{this, "(signal) pitch bend"};
  inlet<> posexInlet{this, "(signal) excitation position"};
  inlet<> poslistLInlet{this, "(signal) left listening position"};
  inlet<> poslistRInlet{this, "(signal) right listening position"};
  inlet<> t60_0Inlet{this, "(signal) decay time at 0 Hz"};
  outlet<> outputL{this, "(signal) left output", "signal"};
  outlet<> outputR{this, "(signal) right output", "signal"};
  outlet<> outputEps{this, "(signal) epsilon", "signal"};
  outlet<> outputData{this, "(list) Data"};

  attribute<number, threadsafe::no, limit::clamp> lambda0{
      this,
      "regularisation parameter",
      100,
      range{0, 1000000},
      setter{MIN_FUNCTION{if (pIntialised){processor->lambda0 = args[0];
} return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> alpha{
    this,
    "stability condition setting",
    0.9,
    range{0.1, 1},
    setter{MIN_FUNCTION{if (pIntialised){processor->alpha = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> f0{
    this,
    "fundamental frequency",
    200,
    range{1, 10000},
    setter{MIN_FUNCTION{if (pIntialised){processor->f0 = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> beta{
    this,
    "beta",
    1e-4,
    range{1e-12, 1},
    setter{MIN_FUNCTION{if (pIntialised){processor->beta = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> t60_0{
    this,
    "first decay time",
    4,
    range{1e-4, 100},
    setter{MIN_FUNCTION{if (pIntialised){processor->t60_0 = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> fd0{
    this,
    "first decay frequency",
    100,
    range{0, 1000},
    setter{MIN_FUNCTION{if (pIntialised){processor->fd0 = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> t60_1{
    this,
    "second decay time",
    2,
    range{1e-4, 100},
    setter{MIN_FUNCTION{if (pIntialised){processor->t60_1 = args[0];
}
return args;
}
}
}
;

attribute<number, threadsafe::no, limit::clamp> fd1{
    this,
    "second decay frequency",
    1000,
    range{100, 10000},
    setter{MIN_FUNCTION{if (pIntialised){processor->fd1 = args[0];
}
return args;
}
}
}
;

attribute<int, threadsafe::no, limit::clamp> nl_mode{
    this,
    "nonlinear mode",
    2,
    range{0, 4},
    setter{MIN_FUNCTION{if (pIntialised){processor->nonlinear_mode = args[0];
}
return args;
}
}
}
;

// Recompute updated coefficients
message<> bang{
    this,
    "bang",
    "Re-update with new physical coefficients.",
    MIN_FUNCTION{if (sr != 0){if (!reinitFlag.load()){
        newProcessor = std::make_shared<StringProcessor<double>>(*processor);
newProcessor->reinitDsp(sr);
reinitFlag.store(true);
cout << "N = " << newProcessor->getN() << endl;
}
}
return args;
}
}
;

message<> dspsetup{this,
                   "dspsetup",
                   MIN_FUNCTION{// number samplerate = args[0];
                                // int vectorsize = args[1];
                                sr = args[0];
cout << "sr =" << sr << endl;
processor->reinitDsp(sr);
cout << "N = " << processor->getN() << endl;
return {};
}
}
;

message<> outstate{
    this,
    "state",
    MIN_FUNCTION{// number samplerate = args[0];
                 // int vectorsize = args[1];
                 std::vector<float> state = processor->getState();
atoms state_atoms;
state_atoms.reserve(state.size());
for (auto v : state)
  state_atoms.push_back(atom(v));
outputData.send(state_atoms);
return {};
}
}
;

message<> number{this,
                 "number",
                 MIN_FUNCTION{if (inlet == 1){pbend = args[0];
}
else if (inlet == 2)
{
  posex = args[0];
}
else if (inlet == 3)
{
  poslistL = args[0];
}
else if (inlet == 4)
{
  poslistR = args[0];
}
else if (inlet == 5)
{
  t60_0mod = args[0];
}
return {};
}
}
;

CubicString(const atom &args = {})
{
  processor = std::make_shared<StringProcessor<double>>(44100);
  pIntialised = true;
}

samples<3>
operator()(sample input,
           sample pbend,
           sample posex,
           sample poslistL,
           sample poslistR,
           sample t60_0)
{
  if (pbendInlet.has_signal_connection())
  {
    this->pbend = pbend;
  }
  if (posexInlet.has_signal_connection())
  {
    this->posex = posex;
  }
  if (poslistLInlet.has_signal_connection())
  {
    this->poslistL = poslistL;
  }
  if (poslistRInlet.has_signal_connection())
  {
    this->poslistR = poslistR;
  }
  if (t60_0Inlet.has_signal_connection())
  {
    this->t60_0mod = t60_0;
  }

  if (reinitFlag.load())
  {
    processor = newProcessor;
    reinitFlag.store(false);
  }
  auto [outL, outR, epsilon] = processor->process(double(input),
                                                  double(this->pbend),
                                                  double(this->posex),
                                                  double(this->poslistL),
                                                  double(this->poslistR),
                                                  double(this->t60_0mod));
  return {{outL, outR, epsilon}};
}
}
;

MIN_EXTERNAL(CubicString);
