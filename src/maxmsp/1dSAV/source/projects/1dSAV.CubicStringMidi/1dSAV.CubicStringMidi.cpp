#include "StringProcessor.h"
#include "c74_min.h"
#include <atomic>
#include <memory>
using namespace c74::min;

// clang-format off

class CubicStringMidi
    : public object<CubicStringMidi>,
      public sample_operator<4, 3>
{
private:
  std::shared_ptr<StringProcessor<double>> processor, newProcessor;
  float sr{0};
  float pbend{0}, posex{0.9}, poslistL{0.3}, poslistR{0.3}, t60_0mod{4};
  std::atomic<bool> reinitFlag{true};

  // note trigger state
  std::atomic<bool> swapProc{false};
  std::atomic<double> velocity{0.0};

  // Excitation signal
  int excitationType{0}; // 0 for "struck" 1 for "pluck"
  float elapsedTimeImpulse{1};
  float inputForce{0};

public:
  MIN_DESCRIPTION{"String model with cubic nonlinearity, midi activated through `note' message"};
  MIN_TAGS{"audio"};
  MIN_AUTHOR{"Thomas Risse"};

  // no sample input inlet: note event is handled through message<>
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
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->lambda0 = args[0];
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> alpha{
    this,
    "stability condition setting",
    0.9,
    range{0.1, 1},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->alpha = args[0];
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> f0{
    this,
    "fundamental frequency",
    200,
    range{1, 10000},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->f0 = args[0];
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> beta{
    this,
    "beta",
    1e-4,
    range{1e-12, 1},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->beta = args[0];
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> impulseWidth{
    this,
    "impulse width (s)",
    1e-3,
    range{1e-5, 1e-1},
    setter{
      MIN_FUNCTION{
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> t60_0{
    this,
    "first decay time",
    4,
    range{1e-4, 100},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->t60_0 = args[0];
        newProcessor->t60_1 = newProcessor->t60_0 * brightness;
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> fd0{
    this,
    "first decay frequency",
    100,
    range{0, 1000},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->fd0 = args[0];
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> brightness{
    this,
    "brightness",
    0.8,
    range{1e-2, 1},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->t60_1 = newProcessor->t60_0 * brightness;
        return args;
      }
    }
  };

  attribute<number, threadsafe::no, limit::clamp> fd1{
    this,
    "brightness frequency",
    1000,
    range{100, 10000},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->fd1 = args[0];
        return args;
      }
    }
  };

  attribute<int, threadsafe::no, limit::clamp> nl_mode{
    this,
    "nonlinear mode",
    2,
    range{0, 4},
    setter{
      MIN_FUNCTION{
        if (reinitFlag.exchange(false)) 
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);
        newProcessor->nonlinear_mode = args[0];
        return args;
      }
    }
  };

  message<> note{
    this,
    "note",
    "Trigger a note: arg0 frequency [Hz], arg1 velocity [0..1]",
    MIN_FUNCTION{
      if (args.size() >= 2 && sr > 0){
        double velocity = std::max(0.0, std::min(1.0, (double)args[1]));
        this->velocity.store(velocity);

        if (reinitFlag.exchange(false))
          newProcessor = std::make_shared<StringProcessor<double>>(*processor);

        newProcessor->f0 = args[0];
        swapProc.store(newProcessor->reinitDsp(sr));
      }
      return args;
    }
  };

  message<> dspsetup{
    this,
    "dspsetup",
    MIN_FUNCTION{
      sr = args[0];
      processor->reinitDsp(sr);
      return {};
    }
  };

  message<> outstate{
    this,
    "state",
    MIN_FUNCTION{
      std::vector<float> state = processor->getState();
      atoms state_atoms;
      state_atoms.reserve(state.size());
      for (auto v : state)
        state_atoms.push_back(atom(v));
      outputData.send(state_atoms);
      return {};
    }
  };

  message<> number{
    this,
    "number",
    MIN_FUNCTION{
      if (inlet == 1)
        pbend = args[0];
      else if (inlet == 2)
        posex = args[0];
      else if (inlet == 3)
        poslistL = args[0];
      else if (inlet == 4)
        poslistR = args[0];
      else if (inlet == 5)
        t60_0mod = args[0];
      return {};
    }
  };

  CubicStringMidi(const atom &args = {})
  {
    processor = std::make_shared<StringProcessor<double>>(44100);
  }

  samples<3>
  operator()(sample posex,
            sample poslistL,
            sample poslistR,
            sample t60_0)
  {
    if (swapProc.exchange(false))
    {
      reinitFlag.store(true);
      processor = newProcessor;
      elapsedTimeImpulse = 0;
    }

    // Compute input
    if (excitationType == 0)
    {
      inputForce = velocity.load() * sin(M_PI * elapsedTimeImpulse / impulseWidth) * float(elapsedTimeImpulse < impulseWidth);
    }
    else
    {
      inputForce = velocity.load() / 2 *
                  ((1 - cos(M_PI * elapsedTimeImpulse / impulseWidth)) * (float(elapsedTimeImpulse < impulseWidth)) +
                    (1 + cos(M_PI * (elapsedTimeImpulse - impulseWidth) * 10 / impulseWidth)) * float((elapsedTimeImpulse > impulseWidth) and (elapsedTimeImpulse < impulseWidth * 1.1)));
    }
    elapsedTimeImpulse += float(1) / sr;

    if (posexInlet.has_signal_connection())
      this->posex = posex;
    if (poslistLInlet.has_signal_connection())
      this->poslistL = poslistL;
    if (poslistRInlet.has_signal_connection())
      this->poslistR = poslistR;
    if (t60_0Inlet.has_signal_connection())
      this->t60_0mod = t60_0;

    auto [outL, outR, epsilon] = processor->process(double(inputForce),
                                                    double(0.0), // no pbend dynamic in note API
                                                    double(this->posex),
                                                    double(this->poslistL),
                                                    double(this->poslistR),
                                                    double(this->t60_0mod));

    return {{outL, outR, epsilon}};
  };
};

MIN_EXTERNAL(CubicStringMidi);

// clang-format off