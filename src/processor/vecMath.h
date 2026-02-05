#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif
// Use VDSP for vectorization ?
// Assert for sizes, works only in debug
// Passer VectorType
// faire avec des pointeurs et passer la taille, appeler avec A.data

namespace vecMath
{

  // void setValue(std::vector<double> &vec, double &value){
  //     vDSP_vfillD(&value, vec.data(), 1, vec.size());
  // };

  // void setValue(std::vector<float> &vec, float &value){
  //     vDSP_vfill(&value, vec.data(), 1, vec.size());
  // };

  // void Dmin(const std::vector<double> &invec, std::vector<double> &outvec,
  // const double &scaling){
  //     outvec[0] = invec[0];
  //     vDSP_vsubD(invec.data() + 1, 1, invec.data(), 1, outvec.data()+1, 1,
  //     invec.size()-1); outvec[invec.size()] = -invec[invec.size() - 1];
  //     vDSP_vsmulD(outvec.data(), 1, &scaling, outvec.data(), 1, invec.size());
  // };

  // void Dmin(const std::vector<float> &invec, std::vector<float> &outvec, const
  // float &scaling){
  //     outvec[0] = invec[0];
  //     vDSP_vsub(invec.data() + 1, 1, invec.data(), 1, outvec.data()+1, 1,
  //     invec.size()-1); outvec[invec.size()] = -invec[invec.size() - 1];
  //     vDSP_vsmul(outvec.data(), 1, &scaling, outvec.data(), 1, invec.size());
  // };

  // void Dplus(const std::vector<double> &invec, std::vector<double> &outvec,
  // const double &scaling){
  //     vDSP_vsubD(invec.data() + 1, 1, invec.data(), 1, outvec.data(), 1,
  //     invec.size()-1); vDSP_vsmulD(outvec.data(), 1, &scaling, outvec.data(),
  //     1, invec.size()-1);
  // };

  // void Dplus(const std::vector<float> &invec, std::vector<float> &outvec, const
  // float &scaling){
  //     vDSP_vsub(invec.data() + 1, 1, invec.data(), 1, outvec.data(), 1,
  //     invec.size()-1); vDSP_vsmul(outvec.data(), 1, &scaling, outvec.data(), 1,
  //     invec.size()-1);
  // };

  // void cube(const std::vector<double> &invec, std::vector<double> &outvec){
  //     vDSP_vmulD(invec.data(), 1, invec.data(), 1, outvec.data(), 1,
  //     invec.size()); vDSP_vmulD(outvec.data(), 1, invec.data(), 1,
  //     outvec.data(), 1, invec.size());
  // }

  // void cube(const std::vector<float> &invec, std::vector<float> &outvec){
  //     vDSP_vmul(invec.data(), 1, invec.data(), 1, outvec.data(), 1,
  //     invec.size()); vDSP_vmul(outvec.data(), 1, invec.data(), 1,
  //     outvec.data(), 1, invec.size());
  // }

  // void sdiv(const std::vector<double> &invec, std::vector<double> &outvec,
  // const double &denominator){
  //     vDSP_vsdivD(invec.data(), 1, &denominator, outvec.data(), 1,
  //     invec.size());
  // };

  // void sdiv(const std::vector<float> &invec, std::vector<float> &outvec, const
  // float &denominator){
  //     vDSP_vsdiv(invec.data(), 1, &denominator, outvec.data(), 1,
  //     invec.size());
  // };

  // void add(const std::vector<double> &invec1, const std::vector<double>
  // &invec2, std::vector<double> &outvec, const double &scaling){
  //     vDSP_vaddD(invec1.data(), 1, invec2.data(), 1, outvec.data(), 1,
  //     invec1.size()); vDSP_vsmulD(outvec.data(), 1, &scaling, outvec.data(), 1,
  //     invec1.size());
  // }

  // void add(const std::vector<float> &invec1, const std::vector<float> &invec2,
  // std::vector<float> &outvec, const float &scaling){
  //     vDSP_vadd(invec1.data(), 1, invec2.data(), 1, outvec.data(), 1,
  //     invec1.size()); vDSP_vsmul(outvec.data(), 1, &scaling, outvec.data(), 1,
  //     invec1.size());
  // }

  // void substract(const std::vector<double> &invec1, const std::vector<double>
  // &invec2, std::vector<double> &outvec, const double &scaling){
  //     vDSP_vsubD(invec1.data(), 1, invec2.data(), 1, outvec.data(), 1,
  //     invec1.size()); vDSP_vsmulD(outvec.data(), 1, &scaling, outvec.data(), 1,
  //     invec1.size());
  // }

  // void substract(const std::vector<float> &invec1, const std::vector<float>
  // &invec2, std::vector<float> &outvec, const float &scaling){
  //     vDSP_vsub(invec1.data(), 1, invec2.data(), 1, outvec.data(), 1,
  //     invec1.size()); vDSP_vsmul(outvec.data(), 1, &scaling, outvec.data(), 1,
  //     invec1.size());
  // }

  // void addmul(const std::vector<double> &invec1, const double &scaling, const
  // std::vector<double> &invec2, std::vector<double> &outvec) {
  //     vDSP_vsmaD(invec1.data(), 1, &scaling, invec2.data(), 1, outvec.data(),
  //     1, invec1.size());
  // }

  // void addmul(const std::vector<float> &invec1, const float &scaling, const
  // std::vector<float> &invec2, std::vector<float> &outvec) {
  //     vDSP_vsma(invec1.data(), 1, &scaling, invec2.data(), 1, outvec.data(), 1,
  //     invec1.size());
  // }

  // void norm1(const std::vector<double> &invec, double &out){
  //     vDSP_svemgD(invec.data(), 1, &out, invec.size());
  // }

  // void norm1(const std::vector<float> &invec, float &out){
  //     vDSP_svemg(invec.data(), 1, &out, invec.size());
  // }

  // void dot(const std::vector<double> &invec1, const std::vector<double>
  // &invec2, double &out){
  //     vDSP_dotprD(invec1.data(), 1, invec2.data(), 1, &out, invec1.size());
  // }

  // void dot(const std::vector<float> &invec1, const std::vector<float> &invec2,
  // float &out){
  //     vDSP_dotpr(invec1.data(), 1, invec2.data(), 1, &out, invec1.size());
  // }

  // void sgn(const std::vector<double> &invec, std::vector<double> &outvec){
  //     vDSP_vabsD(invec.data(), 1, outvec.data(), 1, invec.size());
  // }

  // void sgn(const std::vector<float> &invec, std::vector<float> &outvec){
  //     vDSP_vabs(invec.data(), 1, outvec.data(), 1, invec.size());
  // }

  template <typename T, typename T2>
  void setValue(std::vector<T> &vec, T2 value)
  {
    std::fill(vec.begin(), vec.end(), value);
  };

  template <typename T, typename T2>
  void Dmin(const std::vector<T> &invec, std::vector<T> &outvec, T2 scaling)
  {
    outvec[0] = invec[0] * scaling;
    for (std::size_t i = 1; i < invec.size(); i++)
    {
      outvec[i] = (invec[i] - invec[i - 1]) * scaling;
    }
    outvec[invec.size()] = -invec[invec.size() - 1] * scaling;
  };

  template <typename T, typename T2>
  void Dplus(const std::vector<T> &invec, std::vector<T> &outvec, T2 scaling)
  {
    for (std::size_t i = 0; i < invec.size() - 1; i++)
    {
      outvec[i] = (invec[i + 1] - invec[i]) * scaling;
    }
  };

  template <typename T>
  void cube(const std::vector<T> &invec, std::vector<T> &outvec)
  {
    for (std::size_t i = 0; i < invec.size(); i++)
    {
      outvec[i] = invec[i] * invec[i] * invec[i];
    }
  }

  template <typename T, typename T2>
  void sdiv(const std::vector<T> &invec, std::vector<T> &outvec, T2 denominator)
  {
    for (std::size_t i = 0; i < invec.size(); i++)
    {
      outvec[i] = invec[i] / denominator;
    }
  };

  template <typename T, typename T2>
  void add(const std::vector<T> &invec1, const std::vector<T> &invec2,
           std::vector<T> &outvec, T2 scaling)
  {
    for (std::size_t i = 0; i < invec1.size(); i++)
    {
      outvec[i] = scaling * (invec1[i] + invec2[i]);
    }
  }

  template <typename T, typename T2>
  void substract(const std::vector<T> &invec1, const std::vector<T> &invec2,
                 std::vector<T> &outvec, T2 scaling)
  {
    for (std::size_t i = 0; i < invec1.size(); i++)
    {
      outvec[i] = scaling * (invec1[i] - invec2[i]);
    }
  }

  template <typename T, typename T2>
  void norm1(const std::vector<T> &invec, T2 &out)
  {
    out = 0;
    for (std::size_t i = 0; i < invec.size(); i++)
    {
      out += abs(invec[i]);
    }
  }

  template <typename T, typename T2>
  void dot(const std::vector<T> &invec1, const std::vector<T> &invec2, T2 &out)
  {
    out = 0;
    for (std::size_t i = 0; i < invec1.size(); i++)
    {
      out += invec1[i] * invec2[i];
    }
  }

  template <typename T>
  int sgn(T val) { return (T(0) < val) - (val < T(0)); }

} // namespace vecMath