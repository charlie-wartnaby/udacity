////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : PID Control project
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////


#ifndef PID_H
#define PID_H

using namespace std::chrono;

class PID {
public:
  /*
  * Errors
  * CW: these provided names don't make sense. We will have P, I and D
  *     terms, but they all use the same error as input.
  */
  double p_error;
  double i_error;
  double d_error;

  /*
  * Coefficients
  */ 
  double Kp;
  double Ki;
  double Kd;

  // CW: added member variables for computation
  // of D and I terms:
  double prev_error;     // for D term
  double integral_error; // for I term
  bool   first_exec;
  steady_clock::time_point prev_time;

  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();
};

#endif /* PID_H */
