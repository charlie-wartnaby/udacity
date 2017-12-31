////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : PID Control project
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cstdio>
#include <iostream>

#include "PID.h"



using namespace std;
using namespace std::chrono;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  // Store provided gains for run-time use
  this->Kp = Kp;
  this->Ki = Ki;
  this->Kd = Kd;

  // Initiliase member variables
  prev_error = 0.0;
  integral_error = 0.0;
  first_exec = true;
  prev_time = steady_clock::now();
}

void PID::UpdateError(double cte) {

  // CW: as noted in the header file, I don't like the provided names
  // p_error, i_error etc, which are intended for the different *terms*
  // in the PID equation -- but those are all based on the same *error*!
  // I've kept the provided names to avoid marking confusion, however.

  // I first implemented this controller effectively assuming delta t=1
  // per iteration, as this was not supplied by the telemetry from the
  // simulator (or at least not passed to this class/method). But on
  // rerunning my controller on a later occasion with the same gains, the
  // behaviour was noticeably worse, and gave a much larger average square
  // CTE. Guessing this might be because the simulator is effectively working
  // in real time but the execution of this controller program is
  // asynchronous to that, I updated it to use actual elapsed time
  // on the PC to get delta time between iteraions, in the hope of removing
  // any inadvertent dependency it had on exactly how the simulator and
  // control program were running on the PC.
  steady_clock::time_point time_now = steady_clock::now();
  double delta_t = chrono::duration_cast<chrono::microseconds>(time_now - prev_time).count() / 1e6;
  //std::cout << "Debug: dt=" << delta_t << std::endl;

  // Proportional term
  p_error = -Kp * cte;

  // Integral term
  if (first_exec)
  {
    integral_error += cte * delta_t;
    i_error = -Ki * integral_error;
  }
  else
  {
    integral_error = 0.0;
    i_error = 0.0;
  }

  // Differential term
  if (first_exec)
  {
    d_error = 0.0; // don't introduce aritificial derivative spike when first run
  }
  else
  {
    const double min_delta_t = 1e-4;  // div by zero protection
    if (delta_t < min_delta_t) delta_t = min_delta_t;
    d_error = -Kd * (cte - prev_error) / delta_t;
  }

  // Store current CTE for next time
  prev_error = cte;

  first_exec = false;
  prev_time = steady_clock::now();
}
double PID::TotalError() {

  // CW: I haven't changed the provided names, but this is not about the
  // total error but rather the total of the P, I and D terms, i.e. the
  // controller output. They are not 'errors' and that is not what we're
  // computing here, I don't like the names.

  double total_control_output = p_error + i_error + d_error;

  return total_control_output;
}

