# Model Predictive Control Controller Project

Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.0

Date: 05Feb2018

## Notes against rubric

The Windows term 2 simulator car successfully gets round the track when run
using this controller in automonous mode as required, without popping up onto
kerbs or running into dirt areas etc, with an artificial actuator latency
of 100 ms included.

The code builds without errors or warnings in a Windows bash environment using a makefile
generated from the provided CMakeLists.txt.

## Code origins and IPOPT

Chunks of the "Mind the Line" course quiz code were used to get something working
initially, but these needed significant changes which are described below.

The quiz code did at least provide an implementation which worked with the
CppAD IPOPT optimiser library, the workings of which were tricky to understand.
I commented extensively on how I believed the setup of the optimiser worked,
as this was far from clear to me in the example code.

## Handling latency

### Latency total

The latency (actuator delay) is set to 100 ms as required. This is now a variable
in main.cpp:

`const int actuator_delay_ms = 100;`

In addition to that delay, main.cpp is instrumented to measure the elapsed clock
time involved in setting up and solving the MPC optimisation at each iteration:

```
// Start measuring computing time to include in latency allowed for by MPC
high_resolution_clock::time_point t_compute_start = high_resolution_clock::now();
...
[all calculations in here]
...
high_resolution_clock::time_point t_compute_end = high_resolution_clock::now();
auto duration = duration_cast<milliseconds>(t_compute_end - t_compute_start).count();
compute_delay_ms = (int)duration;
cout << "Computation elapsed time = " << compute_delay_ms << "ms\n";
```

This allows the program to automatically compensate for the computational delay,
which could change significantly depending on how many points were fed to the
IPOPT optimiser.

### Algorithm changes to incorporate latency

In the quiz code, there was no concept of actuator latency, and the idea was
to use the i=0 values (corresponding to t=0) of actuator commands as the output
from the controller at each iteration.

However, with latency introduced, control would be unstable using the t=0
actuator values. Instead, the MPC solver should explicitly incorporate the
concept that the 'old' actuator values would still be in force at t=0, and
remain so until the latency period had expired. Therefore I altered the
code as follows:

* the initial actuator values (delta and a) were pinned at the 'old' values from
  the previous iteration of the solver using the upper and lower bounds:
  ```vars_lowerbound[delta_start] = vars_upperbound[delta_start] = steering_cmd_prev_;
  vars_lowerbound[a_start]     = vars_upperbound[a_start]     = throttle_cmd_prev_;```

* the first delta time value used in the computation was the latency, instead of
  the 'desired' dt value:
  ```double this_dt = (t_idx == 1 ? latency_sec : dt);```

* the i=1 actuator values from the optimised solution were returned as the
  'new' command values, not i=0, as these correspond to the acuation that
  should be applied after the latency period has expired for optimal control
  ```double new_steering_cmd = solution.x[delta_start + 1];
  double new_throttle_cmd = solution.x[a_start + 1];```

## Polynomial fit

For flexibility, the order of the polynomial used to fit the desired line
waypoints was kept as a hyperparameter, rather than being hard-coded:

```const int poly_order = 3; // Order of polynomial used to fit desired forward path; cubic recommended in lectures, even 2 gives decent fit```

The existing quiz code assumed only a first-order polynomial, and certainly
not a parameterised order. So wherever the polynomial or its derivative were
evaluated, this needed to be amended. The CppAD `Poly()` call was used,
which handily takes the required derivative as its first parameter
(i.e. 0 to evaluate the polynomial itself, 1 to obtain its first derivative, etc):

```
AD<double> y_approx_desired0 = CppAD::Poly(0, coeffs_, x0); // evaluate waypoint polynomial at x0, was 'f' in quiz code
AD<double> dy_dx_desired0    = CppAD::Poly(1, coeffs_, x0); // first derivative of waypoint polynomial at x0
```

This also required changes to the supplied `polyfit()` utility to compute derivatives,
to obtain the initial steering angle from the fit polynomial. I added a new parameter
for the required derivative, similar to the CppAD `Poly()` function.

## Evaluation of error terms

Starting with the quiz code, I found that the MPC "solution" fitted the desired
line reasonably in a straight line, but actually veered off in the opposite
direction in curves, almost a mirror image of the desired line. This problem
was reported on the Udacity forum here:
https://discussions.udacity.com/t/trajectory-seems-off-during-curvy-waypoints/272947

The cause of this problem was not obvious, as the evaluation of the trajectory
in FG_eval::() looked OK. However, I felt that the way that the cross-track
and psi (yaw) errors were extrapolated separately, as if they were independent
variables from the predicted track, was unnecessarily indirect. I replaced those
evaluations with a more obvious strategy of computing them directly from the
extrapolated 'y' and 'psi' values at the next time iteration in the kinematic
model which solved the problem; the MPC trajectory then converged on the
desired line:

```
AD<double> cte1_predicted  = y1_predicted - y1_desired;  // Have done my own thing here rather than extrapolating old CTE as if it were independent variable
AD<double> epsi1_predicted = psi1_predicted - psi_desired1; // Have used my own evaluation here, not linear extrapolation of old epsi suggested
```

I also restructured the code to first compute all of the state variables at the
next iteration, and only then compute the differences in `fg[]` which were
required to be set to zero by the optimiser, to make things easier to understand.

## Weighting

Starting with the quiz code cost function, I simply added weights to each term
which could be tuned as hyperparameters:

```
double weight_cte = 2.0;
double weight_epsi = 0.5;
... etc
fg[0] += weight_cte * CppAD::pow(vars[cte_start + t_idx], 2);
... etc
```

The weights were adjusted to give more influence to cross-track error and
less to smoothness, but actually the behaviour was fairly insensitive to changes
in these parameters.

## Tuning and performance

I started with N=10 and dt=0.1, but found that while a good trajectory could be
found for some time, the IPOPT optimiser could then start returning wildly
'snakey' trajectory solutions that would cause the car to go off-track.

For robustness, and much faster optimisation, I found that a much smaller number
of points with wider time spacing (leaving IPOPT with far fewer independent
variables to optimise) was far more robust:

```
size_t N = 4;             // Number of forward timesteps to compute; solver fit much more robust using few points.
double dt = 0.25;          // Delta time (sec) between timesteps
```

Even then I found I had to tune the dt value to give good enough control for
the car to negotiate the track reliably, without extending the time horizon
so far that unstable behaviour resulted from the optimiser.
