# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.1

Date: 30Oct2017

## Notes against rubric

The template code provided in src has been completed so as to successfully implement
a Kalman filter fusing simulated laser and radar measurements to estimate the position
of a moving vehicle.

This built and executed successfully using the Windows 10 bash environment.

Using Dataset 1, the final Root Mean Square Error (RMSE) values between the estimated
trajectory and ground truth values for [x, y, vx, vy] are [0.0977, 0.0854, 0.4529, 0.4717],
which are all lower (better) than the required standard of [0.11, 0.11, 0.52, 0.52].

## Using laser or radar alone

As noted in FusionEKF.cpp, the simulation was run using only laser measurements, only
radar measurements, or both. Here were the resulting RMSE values:

Laser only: [0.1839 0.1542 0.6672 0.4836]

Radar only: [0.2324 0.3361 0.5327 0.7162]

Both:       [0.0977 0.0854 0.4529 0.4717]

While the best results are obtained fusing data from both sensors, qualitatively at least
a very reasonable trajectory was obtained using just one or the other.

## Changes to template code provided

Parts of the Kalman filter update step are common to the laser and radar algorithms,
so that was factored out into a common method KalmanFilter::UpdateCommon().

As provided, the program crashed when the simulator disconnected from it. This
was fixed in  h.onDisconnection() in main.cpp.

To aid debugging, the facility to work with just a few test points (either laser or radar)
was introduced in main.cpp if debug_laser or debug_radar are set true. Otherwise, the
program operates as a server for the simulator as intended.

## Changes after first review

All code review comments were actioned following first review. In particular, the bug
in Tools::CalculateRMSE() was fixed (missing square root at the end), which was
previously giving very flattering error values!
