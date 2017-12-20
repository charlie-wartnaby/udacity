# Unscented Kalman Filter Project

Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.0

Date: 25Nov2017

## Notes against rubric

The template code provided in src has been completed so as to successfully implement
an Unscented Kalman Filter fusing simulated laser and radar measurements to estimate the position
of a moving vehicle.

This built and executed successfully using the Windows 10 bash environment.

Using Dataset 1, the final Root Mean Square Error (RMSE) values between the estimated
trajectory and ground truth values for [x, y, vx, vy] are [0.0614, 0.0849, 0.3510, 0.2191],
which are all lower (better) than any of the required standards for the different datasets
in the rubric. Note: the rubric mentioned old and new versions of the project, and text filenames
such as "obj_pose-laser-radar-synthetic-input.txt"; but the term 2 Windows simulator used
no such explicit text files, so I do not know for sure which version it is in those terms.

## Using laser or radar alone

By setting UKF::use_laser_ or UKF::use_radar_ to false, the filter could be initialised and
run using only one of those types of measurement. The resulting RMSE values were:

Laser only: [0.1575	0.1462	0.4716	0.2936]

Radar only: [0.2020	0.2280	0.4482	0.3052]

Both:       [0.0614	0.0849	0.3510	0.2191]

While the best results are obtained fusing data from both sensors, qualitatively at least
a very reasonable trajectory was obtained using just one or the other.

## Parameter optimisation

The simulation was rerun numerous times with different values of the parameters intended
for student tuning. The final values chosen which were about optimal were:

Process noise in linear acceleration (std_a_) = 0.5 m/s^2
Process noise in angular acceleration (std_yawdd_) = 0.5 rad/s^2
Spreading parameter (lambda_) = 3 - n_aug_

## Normlised Innovation Squared (eta)

To check parameter consistency, eta values for laser and radar measurements were sent
to standard output such that they could be redirected to a file. The values were:

laser: eta = 1.80 average +/- 1.82 standard deviation (2 DoF, expect 0.10 to 5.99)
radar: eta = 2.86 average +/- 2.24 standard deviation (3 DoF, expect 0.35 to 7.82)

These values are within the expected ranges, indicating that the parameters were
reasonably chosen, as we would expect given that they also yielded the lowest
RMSE values.

## Changes to template code provided

Parts of the Kalman filter update step are common to the laser and radar algorithms,
so those were factored out into a common methods to avoid duplication, to improve
maintainability and reduce the chance of coding errors.

Some effort was expended making the program robust against the simulator being
restarted; if this is detected, it re-initialises the filter. Infinite loops are
prevented in angle normalisation (due to implausibly large angles emerging)
are avoided, with a warning.

As provided, the program crashed when the simulator disconnected from it. This
was fixed in  h.onDisconnection() in main.cpp.

To aid debugging, the facility to work with just a few test points (either laser or radar)
was introduced in main.cpp if debug_laser or debug_radar are set true. Otherwise, the
program operates as a server for the simulator as intended.
