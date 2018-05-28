# Path Planning Project

Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.0

Date: 28May2018

## Notes against rubric

The Windows term 3 simulator car successfully drives round the track for
more than 4.3 miles without incident, staying close (but not exceeding)
the 50 mph speed limit when possible, and changing lanes to pass slower
cars. Incidents automatically flagged by the simulator include collisions,
speed limit violations, excessive acceleration or jerk, and time outside
lanes, so all of these have been avoided.

Lane changes are achieved smoothly using a spline trajectory.

Speed changes are achieved smoothly using a controller that filters
the current desired acceleration to avoid jerk.

It was tested with the simulator running both in lowest-resolution,
highest-speed mode and highest-resolution, maximum quality mode, requiring
some changes to make it automatically adapt to quite different run-time calling
rates.

The code builds without errors or warnings in a Windows 10 bash environment using a makefile
generated from the provided CMakeLists.txt.

## Path generation detail

### Basic behavioural algorithm

At each iteration, the program iterates over all other vehicles in the sensor fusion list,
and identifies (within tolerance parameters) whether there is another car a) in front of
it in the same lane, b) in the lane to its left, or c) in the lane to its right. It then
behaves as follows:

1. If it is not already in the rightmost ("slow") lane and the lane to its right is
vacant, it starts a lane change to the right. This fits with European roads where
drivers are expected to occupy that lane until such time as they wish to overtake.
This also means it will change lanes if blocked by a vehicle in front and the right
lane is free.
2. Otherwise, if there is no car in front, it remains in lane, aiming to maintain just
under the posted speed limit of 50 mph.
3. Otherwise, blocked in by a car in front, it changes lane to the left if it is not
already in the leftmost lane and that is vacant.
4. Otherwise, it remains blocked by the car in front, and adjusts its target speed to
be just a little below that of the car in front, to avoid colliding with it.

This is implemented here:

```
if (ok_to_change_lanes && right_available && desired_lane_idx < num_lanes - 1)
{
    // Just for fun, prefer inner ("slow") lane if available, which is what we do
    // in Europe at least. This also covers the case where we happen to be stuck
    // behind another car in this lane but the right lane is free.
    desired_lane_idx++;
}
else if (!ahead_limiting)
{
    // Stay in current lane, as there is nothing in our way
}
else if (ok_to_change_lanes && left_available && desired_lane_idx > 0)
{
    // We are stuck behind another car but the lane to our left is free, so switch
    desired_lane_idx--;
}
else
{
    // Stuck behind another vehicle in this lane so just regulate speed to avoid it
    target_speed_m_s = this_lane_max_speed;
}
```

### Track wraparound

The rubric requires that the car drives safely for at least 4.32 miles, but this number is
not explained. It is in fact the distance round the track, and hence maximum Frenet
s-coordinate value seen. So the rubric limit is quietly saving we students from having
to worry about odd behaviour when the track wraps around.

But to be robust, this program allows for the fact that the Frenet s-coordinate
wraps around, e.g. if another car has s=0.01 miles and our ego car has s=4.31 miles,
actually this means the other car is just 0.02 miles ahead of the ego car, and should
be considered as a nearby target.

The maximum s-coordinate seen for any car (ego or other) is tracked for this purpose
by function `check_for_greatest_s()`.

### Finite state machine

The behaviour of the car is implicitly a finite state machine, in that once it is
committed to a lane change, this is completed, and no new lane change is allowed
for a few seconds. Hence "changing lanes" is effectively a different state to
"keeping in lane". This is implemented as required minimum time between lane changes,
to avoid the possibility of repeated lane-change decisions which might lead to the
car illegally remaining between lanes for more than the allowed time if by chance
the traffic situation is similar in the two lanes:

```
// Avoid rapid consecutive lane change decisions, otherwise could end up triggering
// out of lane warning if keep changing mind due to similar traffic ahead in
// two lanes which might leave us driving between them for a while
double min_time_between_lane_changes = 4.0;
bool ok_to_change_lanes = (time_since_last_lane_change >= min_time_between_lane_changes);
```

### Space trajectory generation

The detailed forwards path is computed using the method shown in the project walkthrough
video, with some small changes.

Firstly, any unused path points sent to the simulator last time that have not yet been
consumed are retained as the first part of the desired path for this time.

Secondly, a spline is fitted to a few well-separated points taken from the map ahead
of the ego car, centred in the desired lane. There is a tradeoff here:
* Wider spacing between distant points gives a smoother trajectory with less
  lateral acceleration and jerk, but less accurate lane following.
* Closer spacing between distant points follows the lane more accurately, but
  starts to exhibit the discontinuities present in the map waypoints, leading
  to unwanted lateral jerk.

As a compromise, the spacing between future points is somewhat shorter when simply
keeping lane (the normal case), and somewhat larger when manouevering to a different
lane, when lateral acceleration and jerk would otherwise be significant:

```
double dist_between_distant_points = changing_lanes ? 60 : 30;
```

Points at the required simulation time interval (0.02 sec) are then taken from the
spline fit through those distant lane-centre points, and appended to the 
unconsumed path points from last time.

In a small change from the walkthrough video, the current desired acceleration is
included in the computation of the projected waypoints, to avoid jerk.

The spline fit is made in ego-relative coordinates to ensure it is a single-valued
function y(x) for moderate x values. In global coordinates, it would sometimes be
multi-valued, and a spline fit would fail (e.g. if the track was curving around
the northwards direction). This is as explained in the walkthrough video. After
the fit is made, the points are converted back to x-y global coordinates as
required by the simulator.

### Speed control mechanism

Vehicle speed is controlled as follows, to avoid longitudinal jerk:

```
// Adjust speed towards target, limiting acceleration and jerk.
// To do this, I've maintained a value of the current acceleration, which we do
// not want to change too abruptly (i.e. cause jerk). This is smoothly pushed
// towards our currently desired acceleration via a first-order filter to
// avoid abrupt changes. The filter takes account of the real-time rate at which
// this function is being called, which depends quite a lot on simulator mode.
//
// Start with desired raw acceleration; needs to be quite strong to avoid collisions.
// Want to accelerate if going too slowly, decelerate if going too fast, compared
// with currently desired speed:
double desired_accel = (target_speed_m_s - current_speed_m_s) * 0.25; // at zero to get to 20 m/s ~4 m/s2

// First-order filter on acceleration to avoid sudden changes (jerk):
double accel_filter_const = 2.0; // by experiment
current_accel_m_s2 += (desired_accel - current_accel_m_s2) * delta_t_program * accel_filter_const;

// Clip acceleration to avoid exceeding allowed maximum:
double max_accel = 3.0; // too close to limit of 5 and risk acceleration warnings
if (current_accel_m_s2 > max_accel) current_accel_m_s2 = max_accel;
if (current_accel_m_s2 < -max_accel) current_accel_m_s2 = -max_accel;

// Now acceleration is computed, update current speed, allowing for real-time
// calling rate of this code:
current_speed_m_s += current_accel_m_s2 * delta_t_program;
```

### Simulator problems

With the ego vehicle proceeding at the speed limit, it was still possible for
it to be "caught out" by one of the
other cars cutting in front of it at much lower speed. This was seen a couple of times
with the simulator in minimum-resolution, maximum speed mode (that might be a 
coincidence). Indeed at least one crash between other vehicles was seen in the
simulator unrelated to the ego car, perhaps because of this same problem.

In a real system, we could apply
maximum braking force in that situation. Here that would be penalised anyway as exceeding
tolerable acceleration or jerk.

As a workaround, we conservatively pass vehicles in adjacent lanes only slowly,
so that the algorithm controlling those other vehicles is less inclined to have them
cut abruptly in front of the ego vehicle. See `safe_overtake_speed_propn`:

```
// Avoid passing too fast as simulator may make other vehicle cut in front of us
double safe_speed = speed * safe_overtake_speed_propn;
if (safe_speed < target_speed_m_s) target_speed_m_s = safe_speed;
```
