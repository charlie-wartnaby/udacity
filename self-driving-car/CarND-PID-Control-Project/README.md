# PID Controller Project

Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.0

Date: 31Dec2017

## Notes against rubric

The Windows term 2 simulator car successfully gets round the track when run
using this controller in automonous mode as required, without popping up onto
kerbs or running into dirt areas etc.

The PID control code is implemented in a straightforward manner fitting the
template code structure, with most changes in PID.cpp. However, some changes
were required to cope with the differing behaviour of the simulator in
different graphics modes (see below). Ultimately that meant the controller
was not tuned optimally for any one simulator mode, but at least acceptably
for all modes, at least as run on my PC.

## Expected effects of P, I and D terms

The rubric asks for some description of the effect of the three different
terms in the PID controller.

### Proportional term

This provides the command to give a stronger control response if the system
is further from the desired setpoint. Hence it works to achieve the gross
correction required to achieve the setpoint. A larger gain value brings the system
to the setpoint faster; in this exercise, if the P term is too small
then the simulated car will run off at corners, continuing more or less
straight. However, if the P gain is too large, it results in overshoot
and oscillation.

### Integral term

The integral term 'learns' if there is a persistent error in the control
response. This copes with systematic errors in general. But in this exercise,
it also helps get the car round corners, where a persistent cross-track
error will build up unless steering is maintained even if the car is
currently centred. However, once the corner is complete, the accumulated
I term can result in overshoot and instability, so the gain must not be
too large.

### Differential term

By counteracting strong rates of change in the observed error, the D
terms helps damp overshoot and oscillation introduced by the P and I terms.
In general it may be necessary to increase or decrease the P and D gains
together, as they need to be in some approximate balance, which I did during
informal initial tuning. The D gain must not be too large however, as this
would result in large control outputs from what may be just small changes
due to noise in the measurement.

## Simulator Mode Problem

I initially tuned the PID controller successfully, using informal iterations
of the 'Twiddle' method presented in the lectures, with the Windows car
simulator in a high-resolution graphics mode.

However, when I happened to try running the simulator in a faster low-res
mode, the control was unstable; the car went through ever-widening
oscillations until crashing.

Conversely, attempting to tune the controller to work in 'fast' simulator
mode meant that it behaved in a very jerky fashion in hi-res 'slow' mode,
to the extent that it did not achieve the same road speed.

I worked on the hypothesis that the simulator was effectively working
in real time, and when working harder to produce graphics, it was
interrogating the controller less frequently to get new control settings.
Therefore the effective delta time between updates was longer in high-res
mode (executing asynchronously to the controller).

To counteract this, given that the simulator did not supply the delta
time value, I added code to measure the real elapsed delta time
as the controller ran with the simulator in different modes, and indeed
the actual delta time in real-world clock terms was very different:

|    Simulator mode    |    Real delta t (ms)   |
|:--------------------:|:--------------------------:|
| 1400x1050 Fantastic  |   ~70 (110+ on battery!)   |
| 320x240 Fastest      |   ~15                      |

So I used this real delta time in the calculation of the integral
and derivative terms. Like this, the same gains gave similar performance
in any simulator mode, allowing it to be tuned robustly. Even then,
having optimised the gains carefully for the simulator in medium-resolution
mode, I found I had to detune it somewhat to remain stable in the
highest-resolution mode, where it was working with a lower sample rate.

All this means that the gains I report may differ markedly from gains
calibrated to work with an assumed dt=1.0 sec however.

## Tuning process

I altered the template code in main.cpp so that the program accepted
arguments for the kP, kI and kD values on the command-line, so that
I could conveniently re-run it with different values without
recompilation.

I also had it compute a mean square cross-track error, to give me an
objective measure of the control quality. I ran the simulation for
exactly one lap to be able to compare error values with different gains.

I first found rough values which gave some kind of reasonable 
control by informal experiment. This meant increasing the P term until
the car didn't run off corners, and adding some D term to damp it,
and then feeding in some I term to improve its performance.

After that I used an informal approximation of the 'Twiddle' method
presented in the lectures, i.e. nudging each gain in turn in one
direction, the reverse direction if that made things worse, and reducing
the increment if both directions were worse. I did this with the
simulator in a middling 640x480 Normal mode (see above re simulator
mode effects). That converged OK on an appromimately optimal set
of values.

However, I then had to 'detune' the controller a little to make it stable
still with the simulator running in maximal 1400x1050 Fantastic mode,
which has greater sample time, despite trying to allow for this in the
control program by actually measuring delta time.

Here is the pattern of changes I made:

| kP | kI| kD | avg CTE^2 | Notes |
|:----:|:----:|:----:|:-----------:|:-------:|
| .12 | 2 | 0.1 | 0.537048 | Initial starting point with simlator in 640x480 Simple mode |
| .14 | 2 | 0.1 | 0.418203 | Better, kept new kP |
| .14 | 3 | 0.1 | 0.425437 | Worse |
| .14 | 1.5 | 0.1 | 0.428351 | Worse |
| .14 | 2 | 0.15 | 0.432055 | Worse |
| .14 | 2 | 0.07 | 0.417766 | Better, kept new kD |
| .16 | 2 | 0.07 | 0.35272 | Better, kept new kP |
| .16 | 2.3 | 0.07 | 0.34264 | Better, kept new kI |
| .16 | 2.3 | 0.05 | 0.370054 | Worse |
| .16 | 2.3 | 0.09 | 0.36315 | Worse |
| .18 | 2.3 | 0.07 | 0.300748 | Better, kept new kP |
| .18 | 2.6 | 0.07 | 0.297074 | Better (just), kept new kI |
| .18 | 2.6 | 0.08 | 0.301443 | Touch worse |
| .18 | 2.6 | 0.06 | 0.303282 | Touch worse |
| .21 | 2.6 | 0.07 | 0.256032 | Felt underdamped, but avg error lower |
| .21 | 2.8 | 0.07 | 0.2499915 | Bit better |
| .21 | 2.8 | 0.075 | 0.260377 | Worse |
| .21 | 2.8 | 0.065 | 0.255698 | worse |
| .25 | 2.8 | 0.07 | 0.219205 | Better |
| .25 | 3 | 0.07 | 0.210587 | Bit better |
| .25 | 3 | 0.073 | 0.221458 | Worse |
| .25 | 3 | 0.068 | 0.224773 | Worse |
| .3 | 3 | 0.07 | 0.211253 | Touch worse |
| .23 | 3 | 0.07 | 0.230357 | Worse |
| .25 | 3.3 | 0.07 | 0.214969 | Worse |
| .25 | 2.9 | 0.07 | 0.216155 | Worse |
| .25 | 3 | 0.07 | 0.197 | Rerun using 320x240 Fastest mode as check, performing well with these simulator settings OK |
| .25 | 3 | 0.07 |  | Rerun using 1400x1050 Fantastic mode as check, but got oscillatory and went off-track. Also on battery so CPU may be slowed down |
| .18 | 2.6 | 0.07 | 0.839868 | Continuing using 1400x1050 Fantasstic mode, dialled back gains, barely stable but got round |
| .14 | 2 | 0.07 | 0.57 | 1400x1050 Fantatic: Still swaying a fair bit but stayed on track for 2+ laps |
| .14 | 2 | 0.07 | 0.439877 | Rerun using 320x240 Fastest mode as check, pretty smooth though a bit wide on some corners |

So the end result was a bit of a compromise to work acceptably with all
simulator modes, but not really optimally in any one mode.

This data is copied & pasted from the colour-highlighted accompanying notes in
gain_optimisation.xlsx, which also charted previous tuning efforts
before fixing problems in the program to cope with the simulator mode
speed changes.