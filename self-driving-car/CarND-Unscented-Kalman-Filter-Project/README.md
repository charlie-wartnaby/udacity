# Kidnapped Vehicle (Particle Filter) Project

Self-Driving Car Engineer Nanodegree Program

Author: Charlie Wartnaby, Applus IDIADA

Email: charlie.wartnaby@idiada.com

Version: 1.0

Date: 20Dec2017

## Notes against rubric

The Windows term 2 simulator reports success when run against this implementation.

## Comments and problems

As in the last two projects, I modified main.cpp so that the program did not crash
if the simulator is disconnected.

The simulator was reporting a large yaw error, when actually the yaw angle agreed well
with the ground-truth trajectory. Following a tip on the forum, I found this was due
to my normalisation of yaw angle to a range of [-pi,pi]. It turned out that the
simulator did not expect any normalisation and failed if I normalised the angle to
either [-pi,pi] or [0,2.pi]. Hence I stubbed out the angle normalisation.

I did not fill the dataAssociation() method provided, because I felt its prototype
encouraged handling the data the wrong way round (converting map landmarks to the
vehicle frame for each particle at each sample, instead of converting vehicle
frame observations to map coordinates, which is much more efficient). See detailed
comments in the code there.

In updateWeights() we were provided with standard deviations in landmark
coordinates, but no expected error in sensor measurements. I felt this might be the
wrong way round, but used the landmark errors anyway, with comments explaining this.