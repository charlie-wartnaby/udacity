////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : Kidnapped vehicle (particle filter) project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <cassert>

#include "particle_filter.h"

using namespace std;


// To avoid divisions by zero etc:
const double epsilon = 1e-10;

// When looking for shortest distance:
const double very_large_distance = 1e10;
// CW additional helper functions; put in here rather than in
// helper_functions.h because only this file is submitted
// for assessment, so want to ensure it builds with other
// files left as they were
ostream& operator<<(ostream& os, const Particle& p)
{
    os << "p[id=" << p.id << "] (x,y,theta)=(" << p.x << "," << p.y << "," << p.theta << ") w=" << p.weight;
    return os;
}

// Taken from Kalman filter project submissions:
double NormaliseAnglePlusMinusPi(double angle)
{
    // Normalises the angle provided to be in the interval [-pi, pi]

    //******* PROBLEM WORKAROUND **********
    // From forum, simulator does not expect normalised angles so get big error reported if
    // do this! Get large error if allow [-pi,pi) range, and smaller error but still big
    // enough to fail the test if allow [0, 2pi) range.
    // https://discussions.udacity.com/t/yaw-error-larger-than-the-max/306716/10
    // So stubbing out normalisation here, to work around simulator problem:
    return angle;

    /*
    // Not done by reference as we want to use it in at least one place
    // on a matrix element which we can't pass directly as a reference

    if ((angle >  20.0 * M_PI) ||
        (angle < -20.0 * M_PI) ||
        isnan(angle))
    {
        // Angle looks implausible. To avoid locking up with very lengthy
        // or even infinite* loop, force it to zero (with a warning).
        // (Shouldn't happen unless we get wildly wrong data, e.g. restarting
        // simulator at completely different point.)
        // *Can get infinite loop if number is so large that subtracting 2.pi
        // leaves same number behind, in double representation.
        cerr << "WARNING: angle=" << angle << " replaced with zero" << endl;
        angle = 0.0;
    }
    else
    {
        while (angle < -M_PI)
        {
            angle += 2 * M_PI;
        }

        while (angle > M_PI)
        {
            angle -= 2 * M_PI;
        }
    }

    return angle;
    */
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Start with modest number so not too slow, and see how it performs (passes as submitted)
    num_particles = 50;

    // Gaussian distribution setup taken from my solution code to
    // "Lesson 14: Implementation of a Particle Filter/4. Program Gaussian Sampling: Code"
    default_random_engine gen;

    normal_distribution<double> distrib_x    (x,     std[0]);
    normal_distribution<double> distrib_y    (y,     std[1]);
    normal_distribution<double> distrib_theta(theta, std[2]);

    for (int i = 0; i < num_particles; i++)
    {
        Particle p = Particle();
        p.id = i;

        p.x = distrib_x(gen);
        p.y = distrib_y(gen);

        // Avoid angles outside range[-pi,pi] that might be generated from statistical
        // distribution if mean angle is close to one of those limits
        p.theta = NormaliseAnglePlusMinusPi(distrib_theta(gen));

        p.weight = 1.0;

        // associations, sense_x and sense_y left empty for now

        particles.push_back(p);
    }

    // Filter now ready to run
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.

    // CW: why does it say measurements there? The prediction depends on our previous state,
    // control inputs and noise, not measurements. And the supplied std_pos[]
    // is the uncertainty in absolute GPS/compass measurements (x, y, theta), not
    // in the velocity and yaw rate, so it doesn't make sense to use those here.
    // Instead we should have some control input noise factors.

    // For now will just add those absolute noise factors but it seems a bit wrong.

	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


    if (fabs(yaw_rate) < epsilon)
    {
        // Yaw rate is about zero so use simple straight-line extrapolation of
        // current position along velocity vector, avoiding division by zero
        double vt = velocity * delta_t;
        for (int i = 0; i < num_particles; i++)
        {
            particles[i].x += vt * cos(particles[i].theta);
            particles[i].y += vt * sin(particles[i].theta);
        }
    }
    else
    {
        // Finite yaw rate so integrate position along circular arc
        // assuming CTRV (constant turn rate and velocity) model
        double v_by_theta_dot = velocity / yaw_rate;
        double delta_theta = yaw_rate * delta_t;
        
        for (int i = 0; i < num_particles; i++)
        {
            double theta = particles[i].theta;
            double new_theta = NormaliseAnglePlusMinusPi(theta + delta_theta); // probably unnecessary as normalise again after noise added
            particles[i].x += v_by_theta_dot * (sin(new_theta) - sin(theta));
            particles[i].y += v_by_theta_dot * (cos(theta) - cos(new_theta));
            particles[i].theta = new_theta;
        }
    }

    // Now adding random noise to each particle's x, y and theta

    // Gaussian distribution setup taken from my solution code to
    // "Lesson 14: Implementation of a Particle Filter/4. Program Gaussian Sampling: Code"
    default_random_engine gen;

    // This time we just want noise offset, so centre each on zero
    normal_distribution<double> distrib_x    (0.0, std_pos[0]);
    normal_distribution<double> distrib_y    (0.0, std_pos[1]);
    normal_distribution<double> distrib_theta(0.0, std_pos[2]);

    for (int i = 0; i < num_particles; i++)
    {
        particles[i].x += distrib_x(gen);
        particles[i].y += distrib_y(gen);

        // Avoid angles outside range[-pi,pi] that might be generated from statistical
        // distribution if mean angle is close to one of those limits
        particles[i].theta = NormaliseAnglePlusMinusPi(particles[i].theta + distrib_theta(gen));
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // CW: the prototype for this method implies we should compare landmarks
    // to observations in the vehicle's frame of reference. But that seems inefficient;
    // to do that for each particle we end up converting the map landmarks repeatedly
    // into different particle's reference frames, at each step.
    // Surely better to do the comparison in map coordinates, so that
    // the map coords never need transformation, by transforming the
    // observations for this particle at this time into map coords instead.
    // At least because the number of observations is likely to be small
    // compared to the complete set of known landmarks -- couldn't transform
    // a map with a million landmarks into particle's frame every step!
    // Hence I haven't completed this function.
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // A bit oddly perhaps, we're given the standard deviation in
    // landmark coords rather than in the measurements of observations.
    // Never mind, will assume the stdevs we're given are the ones to use
    // in our covariance matrix.

    // Contribution factor to the overall weight for each observation/landmark pair is
    //   exp(-1/2 (x - mu)' SIG^-1 (x - mu)) / sqrt(|2.pi.SIG|)
    // The denominator sqrt(|2.pi.SIG|) is constant so we can precompute that just once.
    // SIG is the diagonal matrix with elements (sig-x, sig-y)
    // Magnitude of SIG is sqrt(sig-x^2 + sig-y^2), taking out factor of 2.pi
    const double sig_x = std_landmark[0];
    const double sig_y = std_landmark[1];
    const double sigma_magnitude = sqrt(sig_x * sig_x + sig_y * sig_y);
    const double weight_denom = sqrt(2 * M_PI * sigma_magnitude);

    // Inverse covariance matrix SIG^-1 is just diagonal matrix with elemnts
    // taking reciprocal values of the original diagonal matrix SIG.
    const double sig_inv_00 = 1.0 / sig_x;  // element (0,0)
    const double sig_inv_11 = 1.0 / sig_y;  // element (1,1)

    for (int pIdx = 0; pIdx < num_particles; pIdx++)
    {
        Particle& p = particles[pIdx];

        double weight = 1.0; // Before contribs from each observation multiplied in

        // Transform observations (in this particle's frame) into map coords
        // so that we can compare them to the list of known landmarks.
        // Note: the template code seems us to want to do the opposite
        // (see comments in dataAssociation() above), but that seems crazy:
        // if we have a million known landmarks and only 5 observations,
        // it's obviously better to compare in the map frame rather than
        // converting all the map coords into the vehicle frame, for every
        // particle, at every timestep!

        double cos_theta = cos(p.theta);
        double sin_theta = sin(p.theta);

        for (int obsIdx = 0; obsIdx < observations.size(); obsIdx++)
        {
            Map::single_landmark_s landmarkFromObs;

            // Formula for expanded matrix transform multiplication from
            // "16. Landmarks Quiz Solution" in lectures
            landmarkFromObs.id_i = -1; // associate later
            landmarkFromObs.x_f = p.x + cos_theta * observations[obsIdx].x - sin_theta * observations[obsIdx].y;
            landmarkFromObs.y_f = p.y + sin_theta * observations[obsIdx].x + cos_theta * observations[obsIdx].y;

            // Now we have this observation in map coordinates, search through
            // the list of map landmarks to identify the nearest neighbour, i.e.
            // the known landmark closest to the observation coords in space
            double smallest_distance = very_large_distance;
            int nearest_map_idx = 0;
            for (int landmarkIdx = 0; landmarkIdx < map_landmarks.landmark_list.size(); landmarkIdx++)
            {
                double landmark_observation_dist = dist(landmarkFromObs.x_f,
                                                        landmarkFromObs.y_f,
                                                        map_landmarks.landmark_list[landmarkIdx].x_f,
                                                        map_landmarks.landmark_list[landmarkIdx].y_f);

                if (landmark_observation_dist < smallest_distance)
                {
                    nearest_map_idx = landmarkIdx;
                    smallest_distance = landmark_observation_dist;
                }
            }

            // Now we have associated this observation with the nearest
            // landmark, we can compute its contribution to the overall
            // weight (as a factor in the multivariate Gaussian probability
            // density)
            // CW note: it worries me that this just gets smaller and smaller
            // as we multiply in more observations, whereas if we had no observations
            // we'd keep a weight of 1! Surely the weight should ne normalised
            // somehow so that it doesn't depend on the number of measurements,
            // but rather their average quality in some sense... but at least number
            // of measurements is the same for each particle at each iteration so
            // OK I guess, in that it's all relative.
         
            // Numerator for weight contribution is exp(-1/2 (x - mu)' SIG^-1 (x - mu))
            double delta_x = map_landmarks.landmark_list[nearest_map_idx].x_f - landmarkFromObs.x_f;
            double delta_y = map_landmarks.landmark_list[nearest_map_idx].y_f - landmarkFromObs.y_f;
            // Expanded-out matrix multiplication, taking advantage of knowing that inverse
            // sigma matrix is diagonal:
            double pre_exp = delta_x * sig_inv_00 * delta_x +
                             delta_y * sig_inv_11 * delta_y;
            
            double weight_num = exp(-0.5*pre_exp);
            double weight_contrib_for_obs = weight_num / weight_denom;

            // Finally multiply this in to accumulated overall weight for particle
            weight *= weight_contrib_for_obs;

        } // loop over observations for this particle

        // We now have the new weight
        p.weight = weight;

    } // loop over particles
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    // CW: based on my solution code to "20. Quiz: Resampling Wheel", here in Python:
    //max_w = max(w)
    //beta = 0
    //index = int(random.random() * N)
    //for i in range(N) :
    //    beta += random.random() * 2 * max_w
    //    while beta >= w[index]:
    //        beta -= w[index]
    //        index = (index + 1) % N
    //    p3.append(p[index])

    // As the vector of particles is owned directly as a member variable of
    // this class (not a reference or pointer to such a vector), it's a bit
    // tricky to replace in place with a new vector. But instead I'll
    // append the "new" particles to the end of the list as we create them
    // (by copying existing ones biased by their weights), and finally remove
    // the "old" first half of the vector after we're done so it is effectively
    // replaced with the new resampled set.

    // Find maximum weight in existing set of particles
    double max_weight = -1.0; // definitely beaten by first entry
    int max_weight_idx = -1;
    for (int i = 0; i < num_particles; i++)
    {
        if (particles[i].weight > max_weight)
        {
            max_weight = particles[i].weight;
            max_weight_idx = i;
        }
    }

    // Prepare to iterate over reselection wheel
    double beta = 0.0;
    int num_sampled = 0;
    default_random_engine gen;
    // Start with random particle index in the set
    std::uniform_int_distribution<> distrib_int_num_p(0, num_particles - 1);
    int wheel_idx = distrib_int_num_p(gen);
    // Set up distribution to sample beta up to double max weight
    std::uniform_real_distribution<> distrib_real_2w(0.0, 2.0 * max_weight);

    // Reselect the same number of particles we had last time, but 
    // sampling copies in proportion to their weights
    while (num_sampled < num_particles)
    {
        beta += distrib_real_2w(gen);
        while (beta >= particles[wheel_idx].weight)
        {
            beta -= particles[wheel_idx].weight;
            wheel_idx = (wheel_idx + 1) % num_particles;
        }
        particles.push_back(particles[wheel_idx]);
        num_sampled++;
    }

    // Check we ended up with a new set of particles with same number of entries
    assert(particles.size() == num_particles * 2);

    // Remove the old particles (now first half of vector)
    particles.erase(particles.begin(), particles.begin() + num_particles);
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
