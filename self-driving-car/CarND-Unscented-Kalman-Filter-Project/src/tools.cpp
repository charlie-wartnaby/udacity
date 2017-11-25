////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : unscented Kalman filter project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    // Calculate the root mean square error over the set of estimated
    // and ground truth points provided.
    // Taken from previous (extended Kalman filter) project.
    // Note: following project template here, but not efficient because we
    // are recomputing RMSE over the entire dataset each time we get a new
    // measurement point; would be better to maintain the sums and just
    // divide by the new number of points at each timestep, or even use
    // a moving average over the last 'n' points, but OK for this project.

    int num_est_points = estimations.size();
    int num_truth_points = ground_truth.size();

    if (num_est_points <= 0 || num_truth_points <= 0 || num_est_points != num_truth_points)
    {
        cerr << "Error in CalculateRMSE(): supplied vectors must have non-zero and equal size" << endl;
    }

    int num_est_elements = estimations[0].size();
    int ground_t_elements = ground_truth[0].size();

    // Initialise return vector to a copy of a dynamically-created
    // vector of required size
    retVector_ = VectorXd::Zero(num_est_elements);

    if (num_est_elements <= 0)
    {
        cerr << "Error in CalculateRMSE(): supplied vectors must have non-zero size" << endl;
    }
    else if (num_est_elements != ground_t_elements)
    {
        cerr << "Error in CalculateRMSE(): supplied vectors must have equal size" << endl;
    }
    else
    {
        // Sizes valid so compute mean squared errors

        // Firstly sum of square residuals
        for (int i = 0; i < num_est_points; i++)
        {
            VectorXd residuals = estimations[i] - ground_truth[i];       // element-wise subtraction
            VectorXd resids_sqd = residuals.array() * residuals.array(); // achieves dot product
            retVector_ += resids_sqd;                                    // element-wise addition
        }

        // Form mean by dividing by number of samples
        retVector_ /= num_est_points;      // element-wise division

        // Finally, square root of that mean sum
        retVector_ = retVector_.array().sqrt();
    }

    return retVector_; // Should be copied by caller so we can reuse
}

double Tools::NormaliseAnglePlusMinusPi(double angle)
{
    // Normalises the angle provided to be in the interval [-pi, pi]

    // Not done by reference as we want to use it in at least one place
    // on a matrix element which we can't pass directly as a reference

    if ((angle >  20.0 * M_PI) ||
        (angle < -20.0 * M_PI) ||
        isnan(angle)             )
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
}
