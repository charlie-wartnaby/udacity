////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : extended Kalman filter project.
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
    // Note: following project template here, but not efficient because we
    // are recomputing RMSE over the entire dataset each time we get a new
    // measurement point; would be better to maintain the sums and just
    // divide by the new number of points at each timestep, or even use
    // a moving average over the last 'n' points, but OK for this project.

    int num_est_points = estimations.size();
    int num_truth_points = ground_truth.size();

    if (num_est_points <= 0 || num_truth_points <= 0 || num_est_points != num_truth_points)
    {
        cout << "Error in CalculateRMSE(): supplied vectors must have non-zero and equal size" << endl;
    }

    int num_est_elements = estimations[0].size();
    int ground_t_elements = ground_truth[0].size();

    // Initialise return vector to a copy of a dynamically-created
    // vector of required size
    retVector_ = VectorXd::Zero(num_est_elements);

    if (num_est_elements <= 0)
    {
        cout << "Error in CalculateRMSE(): supplied vectors must have non-zero size" << endl;
    }
    else if (num_est_elements != ground_t_elements)
    {
        cout << "Error in CalculateRMSE(): supplied vectors must have equal size" << endl;
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

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
    // Reusing my quiz code from a lesson, but fixing it so the return matrix
    // is not created on the stack (where it will be destroyed and cannot
    // be safely accessed after this function exits)
    retMatrix_ = MatrixXd(3, 4);  // create new object to return

    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //TODO: YOUR CODE HERE

    // Using my solution code to "Lesson 18: Jacobian Matrix Part 1" quiz

    // Precompute factors that are used repeatedly, to reduce CPU use
    // (especially in square roots) and to simplify downstream code
    float px2_plus_py2         = px*px + py*py;

    // Guard against division by zero before using that further
    const float epsilon = 1e-12;
    px2_plus_py2 = std::max(epsilon, px2_plus_py2);

    float root_px2_plus_py2    = sqrt(px2_plus_py2);
    float px2_plus_py2_pow_3_2 = px2_plus_py2 * root_px2_plus_py2;


    // Compute the Jacobian matrix
    // This specifically computes the Jacobian mapping the partial derivatives
    // of the Cartesian position and velocity components into the space of
    // the radar variables (polar coordinates and radial velocity)
    retMatrix_(0, 0) = px / root_px2_plus_py2;
    retMatrix_(0, 1) = py / root_px2_plus_py2;
    retMatrix_(0, 2) = 0;
    retMatrix_(0, 3) = 0;
    retMatrix_(1, 0) = -py / px2_plus_py2;
    retMatrix_(1, 1) = px / px2_plus_py2;
    retMatrix_(1, 2) = 0;
    retMatrix_(1, 3) = 0;
    retMatrix_(2, 0) = (py * (vx * py - vy * px)) / px2_plus_py2_pow_3_2;
    retMatrix_(2, 1) = (px * (vy * px - vx * py)) / px2_plus_py2_pow_3_2;
    retMatrix_(2, 2) = px / root_px2_plus_py2;
    retMatrix_(2, 3) = py / root_px2_plus_py2;

    return retMatrix_; // To be copied by caller so we can reuse our object
}

float Tools::NormaliseAnglePlusMinusPi(float angle)
{
    // Normalises the angle provided to be in the interval [-pi, pi]

    // Not done by reference as we want to use it in at least one place
    // on a matrix element which we can't pass directly as a reference

    const float pi = 3.141592654;

    while (angle < -pi)
    {
        angle += 2 * pi;
    }

    while (angle > pi)
    {
        angle -= 2 * pi;
    }

    return angle;
}