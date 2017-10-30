////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : extended Kalman filter project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include <iostream> // For debug output

#include "kalman_filter.h"
#include "tools.h"

using namespace std;

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */

    // For the state prediction, we just apply the state transition matrix to the
    // old state vector to get the new state vector; the state transition
    // matrix has already been set up correctly for the delta time since the
    // last measurement. We don't have accelerations as part of our state
    // vector, so this necessarily assumes constant-velocity motion as an
    // approximation (continually corrected by measurements of course).
    x_ = F_ * x_;

    // For the state covariance (uncertainty), use KF equation P' = F.P.F(T) + Q
    // In general the state should become more uncertain in the prediction step,
    // and the longer the delta time the worse it should get; dt is factored
    // into the current values of F and Q.
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */

    // Laser updates, use 'normal' linear update

    // Measurement error y = z - H.x'
    VectorXd H_x = H_ * x_;
    const VectorXd y = z - H_x;

    // The rest is common to laser and radar
    UpdateCommon(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */

    // Radar updates, use non-linear update, and compute measurement/prediction
    // error using cartesian-to-polar equation rather than matrix multiplication

    // First compute the radar coordinates we expect from the current predicted 
    // cartesian state variables
    VectorXd h = VectorXd(3);

    float px = x_[0];
    float py = x_[1];
    float vx = x_[2];
    float vy = x_[3];

    // Note: angle rho is anticlockwise from x-axis (NOT like tutorial video,
    // but consistent with lecture material derivations)
    float rho     = sqrt(px*px + py*py);       // Polar distance from us
    float phi, rho_dot;

    // Guard against undefined angle and div by zero if given (0,0) coords
    const float epsilon = 1e-6;
    if (rho >= epsilon)
    {
        // Normal calculation if we have a finite position vector
        phi = atan2(py, px);             // Rho is anticlockwise from x-axis
        rho_dot = (px * vx + py * vy) / rho; // Radial velocity away from us
    }
    else
    {
        // Backup if we essentially have a zero position vector
        phi = 0.0;
        rho_dot = 0.0;
    }

    h[0] = rho;
    h[1] = phi;
    h[2] = rho_dot;

    // Then measurement/prediction error y is z - h(x'), where z is latest radar measurement
    VectorXd y = VectorXd(z - h);

    // Project tips and tricks warned to keep angle in y normalised to [-pi, pi]
    y[1] = Tools::NormaliseAnglePlusMinusPi(y[1]);

    // The rest is common to laser and radar
    UpdateCommon(y);
}

void KalmanFilter::UpdateCommon(const VectorXd& y)
{
    // Having handled the different laser or radar parts, the rest
    // is common (note the H matrix will be replaced with either the laser version
    // or the radar Jacobian already by this point):

    // Update state vector
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;  // some measure of uncertainty
    MatrixXd Si = S.inverse();
    MatrixXd K = P_ * Ht * Si;  // Kalman gain
    x_ = x_ + (K * y);          // New state, stronger K -> measurement dominates prev estimate

    // Update covariance matrix; P' = (I - KH).P
    P_ -= K * H_ * P_; // updated uncertainty covariance
}
