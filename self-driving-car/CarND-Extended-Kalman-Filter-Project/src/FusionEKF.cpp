////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : extended Kalman filter project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  ekf_ = KalmanFilter();

  // Initial state vector: will get initialised to first actual measurements,
  // so no need to put anything sensible in at this point
  VectorXd X_init = VectorXd(4);

  // Initial state covariance matrix; assume large uncertainties, will quickly
  // come down as measurements feed in. Uncorrelated so diagonal elements
  // only
  MatrixXd P_init = 100.0 * MatrixXd::Identity(4, 4);

  // State transition matrix; some elements will depend on delta time, but
  // mostly constant, so initialise what we can (will just be identity
  // matrix without dt-dependent terms in fact, i.e. old coordinates and
  // velocities unchanged). This will require update at run time when each
  // measurement point gives us a new delta t for the off-diagonal elements.
  MatrixXd F_init = MatrixXd::Identity(4, 4);

  // Laser measurement matrix just selects x and y from current state,
  // as lidar gives us no velocity information
  H_laser_ << 1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0;

  // Process noise covariance matrix gets re-initialised at run time before use
  // so not set to anything sensible here
  MatrixXd Q_init = MatrixXd(4, 4);

  // Nominally initialising Kalman filter object, though several of the matrices here
  // are either initialised later or are switched at run-time depending on
  // the measurement type:
  ekf_.Init(X_init,   // State vector gets initialised on first measurement later
            P_init,   // State covariance initialised to large numbers on diagonal (uncertainty in state)
            F_init,   // State transition matrix gets run-time updates, dt-dependent
            H_laser_, // Measurement function will be switched between laser and radar at run time
            R_laser_, // Measurement noise will be switched between laser and radar at run time
            Q_init);  // Process noise gets run-time rebuild, dt-dependent

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

    // EXPERIMENT ONLY; try skipping either laser or radar measurements to see
    // how well it does using only one type of sensor, uncomment one of these
    // lines to ignore one of the sensor types
    //if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) return;
    //if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) return;
    
    // RESULTS: still followed trajectory pretty well using only laser or only
    // radar, but final RMSE values better using both, as we'd expect:
    //   sensor(s) RMSE x    y      vx     vy
    //    laser    0.1839 0.1542 0.6672 0.4836
    //    radar    0.2324 0.3361 0.5327 0.7162
    //    both     0.0977 0.0854 0.4529 0.4717


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
        // Expect distance (rho), angle anticlockwise from x-axis (phi)
        // and velocity away from us (rho-dot); note that phi does not have
        // same convention as tutorial diagram where it was clockwise from x-axis
        // (in video "14. Radar Measurements" in "Lesson 5: Lidar and Radar Fusion..."),
        // think that video diagram was wrong though as nearby derivations had phi going +ve
        // for +ve y, consistent with phi being anticlockwise from x-axis.
        assert(measurement_pack.raw_measurements_.size() == 3);
        float rho     = measurement_pack.raw_measurements_[0];
        float phi     = measurement_pack.raw_measurements_[1];
        float rho_dot = measurement_pack.raw_measurements_[2];

        ekf_.x_ << rho     * cos(phi),  // x forwards (to the right on simulator display)
                   rho     * sin(phi),  // y upwards
                   rho_dot * cos(phi),  // vx
                   rho_dot * sin(phi);  // vy
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
		// Expect x and y coordinates where x is forwards and y to the left
        // assuming same convention as tutorials
		assert(measurement_pack.raw_measurements_.size() == 2);
		ekf_.x_ << measurement_pack.raw_measurements_[0],  // x
                   measurement_pack.raw_measurements_[1],  // y
                   0.0, 0.0; // Velocity components unknown so assume zero initially
    }
	else
	{
		cout << "ERROR: unexpected sensor type during initialisation" << endl;
        ekf_.x_ << 0, 0, 0, 0; // Initial state unknown
    }

    // Also grab this first timestamp so we can compute delta time
    // from this point onwards
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    return;
  }


  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */

  // Get delta time in sec, assuming measurement timestamps are in usec
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0F;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Update the state transition matrix which we multiply into the state
  // vector to get a new state estimate; assuming constant velocity motion
  // as an approximation, this just means updating the spatial coordinates
  // by the velocities * delta time, while the velocities are assumed unchanged.
  // The diagonal elements all remain at 1 and require no run-time update.
  ekf_.F_(0, 2) = dt; //  x' = x + vx.dt hence 1, 0, dt, 0
  ekf_.F_(1, 3) = dt; //  y' = y + vy.dt hence 0, 1,  0, dt
                      // vx' = vx        hence 0, 0,  1, 0  (no update required for dt)
                      // vy' = vy        hence 0, 0,  0, 1  (no update required for dt)


  // The process noise covariance matrix Q (i.e. uncertainty added to state each
  // time we extrapolate to a new predicted state) is delta time dependent, because
  // the longer the dt we extrapolate over, the more uncertain our prediction is.
  
  const float noise_ax = 9.0, noise_ay = 9.0; // as directed above in template comment

  // The rest is my solution code from the relevant tutorial quiz "12. Laser Measurements Part 3"

  // Precompute some common factors to reduce CPU and code complexity
  float dt2 = dt * dt;
  float dt3_over_2 = dt2 * dt / 2.0;
  float dt4_over_4 = dt2 * dt2 / 4.0;
  const float nax2 = noise_ax; // These noise terms were aleady squared in quiz, which wasn't obvious!
  const float nay2 = noise_ay; // (Are they here? Seems to make no difference anyway so assuming same as quiz.)
  ekf_.Q_ << dt4_over_4 * nax2, 0,                 dt3_over_2 * nax2, 0,
             0,                 dt4_over_4 * nay2, 0,                 dt3_over_2 * nay2,
             dt3_over_2 * nax2, 0,                 dt2 * nax2,        0,
             0,                 dt3_over_2 * nay2, 0,                 dt2 * nay2;

  // Actually run Kalman Filter prediction step, i.e. extrapolate state and
  // increase covariance dependent on delta time since last time
  ekf_.Predict();


  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  // Latest measurement (laser or radar):
  VectorXd z = measurement_pack.raw_measurements_;

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Radar updates, use non-linear update, and compute measurement/prediction
      // error using cartesian-to-polar equation rather than matrix multiplication
      Tools t;
      Hj_ = t.CalculateJacobian(ekf_.x_);
      ekf_.H_ = Hj_;      // Jacobian linear approx
      ekf_.R_ = R_radar_; // Radar measurement covariance
      ekf_.UpdateEKF(z);  // Use Extended version for correct measurement function h
  } else {
      // Laser updates, use 'normal' linear update
      ekf_.H_ = H_laser_; // Normal measurement matrix H
      ekf_.R_ = R_laser_; // Laser measurement covariance
      ekf_.Update(z);
  }

  // print the output (was in template code so haven't removed)
  cout << "Debug: after Update() x_ =" << endl << ekf_.x_ << endl;
  cout << "Debug: after Update() P_ =" << endl << ekf_.P_ << endl;
}
