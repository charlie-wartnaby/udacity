////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : unscented Kalman filter project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include "tools.h"
#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

    // CW: define sizes up front to make initialisations more self-documenting:

    n_x_ = 5; // CRTV model {px, py, v, psi, psi-dot}
    n_aug_ = n_x_ + 2; // Augment true states with linear and yaw acceleration process noise covariance elements
    n_sigma_ = 1 + 2 * n_aug_;

    // Template initialisations:

  // if this is false, laser measurements will be ignored
  use_laser_ = true;

  // if this is false, radar measurements will be ignored
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.5; // Guessing about a tenth of a g max (for accel at least, braking could be stronger though!), then experimented

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5; // From experiment, about optimal

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // CW: back to my own initialisation:

  is_initialized_ = false; // So we know to use first measurement to set up state

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);
  weights_ = VectorXd(n_sigma_);

  time_us_ = 0; // Set up on first iteration

  lambda_ = 3 - n_aug_; // Following guideline from lectures as starting-point, found pretty optimal by experiment

  // Compute sigma point weights, which don't require any run-time update
  // CW: it bothers me that the weight given to the first column -- which should be
  // closest to the mean we want -- is strongly negative! Though all the weights sum
  // to unity OK. Some discussion of that here:
  // https://math.stackexchange.com/questions/796331/scaling-factor-and-weights-in-unscented-transform-ukf
  // (This negative first weight was correct in that it matched the quiz solution in the
  // course however.)
  for (int i = 0; i < n_sigma_; i++)
  {
      double w;
      if (i <= 0)
      {
          w = lambda_ / (lambda_ + n_aug_); // Note: this one is negative!
      }
      else
      {
          w = 1.0 / (2.0 * (lambda_ + n_aug_));
      }
      weights_(i) = w;
  }
  //std::cerr << "Debug: weights=\n" << weights << std::endl; 

}

UKF::~UKF() {

}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
    if (((meas_package.sensor_type_ == meas_package.LASER) && !use_laser_) ||
        ((meas_package.sensor_type_ == meas_package.RADAR) && !use_radar_)   )
    {
        // Ignoring this type of measurement for comparison purposes, skip
    }
    else if (!is_initialized_
        || (meas_package.timestamp_ < time_us_)) // If simulator restarted
    {
        // First measurement, so use that to initialise state as best we can
        cerr << "Initialising filter for first use or restart" << endl;
        InitialiseState(meas_package);

        // done initializing, no need to predict or update
    }
    else
    {
        // Compute delta time since last measurement
        double delta_t_sec = (meas_package.timestamp_ - time_us_) * USEC_TO_SEC;
        time_us_ = meas_package.timestamp_; // Remember new timestamp for next time

        // Update state to current time by extrapolating CRTV trajectory,
        // increasing uncertainty through added process noise
        Prediction(delta_t_sec);

        // Then update state to improved estimate based on new measurement
        // point, reducing uncertainty
        double eta;  // Normalised Innovation Squared
        string meas_type;
        string additional_delimiter; // so laser and radar values in different tab-delimited columns
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
        {
            eta = UpdateRadar(meas_package);
            meas_type = "radar";
            additional_delimiter = "\t";
        }
        else
        {
            eta = UpdateLidar(meas_package);
            meas_type = "laser";
            additional_delimiter = "";
        }

        // Output a record so that if the program output is redirected to a file,
        // it can be conveniently imported e.g. into Excel for graphing/analysis
        cout << meas_type << "\t" << additional_delimiter << eta << endl;
    }

}

void UKF::InitialiseState(MeasurementPackage meas_package)
{
    double x = 0, y = 0;     // Position vector elements
    double px2 = 1, py2 = 1; // Measurement uncertainty in position vectors

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        // Expect distance (rho), angle anticlockwise from x-axis (phi)
        // and velocity away from us (rho-dot). The velocity doesn't help us much
        // as we don't know the yaw angle of the vehicle yet (though we could say
        // which half of the circle it is in depending on whether rho-dot is
        // +ve or -ve)

        assert(meas_package.raw_measurements_.size() == 3);
        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        x = rho * cos(phi);  // x forwards (to the right on simulator display)
        y = rho * sin(phi);  // y upwards

        // Probably overthinking it, but get approx initial covariance by combining
        // measurement uncertainties in radial and angle directions in some manner
        double px_radial = std_radr_ * cos(phi); // radial-noise uncertainty in x direction
        double py_radial = std_radr_ * sin(phi); // radial-noise uncertainty in y direction
        double px_angular = rho * std_radphi_ * sin(phi); // angle-noise uncertainty in x direction
        double py_angular = rho * std_radphi_ * cos(phi); // angle-noise uncertainy in y direction
        px2 = px_radial * px_radial + px_angular * px_angular; // Guess squaring sum of those will give
        py2 = py_radial * py_radial + py_angular * py_angular; //   fair init estimate   
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        // Expect x and y coordinates where x is forwards and y to the left
        // assuming same convention as tutorials
        assert(meas_package.raw_measurements_.size() == 2);
        x = meas_package.raw_measurements_[0];
        y = meas_package.raw_measurements_[1];

        // This time we directly have x- and y- measurement noise to get started with
        px2 = std_laspx_ * std_laspx_;
        py2 = std_laspy_ * std_laspy_;
    }
    else
    {
        cerr << "ERROR: unexpected sensor type during initialisation" << endl;
    }

    // However we obtained them, use the state elements we were able to derive
    // directly from the first measurement, and set the unknown ones to zero
    x_ << x, y,            // We have the position
          0.0, 0.0, 0.0;   // No information yet on speed, yaw or yaw accel

    // Also grab first timestamp so we can compute delta t henceforth
    time_us_ = meas_package.timestamp_;

    // State covariance
    P_ = MatrixXd::Identity(n_x_, n_x_); // but actually will overwrite diagonal elements

    // Initialise state covariance; seems reasonable to use the measurement uncertainty
    // in position we get from laser or radar as the position uncertainty, but for the
    // other elements put in a reasonably large number as we don't know anything about
    // the true state values, so approximate as half maxima we might get
    double max_speed = 10.0; // bicycle so 22 mph good going
    double max_yaw_accel = 1.0; // about pi radians in 3 sec
    P_(0, 0) = px2;
    P_(1, 1) = py2;
    P_(2, 2) = (max_speed / 2) * (max_speed / 2);         // No idea about speed so uncertainty about half max
    P_(3, 3) = (M_PI / 2) * (M_PI / 2);                   // Approx pi/2 uncertainty in yaw angle, i.e. no idea
    P_(4, 4) = (max_yaw_accel / 2) * (max_yaw_accel / 2); // No idea about yaw accel so half max as uncertainty

    is_initialized_ = true;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */


    //****************************************************************
    // Generating sigma points
    //****************************************************************
    // Based on my solution code from quiz "17. Augmentation Assignment 1"

    //create augmented mean vector
    VectorXd x_aug = VectorXd(7);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(7, 7);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state
    x_aug.head(n_x_) = x_;
    // CW: last two elements are nu(linear accel) and nu(yaw rate accel)
    x_aug(n_x_) = 0;
    x_aug(n_x_ + 1) = 0;

    //create augmented covariance matrix
    P_aug.fill(0.0);
    // Top-left portion is the ordinary state covariance:
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    // Bottom right is process noise covariance matrix Q, just 2 diagonal elements
    P_aug(n_x_,     n_x_)     = std_a_     * std_a_;
    P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd sqrt_P_aug = P_aug.llt().matrixL();

    //create augmented sigma points
    // CW: first col is just current state
    Xsig_aug.col(0) = x_aug;

    // CW: iterate over state dimensions for remaining points
    double root_lambda_plus_nx = sqrt(lambda_ + n_aug_); // CW: wasn't clear if 'n' here aug or not, but it was
    MatrixXd lpn_rootp = MatrixXd(n_x_ + 2, n_x_ + 2);
    lpn_rootp = root_lambda_plus_nx * sqrt_P_aug;
    // So for my future reference, each sigma point is the current mean
    // state with a projected sample error in just one dimension, where
    // the uncertainty in that dimension is specific to the state variable it
    // is (hence derived from the covariance matrix P_). With a +error and -error
    // we have two sigma points per state dimension, with the unmodified state
    // vector being the first column.
    for (int i = 0; i < n_aug_; i++)
    {
        Xsig_aug.col(i + 1)          = x_aug + lpn_rootp.col(i);
        Xsig_aug.col(i + n_aug_ + 1) = x_aug - lpn_rootp.col(i);
    }


    //****************************************************************
    // Predicting sigma points (extrapolating to current time)
    //****************************************************************
    // Based on my solution code to quiz "20. Sigma Point Prediction Assignment 1"

    double half_dt2 = 0.5 * delta_t * delta_t;
    for (int i = 0; i < 2 * n_aug_ + 1; i++)
    {
        VectorXd x = Xsig_aug.col(i);

        // Extract named variables for this point to aid understanding
        double v          = x(2); // Speed
        double psi        = x(3); // Yaw angle
        double psi_dot    = x(4); // Yaw rate of change
        double nu_a       = x(5); // Linear acceleration process noise added for this sample
        double nu_psi_dot = x(6); // Yaw rate acceleration process noise added for this sample

        // Compute some common terms to avoid repeated computation
        double cos_psi = cos(psi);
        double sin_psi = sin(psi);
        double v_cos_psi = v * cos_psi;
        double v_sin_psi = v * sin_psi;
        double psi_extrap = psi + psi_dot * delta_t;

        VectorXd x_new = x.head(5); // always add to existing state

        // Component terms common to div by zero or not cases:
        x_new(0) += half_dt2 * cos_psi * nu_a;                  // x
        x_new(1) += half_dt2 * sin_psi * nu_a;                  // y
        x_new(2) += delta_t  * nu_a;                            // v
        x_new(3) += delta_t  * psi_dot + half_dt2 * nu_psi_dot; // psi
        x_new(4) += delta_t  * nu_psi_dot;                      // psi-dot

        if (fabs(psi_dot) < EPSILON)
        {
            // Avoid division by zero
            // New x-y location just extrapolated in straight line as yaw rate at or near zero
            x_new(0) += v_cos_psi * delta_t;
            x_new(1) += v_sin_psi * delta_t;
        }
        else
        {
            // Finite yaw rate, so extrapolate x-y position using integration along
            // circular arc using formulae derived in the lectures
            double v_by_psi_dot = v / psi_dot;
            x_new(0) += v_by_psi_dot * ( sin(psi_extrap) - sin(psi));
            x_new(1) += v_by_psi_dot * (-cos(psi_extrap) + cos(psi));
        }

        //write predicted sigma points into right column
        Xsig_pred_.col(i) = x_new;

    } // Loop over all sigma points


    //****************************************************************
    // Recover state mean and covariance from projected sigma points
    //****************************************************************
    // Based on my solution code to quiz "22. Predicted Mean and Covariance Assignment 1"
    // Note that weights are precalculated now though, as they require no run-time update

    //predict state mean
    x_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++)
    {
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predict state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++)   //iterate over sigma points
    {

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        // Angle normalization to avoid misleading large angle differences 
        // introduced by wraparound
        x_diff(3) = Tools::NormaliseAnglePlusMinusPi(x_diff(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
double UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

    VectorXd z = meas_package.raw_measurements_; // x, y coords

    //****************************************************************
    // Compute predicted state and covariance in laser measurement space
    //****************************************************************
    // CW: adapted from my quiz solution code for "26. Predict Radar Measurement Assignment 1"

    //set measurement dimension, just x and y coords for laser
    n_z_ = 2;

    // Common laser/radar matrix initialisation dependent on that dimension
    SizeUpdateMatrices();

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++)
    {
        double px = Xsig_pred_(0, i);
        double py = Xsig_pred_(1, i);

        Zsig_(0, i) = px;
        Zsig_(1, i) = py;
    }

    // Mean measurement calculation is common to laser and radar
    CalcMeanMeasurementAndCovariance();

    // Add measurement noise on leading diagonal (each component of measurement
    // noise considered uncorrelated to others, so no off-diagonal terms here)
    S_(0, 0) += std_laspx_ * std_laspx_;
    S_(1, 1) += std_laspy_ * std_laspy_;


    // Rest is common to laser and radar update
    double eta = GenericUpdate(z, false); // No angle normalisation required for laser case

    return eta;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
double UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

    VectorXd z = meas_package.raw_measurements_; // rho, phi, rho-dot

  //****************************************************************
  // Compute predicted state and covariance in radar measurement space
  //****************************************************************
  // CW: based on my quiz solution code for "26. Predict Radar Measurement Assignment 1"
  // originally, though now refactored to pull out common laser/radar code

    //set measurement dimension, radar can measure r, phi, and r_dot
    n_z_ = 3;

    // Common laser/radar matrix initialisation dependent on that dimension
    SizeUpdateMatrices();

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++)
    {
        double px  = Xsig_pred_(0, i);
        double py  = Xsig_pred_(1, i);
        double v   = Xsig_pred_(2, i);
        double psi = Xsig_pred_(3, i);
        // Don't need yaw rate of change

        double rho2 = px * px + py * py;
        double rho = fabs(rho2) > EPSILON ? sqrt(rho2) : EPSILON;

        double phi = atan2(py, px);

        double rho_dot = ((px * cos(psi) + py * sin(psi)) * v) / rho;

        Zsig_(0, i) = rho;
        Zsig_(1, i) = phi;
        Zsig_(2, i) = rho_dot;
    }

    // Mean measurement calculation is common to laser and radar
    CalcMeanMeasurementAndCovariance();

    // Add measurement noise on leading diagonal (each component of measurement
    // noise considered uncorrelated to others, so no off-diagonal terms here)
    S_(0, 0) += std_radr_   * std_radr_;
    S_(1, 1) += std_radphi_ * std_radphi_;
    S_(2, 2) += std_radrd_  * std_radrd_;


    // Rest is common to laser and radar update
    double eta = GenericUpdate(z, true); // Need angle normalisation for radar case

    return eta;
}

void UKF::SizeUpdateMatrices()
{
    // CW: based on quiz code for "26. Predict Radar Measurement Assignment 1"

    //create matrix for predicted sigma points in measurement space
    Zsig_ = MatrixXd(n_z_, n_sigma_);

    //mean predicted measurement in measurement space
    z_pred_ = VectorXd(n_z_);

    //measurement covariance matrix S_
    S_ = MatrixXd(n_z_, n_z_);


    // Based on quiz code for "29. UKF Update Assignment 1"

    //create matrix for cross correlation Tc
    Tc_ = MatrixXd(n_x_, n_z_);

}

void UKF::CalcMeanMeasurementAndCovariance()
{
    //calculate mean predicted measurement
    z_pred_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++)
    {
        z_pred_ += weights_(i) * Zsig_.col(i);
    }

    // Calculate measurement covariance matrix S_
    // Start with prediction of covariance due to process noise, i.e.
    // uncertainty in extrapolating state to new time point
    S_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++)
    {
        MatrixXd z_diff = z_pred_ - Zsig_.col(i);
        MatrixXd z_diffT = z_diff.transpose();
        S_ += weights_(i) * z_diff * z_diffT;
    }
}

double UKF::GenericUpdate(VectorXd& z, bool radarAngleNormReqd)
{
    //****************************************************************
    // Kalman update for measurement 
    //****************************************************************
    // Based on my solution code for quiz "29. UKF Update Assignment 1"

    MatrixXd z_diff; // either for one sigma point, or for measurement vs expected

    //calculate cross correlation matrix
    Tc_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++)
    {
        MatrixXd X_diff = Xsig_pred_.col(i).head(n_x_) - x_; // Discarding elements in process noise dimensions at bottom of matrix

        if (radarAngleNormReqd)
        {
            // Angle normalization in case of misleading large angle differences
            // introduced by wraparound, here of steering angle psi
            X_diff(3) = Tools::NormaliseAnglePlusMinusPi(X_diff(3));
        }

        z_diff = Zsig_.col(i) - z_pred_;

        if (radarAngleNormReqd)
        {
            // Further angle normalisation, of radar rho
            z_diff(1) = Tools::NormaliseAnglePlusMinusPi(z_diff(1));
        }

        Tc_ += weights_(i) * X_diff * z_diff.transpose();
    }

    //calculate Kalman gain K;
    MatrixXd S_inv = S_.inverse();
    MatrixXd K = Tc_ * S_inv;

    //update state mean and covariance matrix
    z_diff = z - z_pred_;

    if (radarAngleNormReqd)
    {
        // Angle normalisation of radar rho again
        z_diff(1) = Tools::NormaliseAnglePlusMinusPi(z_diff(1));
    }

    // Actual state update based on this measurement
    x_ += K * z_diff;

    // Actual state covariance update based on this measurement
    MatrixXd K_t = K.transpose();
    P_ -= K * S_ * K_t;

    //****************************************************************
    // Calculation of Normalised Innovation Squared (NIS) (eta)
    //****************************************************************
    // This is a measure of how well the new measurement fits within
    // the n-D elliptical expected covariance uncertainty,
    // which following chi^2 distribution should be
    // mostly between 0.35 and 7.82 (95%... 5% interval) with
    // 3 degrees of freedom (radar), or 0.10 to 5.99 with
    // 2 degrees of freedom (laser).
    // Discussion and formula for this starting around 4 mins
    // in video at "31. Parameters and Consistency"

    MatrixXd eta_matrix = z_diff.transpose() * S_.inverse() * z_diff;

    return eta_matrix(0,0);
}
