////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : unscented Kalman filter project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Some constants (not class members to avoid faff of separate definition for non-int const members in this compiler version)
#define  EPSILON     (1e-10) // Magnitude below which something effectively zero
#define  USEC_TO_SEC (1e-6)  // Microsecond to second conversion factor

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  bool use_radar_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  double std_radrd_ ;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* State dimension
  int n_x_;

  ///* Augmented state dimension
  int n_aug_;

  ///* Sigma point spreading parameter
  double lambda_;


  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Prediction(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   * @return NIS eta value for new measurement
   */
  double UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   * @return NIS eta value for new measurement
   */
  double UpdateRadar(MeasurementPackage meas_package);

private:
    // CW: additional members added by me, to help factor out common
    // code in particular

    // Total number of sigma points for mean and +/- values in each augmented dimension
    int n_sigma_;

    void InitialiseState(MeasurementPackage meas_package);

    // Number of dimensions in measurement space during update step
    int n_z_;

    // Predicted sigma points in measurement space
    MatrixXd Zsig_;

    // Mean predicted measurement in measurement space
    VectorXd z_pred_;

    // Measurement covariance matrix S
    MatrixXd S_;

    // Cross correlation matrix Tc
    MatrixXd Tc_;

    // Common laser/radar method to size update step matrices
    void SizeUpdateMatrices();

    // Method to implement common laser/radar measurement prediction steps
    void CalcMeanMeasurementAndCovariance();

    // Method to implement common laser/radar final update steps
    // @param  z                   new measurement value
    // @param  radarAngleNormReqd  whether wraparound angle normalisation required
    // @return                     NIS eta value for new measurement
    double GenericUpdate(VectorXd& z, bool radarAngleNormReqd);
};

#endif /* UKF_H */
