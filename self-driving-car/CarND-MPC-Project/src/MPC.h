#ifndef MPC_H
#define MPC_H

#include <vector>
#include "Eigen-3.3/Eigen/Core"

using namespace std;

// Prototypes for provided utility functions in main.cpp
extern Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order);
extern double polyeval(int derivative, Eigen::VectorXd coeffs, double x);

class MPC {
 public:
  MPC();

  virtual ~MPC();

  // Solve the model given an initial state and polynomial coefficients.
  // Return the first actuatotions.
  vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs);

  // State variables in absolute coordinates
  double px_;
  double py_;
  double psi_;
  double v_;

  // Expected actuator latency to cope with
  double actuator_latency_s_;
  void SetActuatorLatency(double delay_sec);
  int latency_timesteps_;

  // Previous actuator commands, which will apply for the first part of our
  // future prediction
  double throttle_cmd_prev_;
  double steering_cmd_prev_;

  // Polynomial fitted to desired waypoints
  Eigen::VectorXd desired_path_poly_coeffs_;

  vector<double> mpc_pred_points_x_;
  vector<double> mpc_pred_points_y_;

  void SetState(double px, double py, double psi, double v);
  void AbsoluteToCarCoords(vector<double> absX, vector<double> absY, vector<double>& carX, vector<double>& carY);
  void CalculateDesiredPathPoly(vector<double> desiredCarX, vector<double> desiredCarY);
  void FindOptimalActuatorCommands(double& steer_value, double& throttle_value);
};

#endif /* MPC_H */
