#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  MatrixXd CalculateJacobian(const VectorXd& x_state);

  // A utility to normalise angles
  static float NormaliseAnglePlusMinusPi(float angle);

private:
  /**
  * Statically-allocated objects that we can safely return from our helper methods
  * for the client to copy (the tutorial code returned objects created on the stack
  * within methods, which should not be assumed to still exist once those functions
  * have exited). (Get reused, so not safe against reentrant or repeated calls.)
  */
	VectorXd retVector_;
	MatrixXd retMatrix_;

};

#endif /* TOOLS_H_ */
