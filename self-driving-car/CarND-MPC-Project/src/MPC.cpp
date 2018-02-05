////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : Model Predictive Control project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////


#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include <cppad/utility/poly.hpp> // For evaluation of polynomial in CppAD types
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// TODO: Set the timestep length and duration
// CW: plus any other hyperparameters I've added
size_t N = 4;             // Number of forward timesteps to compute; solver fit much more robust using few points.
double dt = 0.25;          // Delta time (sec) between timesteps
const int poly_order = 3; // Order of polynomial used to fit desired forward path; cubic recommended in lectures, even 2 gives decent fit

double weight_cte = 5.0;
double weight_epsi = 0.5;
double weight_v = 0.5;
double weight_delta = 0; // why avoid steering at all?
double weight_a = 0.5; // 
double weight_delta_diff = 0.2;
double weight_a_diff = 0.2;


double latency_sec = 0; // set during run time

const size_t num_state_vars = 4;
const size_t num_actuator_vars = 2;

const int state_x_idx = 0;
// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;


// Both the reference cross track and orientation errors are 0.
// The reference velocity is set to 40 mph.
double ref_v = 40;

// The solver takes all the state variables and actuator
// variables in a singular vector. Thus, we should to establish
// when one variable starts and another ends to make our lifes easier.
size_t x_start = 0;
size_t y_start = x_start + N;
size_t psi_start = y_start + N;
size_t v_start = psi_start + N;
size_t cte_start = v_start + N;
size_t epsi_start = cte_start + N;
size_t delta_start = epsi_start + N;
size_t a_start = delta_start + N - 1;


class FG_eval {
 public:
  // Fitted polynomial coefficients
  vector<AD<double>> coeffs_;
  FG_eval(Eigen::VectorXd coeffs)
  {
    coeffs_.clear();
    for (int i = 0; i < coeffs.size(); i++)
    {
      coeffs_.push_back(coeffs[i]);
    }
  }


  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& fg, const ADvector& vars) {
    // TODO: implement MPC
    // `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
    // NOTE: You'll probably go back and forth between this function and
    // the Solver function below.

    size_t t_idx; // timestep index

    // The cost is stored is the first element of `fg`.
    // Any additions to the cost should be added to `fg[0]`.
    fg[0] = 0;

    // The part of the cost based on the reference state.
    // CW: starting at t=1, because t=0 is fixed by 'old' actuator
    // commands and existing state, which we are powerless to change
    for (t_idx = 1; t_idx < N; t_idx++) {
      fg[0] += weight_cte * CppAD::pow(vars[cte_start + t_idx], 2);
      fg[0] += weight_epsi * CppAD::pow(vars[epsi_start + t_idx], 2);
      fg[0] += weight_v * CppAD::pow(vars[v_start + t_idx] - ref_v, 2);
    }

    // Minimize the use of actuators.
    for (t_idx = 1; t_idx < N - 1; t_idx++) {
      fg[0] += weight_delta * CppAD::pow(vars[delta_start + t_idx], 2);
      fg[0] += weight_a * CppAD::pow(vars[a_start + t_idx], 2);
    }

    // Minimize the value gap between sequential actuations.
    for (t_idx = 1; t_idx < N - 2; t_idx++) {
      fg[0] += weight_delta_diff * CppAD::pow(vars[delta_start + t_idx + 1] - vars[delta_start + t_idx], 2);
      fg[0] += weight_a_diff * CppAD::pow(vars[a_start + t_idx + 1] - vars[a_start + t_idx], 2);
    }

    //
    // Setup Constraints
    //
    // NOTE: In this section you'll setup the model constraints.

    // Initial constraints
    //
    // We add 1 to each of the starting indices due to cost being located at
    // index 0 of `fg`.
    // This bumps up the position of all the other values.
    fg[1 + x_start] = vars[x_start];
    fg[1 + y_start] = vars[y_start];
    fg[1 + psi_start] = vars[psi_start];
    fg[1 + v_start] = vars[v_start];
    fg[1 + cte_start] = vars[cte_start];
    fg[1 + epsi_start] = vars[epsi_start];

    // The rest of the constraints
    for (t_idx = 1; t_idx < N; t_idx++) {

      // For first iteration, the timestep is the actuator latency; while this
      // time passes, we are stuck with our command values from last time. After
      // that we will be iterating at our planned delta time and choosing our
      // own new actuator commands.
      double this_dt = (t_idx == 1 ? latency_sec : dt);

      // The state at time t + dt .
      AD<double> x1 = vars[x_start + t_idx];
      AD<double> y1 = vars[y_start + t_idx];
      AD<double> psi1 = vars[psi_start + t_idx];
      AD<double> v1 = vars[v_start + t_idx];
      AD<double> cte1 = vars[cte_start + t_idx];
      AD<double> epsi1 = vars[epsi_start + t_idx];

      // The state at time t.
      AD<double> x0 = vars[x_start + t_idx - 1];
      AD<double> y0 = vars[y_start + t_idx - 1];
      AD<double> psi0 = vars[psi_start + t_idx - 1];
      AD<double> v0 = vars[v_start + t_idx - 1];
      AD<double> cte0 = vars[cte_start + t_idx - 1];
      AD<double> epsi0 = vars[epsi_start + t_idx - 1];

      // Only consider the actuation at time t.
      AD<double> delta0 = vars[delta_start + t_idx - 1];
      AD<double> a0 = vars[a_start + t_idx - 1];

      AD<double> y_approx_desired0 = CppAD::Poly(0, coeffs_, x0); // evaluate waypoint polynomial at x0, was 'f' in quiz code
      AD<double> dy_dx_desired0    = CppAD::Poly(1, coeffs_, x0); // first derivative of waypoint polynomial at x0
      AD<double> psi_desired0 = CppAD::atan(dy_dx_desired0);

      // Here's `x` to get you started.
      // The idea here is to constraint this value to be 0.
      //
      // Recall the equations for the model:
      // x_[t+1] = x[t] + v[t] * cos(psi[t]) * dt
      // y_[t+1] = y[t] + v[t] * sin(psi[t]) * dt
      // psi_[t+1] = psi[t] + v[t] / Lf * delta[t] * dt
      // v_[t+1] = v[t] + a[t] * dt
      // cte[t+1] = f(x[t]) - y[t] + v[t] * sin(epsi[t]) * dt
      // epsi[t+1] = psi[t] - psi_desired0[t] + v[t] * delta[t] / Lf * dt

      // CW: based on quiz code but broken down into steps so I can understand it better.
      // First, calculating what new state we expect at t=i+1 based on t=i state:
      AD<double> delta_psi       = v0 * delta0 / Lf * this_dt;
      AD<double> x1_predicted    = x0 + v0 * CppAD::cos(psi0) * this_dt;
      AD<double> y1_desired      = CppAD::Poly(0, coeffs_, x1_predicted);
      AD<double> dy_dx_desired1  = CppAD::Poly(1, coeffs_, x1_predicted);
      AD<double> psi_desired1    = CppAD::atan(dy_dx_desired1);
      AD<double> y1_predicted    = y0 + v0 * CppAD::sin(psi0) * this_dt; // CW: but shouldn't we allow for steering angle and integrate properly round circular arc?
      AD<double> psi1_predicted  = psi0 + delta_psi;
      AD<double> v1_predicted    = v0 + a0 * this_dt;
      AD<double> cte1_predicted  = y1_predicted - y1_desired;  // Have done my own thing here rather than extrapolating old CTE as if it were independent variable
      AD<double> epsi1_predicted = psi1_predicted - psi_desired1; // Have used my own evaluation here, not linear extrapolation of old epsi suggested

      // We then want the optimiser to get the difference between our model predicted
      // states and the values it comes out with to be all zeroes as constraints, so the
      // optimiser is forced to use our prediction of future state values:
      fg[1 + x_start    + t_idx] = x1    - x1_predicted;
      fg[1 + y_start    + t_idx] = y1    - y1_predicted;
      fg[1 + psi_start  + t_idx] = psi1  - psi1_predicted;
      fg[1 + v_start    + t_idx] = v1    - v1_predicted;
      fg[1 + cte_start  + t_idx] = cte1  - cte1_predicted;
      fg[1 + epsi_start + t_idx] = epsi1 - epsi1_predicted;
    }
  }
};

//
// MPC class definition implementation.
//
MPC::MPC()
{
  throttle_cmd_prev_ = 0;
  steering_cmd_prev_ = 0;
}

MPC::~MPC() {}


void MPC::SetActuatorLatency(double delay_sec)
{
  // Rather than just treat the delay with which actuator commands are sent
  // back to the simulator as an uncontrolled perturbation, it will actually
  // be considered in the MPC prediction. So we calculate how many timesteps
  // it takes before a command to the actuators (throttle & steering) to
  // get through to the simulator.

  // Written to global so that it's accessible to FG_eval() class
  latency_sec = delay_sec;
}



void MPC::SetState(double px, double py, double psi, double v)
{
  px_  = px;
  py_  = py;
  psi_ = psi;
  v_   = v;
}

void MPC::AbsoluteToCarCoords(vector<double> absX, vector<double> absY, vector<double>& carX, vector<double>& carY)
{
  size_t n_points = absX.size();
  if (n_points != absY.size())
  {
    cerr << "ERROR: mismatched coordinate vector sizes in MPC::AbsoluteToCarCoords()\n";
    return;
  }

  carX.clear();
  carY.clear();

  // Use homogenous transform to convert absolute coordinates to car coordinates
  for (size_t i = 0; i < n_points; i++)
  {
    double px = absX[i];
    double py = absY[i];

    // Translate to same origin as car coordinates
    double offset_px = px - px_;
    double offset_py = py - py_;

    double cos_psi = cos(psi_);
    double sin_psi = sin(psi_);

    // Rotate coordinates into car's rotated frame of reference (transpose
    // of matrix to do the opposite in particle filter project, where
    // we were going from sensor readings in car coords to absolute map coords)
    double rot_px = offset_px *  cos_psi + offset_py * sin_psi;
    double rot_py = offset_px * -sin_psi + offset_py * cos_psi;

    // Build up vectors of transformed points now in car coords
    carX.push_back(rot_px);
    carY.push_back(rot_py);
  }
}

void MPC::CalculateDesiredPathPoly(vector<double> desiredCarX, vector<double> desiredCarY)
{
  Eigen::VectorXd desired_x_eigen_format = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(desiredCarX.data(), desiredCarX.size());
  Eigen::VectorXd desired_y_eigen_format = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(desiredCarY.data(), desiredCarY.size());

  desired_path_poly_coeffs_ = polyfit(desired_x_eigen_format, desired_y_eigen_format, poly_order);
}

void MPC::FindOptimalActuatorCommands(double& steer_value, double& throttle_value)
{
  // Current state in car coordinates, in same form as used by quiz
  // code
  Eigen::VectorXd state(6);

  // As we are car-relative, initial coordinates and yaw angle are always
  // zero
  state[0] = 0;
  state[1] = 0;
  state[2] = 0;

  // But we aren't using car-relative velocity:
  state[3] = v_;

  // CW why are the error terms considered state anyway?

  // Initial cross-track error is just constant term from fit to desired
  // waypoint polynomial, as that is in car-relative coordinates and
  // so we are at x=0 and thus all higher terms disappear:
  state[4] = -desired_path_poly_coeffs_[0]; // diff from ideal -ve though gets squared in cost func anyway

  // Similarly initial steering error is just related to the gradient of
  // the polynomial at x=0; if we had no steering error, the poly
  // curve would be flat at x=0.
  // Note: now evaluating derivative of polynomial of arbitrary order,
  // quiz code just used first coeff assuming 1st-order poly.
  double dy_by_dx_at_0 = polyeval(1, desired_path_poly_coeffs_, 0.0); // '1' for first derivative
  state[5] = -atan(dy_by_dx_at_0); // diff from ideal -ve though gets squared in cost func anyway

  // Actually run solver to find optimum set of actuator commands to try and get
  // vehicle trajectory to match desired path polynomial well
  auto vars = Solve(state, desired_path_poly_coeffs_);

  // Keep just the actuator commands we wanted
  steer_value = vars[0];
  throttle_value = vars[1];
}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
  bool ok = true;
  size_t i;
  typedef CPPAD_TESTVECTOR(double) Dvector;

  // Quiz code
  double x = state[0];
  double y = state[1];
  double psi = state[2];
  double v = state[3];
  double cte = state[4];
  double epsi = state[5];


  // TODO: Set the number of model variables (includes both states and inputs).
  // For example: If the state is a 4 element vector, the actuators is a 2
  // element vector and there are 10 timesteps. The number of variables is:
  //
  // 4 * 10 + 2 * 9
  //
  // CW: in that example have 10 instances of the state because it is relevant
  // at t=0 (starting-point) and the final state (included in cost computation).
  // So the assumption here in the template code is that we are computing time
  // across N timesteps including the zeroth one, so the total _time_ covered
  // is actually (N-1)*dt not N*dt.
  // Hence only 9 actuator commands because the final ones we would set have
  // no effect until what would be timestep beyond our horizon.
// TODO  size_t n_vars = (num_state_vars * N) + (num_actuator_vars * (N - 1));

  // TODO: Set the number of constraints
  // CW: only the actuator commands are the free variables to manipulate.
  // They both have range of [-1,1] here, though the steering angle is
  // rescaled  before being passed to the simulator.
// TODO  size_t n_constraints = num_actuator_vars;

// Quiz code
// N timesteps == N - 1 actuations
  size_t n_vars = N * 6 + (N - 1) * 2;
  // Number of constraints
  size_t n_constraints =  N * 6;


  // Initial value of the independent variables.
  // SHOULD BE 0 besides initial state.
  Dvector vars(n_vars);
  for (i = 0; i < n_vars; i++) {
    vars[i] = 0;
  }

  // Quiz code
  // Set the initial variable values
  vars[x_start] = x;
  vars[y_start] = y;
  vars[psi_start] = psi;
  vars[v_start] = v;
  vars[cte_start] = cte;
  vars[epsi_start] = epsi;

  // CW: start the first iteration at t=0, but using the old acutator command
  // values; they will still be in effect from last time until our predicted
  // latency time has elapsed
  vars[delta_start] = steering_cmd_prev_;
  vars[a_start]     = throttle_cmd_prev_;

  Dvector vars_lowerbound(n_vars);
  Dvector vars_upperbound(n_vars);
  // TODO: Set lower and upper limits for variables.


  // Quiz code
  // Set all non-actuators upper and lowerlimits
  // to the max negative and positive values.
  for (i = 0; i < delta_start; i++) {
    vars_lowerbound[i] = -1.0e19;
    vars_upperbound[i] = 1.0e19;
  }

  vars_lowerbound[delta_start] = vars_upperbound[delta_start] = steering_cmd_prev_;
  vars_lowerbound[a_start]     = vars_upperbound[a_start]     = throttle_cmd_prev_;

  // The upper and lower limits of delta are set to -25 and 25
  // degrees (values in radians).
  // NOTE: Feel free to change this to something else.
  for (i = delta_start + 1; i < a_start; i++) {
    vars_lowerbound[i] = -0.436332;
    vars_upperbound[i] = 0.436332;
  }

  // Acceleration/decceleration upper and lower limits.
  // NOTE: Feel free to change this to something else.
  for (i = a_start + 1; i < n_vars; i++) {
    vars_lowerbound[i] = -1.0;
    vars_upperbound[i] = 1.0;
  }



  // Lower and upper limits for the constraints
  // Should be 0 besides initial state.
  // CW: see my comments below for why. These constraints are on the
  // elements of fg[], not vars[] (as they have their own lower and
  // upper bounds). Constraints on fg[] seem to force the model solver
  // to fit the state extrapolation we compute using the vehicle model.
  Dvector constraints_lowerbound(n_constraints);
  Dvector constraints_upperbound(n_constraints);

  // CW: during solving, each constraint in fg[] after the
  // t=0 value is set to the *difference* between the value that it
  // should be (from iterating the vehicle model) and what the
  // var[] value is (I guess explored by the solver somehow). Hence the
  // solver must achieve zero for those differences to correctly
  // fit the vehicle model extrapolation of states; and that's why
  // the constraints are set to zero.
  for (i = 0; i < n_constraints; i++) {
    constraints_lowerbound[i] = 0;
    constraints_upperbound[i] = 0;
  }

  // CW: but for the initial state, the values are fixed at what we
  // know the t=0 state to be, hence the constraints here are set to
  // their values rather than zero. (Maybe this could have been made
  // more systematic by setting the fg[] terms for t=0 also to the 
  // difference between the required t=0 states and the corresponding
  // var[] elements so they would also need to be constrained to zero,
  // but following what the 'Mind the Line' quiz did here.)
  constraints_lowerbound[x_start] = x;
  constraints_lowerbound[y_start] = y;
  constraints_lowerbound[psi_start] = psi;
  constraints_lowerbound[v_start] = v;
  constraints_lowerbound[cte_start] = cte;
  constraints_lowerbound[epsi_start] = epsi;

  constraints_upperbound[x_start] = x;
  constraints_upperbound[y_start] = y;
  constraints_upperbound[psi_start] = psi;
  constraints_upperbound[v_start] = v;
  constraints_upperbound[cte_start] = cte;
  constraints_upperbound[epsi_start] = epsi;

  // object that computes objective and constraints
  FG_eval fg_eval(coeffs);

  //
  // NOTE: You don't have to worry about these options
  //
  // options for IPOPT solver
  std::string options;
  // Uncomment this if you'd like more print information
  options += "Integer print_level  0\n";
  // NOTE: Setting sparse to true allows the solver to take advantage
  // of sparse routines, this makes the computation MUCH FASTER. If you
  // can uncomment 1 of these and see if it makes a difference or not but
  // if you uncomment both the computation time should go up in orders of
  // magnitude.
  options += "Sparse  true        forward\n";
  options += "Sparse  true        reverse\n";
  // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
  // Change this as you see fit.
  // CW TODO: the deliberate delay introduced in main.cpp before returning
  // actuator commands to the simulator should include this computation time,
  // rather than just being added to it arbitrarily.
  options += "Numeric max_cpu_time          0.5\n";

  // place to return solution
  CppAD::ipopt::solve_result<Dvector> solution;

  // solve the problem
  CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

  // Check some of the solution values
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // Cost
  auto cost = solution.obj_value;
  std::cout << "Cost=" << cost << " solved ok=" << ok << std::endl;

  // Save the MPC-predicted path points for display
  mpc_pred_points_x_.clear();
  mpc_pred_points_y_.clear();
  for (i = 0; i < N; i++)
  {
    mpc_pred_points_x_.push_back(solution.x[x_start + i]);
    mpc_pred_points_y_.push_back(solution.x[y_start + i]);
  }

  // TODO: Return the first actuator values. The variables can be accessed with
  // `solution.x[i]`.
  //
  // {...} is shorthand for creating a vector, so auto x1 = {1.0,2.0}
  // creates a 2 element double vector.

  // CW note: I am taking the dt=1 command values, because the t=0 values are the
  // 'old' ones from last time. The first actuator commands we can apply will take
  // effect only after the expected latency, which are the t+(latency t) values.
  double new_steering_cmd = solution.x[delta_start + 1];
  double new_throttle_cmd = solution.x[a_start + 1];
  steering_cmd_prev_ = new_steering_cmd;
  throttle_cmd_prev_ = new_throttle_cmd;

  return { new_steering_cmd, new_throttle_cmd };
}
