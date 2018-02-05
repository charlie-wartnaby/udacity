////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : Model Predictive Control project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////


#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "MPC.h"
#include "json.hpp"

// for convenience
using json = nlohmann::json;
using namespace std::chrono;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Latency introduced into actuator commands back to simulator,
// must be 100 ms for the submitted project
const int actuator_delay_ms = 100;
// Computing latency measured too
int compute_delay_ms = 0;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.rfind("}]");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

// Evaluate a polynomial.
// CW: modified to compute derivatives, required to obtain steering
// angle from desired path polynomial
double polyeval(int derivative, Eigen::VectorXd coeffs, double x) {
  double result = 0.0;
  for (int i = 0; i < coeffs.size(); i++) {

    if (i < derivative) continue; // zero after differentiation

    // Multiply the coefficient up according to the derivative
    // of the power of this term
    int deriv_multiplier = 1;
    int power = i;
    for (int j = 0; j < derivative; j++)
    {
      deriv_multiplier *= power--;
    }

    result += deriv_multiplier * coeffs[i] * pow(x, i - derivative);
  }

  return result;
}

// Fit a polynomial.
// Adapted from
// https://github.com/JuliaMath/Polynomials.jl/blob/master/src/Polynomials.jl#L676-L716
Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals,
                        int order) {
  assert(xvals.size() == yvals.size());
  assert(order >= 1 && order <= xvals.size() - 1);
  Eigen::MatrixXd A(xvals.size(), order + 1);

  for (int i = 0; i < xvals.size(); i++) {
    A(i, 0) = 1.0;
  }

  for (int j = 0; j < xvals.size(); j++) {
    for (int i = 0; i < order; i++) {
      A(j, i + 1) = A(j, i) * xvals(j);
    }
  }

  auto Q = A.householderQr();
  auto result = Q.solve(yvals);
  return result;
}

int main() {
  uWS::Hub h;

  // MPC is initialized here!
  MPC mpc;

  h.onMessage([&mpc](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    string sdata = string(data).substr(0, length);
    cout << sdata << endl;
    if (sdata.size() > 2 && sdata[0] == '4' && sdata[1] == '2') {
      string s = hasData(sdata);
      if (s != "") {
        auto j = json::parse(s);
        string event = j[0].get<string>();
        if (event == "telemetry") {
          // Start measuring computing time to include in latency allowed for by MPC
          high_resolution_clock::time_point t_compute_start = high_resolution_clock::now();

          // j[1] is the data JSON object
          // CW: data descriptions taken from DATA.md supplied
          // Note: we initially get a vector of 6 waypoints. So presumably the simulator is clever enough to give
          // us just the next few points relevant to our desired trajectory, rather than points for the whole circuit.
          vector<double> ptsx = j[1]["ptsx"]; // The global x positions of the waypoints.
          vector<double> ptsy = j[1]["ptsy"]; // The global y positions of the waypoints. This corresponds to the z coordinate in Unity since y is the up-down direction.
          double px = j[1]["x"]; // The global x position of the vehicle.
          double py = j[1]["y"]; // The global y position of the vehicle.
          double psi = j[1]["psi"]; // The orientation of the vehicle in radians converted from the Unity format to the standard format expected in most mathemetical functions(more details below).
          double v = j[1]["speed"]; // The current velocity in mph.


          /*
          * TODO: Calculate steering angle and throttle using MPC.
          *
          * Both are in between [-1, 1].
          *
          */

          // Update current known state vector of vehicle in MPC controller
          mpc.SetState(px, py, psi, v);

          //Display the waypoints/reference line
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          // Convert waypoints from absolute to car coordinates; the simulator expects car
          // coordinates to display them correctly (as the yellow ideal line to follow)
          mpc.AbsoluteToCarCoords(ptsx, ptsy, next_x_vals, next_y_vals);

          // Fit a polynomial to the ideal waypoints, to give us a smooth reference
          // function we are aiming to follow with our controller. Once we have that,
          // we can calculate cross-track deviations from it at arbitrary x values
          // ahead of the car.
          mpc.CalculateDesiredPathPoly(next_x_vals, next_y_vals);

          // Explicitly accounting for expected latency in actuator commands taking effect:
          mpc.SetActuatorLatency((compute_delay_ms + actuator_delay_ms) / 1000.0);

          // Now we can actually run the MPC solver to find the optimal set of
          // actuator commands up until the time horizon to best fit the desired
          // trajectory starting from the current state
          double steer_value = 0;
          double throttle_value = 0.1;
          mpc.FindOptimalActuatorCommands(steer_value, throttle_value);

          json msgJson;

          // NOTE: Remember to divide by deg2rad(25) before you send the steering value back.
          // Otherwise the values will be in between [-deg2rad(25), deg2rad(25] instead of [-1, 1].
          // CW: but this conflicts with DATA.md, which describes steering_angle as "The current 
          // steering angle in radians." Remarkably, the MPC control works pretty well whether or not
          // I apply the factor to convert to [-25, 25 deg] range, but have left it in as suggested.
          // Also flipping the sign here as the model has positive delta (steering angle) pointing
          // anticlockwise, but simulator has positive steering command as turn to the right.
          double simulator_steer_value = steer_value / -deg2rad(25);

          msgJson["steering_angle"] = simulator_steer_value;
          msgJson["throttle"] = throttle_value; // The current throttle value [-1, 1].

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Green line

          bool show_desired_instead_of_mpc_predicted = false; // can set this for debug purposes
          if (show_desired_instead_of_mpc_predicted)
          {
            // Debug: setting green line to poly fit to waypoints for now to check fit looks OK,
            // used while debugging coordinate conversions and figuring out what order of polynomial
            // would give a decent fit
            vector<double> poly_x_vals;
            vector<double> poly_y_vals;
            for (size_t i = 0; i < next_x_vals.size(); i++)
            {
              poly_x_vals.push_back(next_x_vals[i]);
              poly_y_vals.push_back(polyeval(0, mpc.desired_path_poly_coeffs_, next_x_vals[i]));
            }
            msgJson["mpc_x"] = poly_x_vals;
            msgJson["mpc_y"] = poly_y_vals;
          }
          else
          {
            // Normal: show MPC predicted trajector as green line
            msgJson["mpc_x"] = mpc.mpc_pred_points_x_;
            msgJson["mpc_y"] = mpc.mpc_pred_points_y_;
          }

          //.. add (x,y) points to list here, points are in reference to the vehicle's coordinate system
          // the points in the simulator are connected by a Yellow line

          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;


          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          std::cout << msg << std::endl;

          // Finish measuring computing time to account for as additional latency next time
          high_resolution_clock::time_point t_compute_end = high_resolution_clock::now();
          auto duration = duration_cast<milliseconds>(t_compute_end - t_compute_start).count();
          compute_delay_ms = (int)duration;
          cout << "Computation elapsed time = " << compute_delay_ms << "ms\n";

          // Latency
          // The purpose is to mimic real driving conditions where
          // the car does actuate the commands instantly.
          //
          // Feel free to play around with this value but should be to drive
          // around the track with 100ms latency.
          //
          // NOTE: REMEMBER TO SET THIS TO 100 MILLISECONDS BEFORE
          // SUBMITTING.
          this_thread::sleep_for(chrono::milliseconds(actuator_delay_ms));
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    // CW: provided code did this, but caused segfault at least in Win10 bash environment;
    // with the close() call commented out, does not crash, and allows reconnection if
    // close simulator and reopen it:
    // ws.close();

    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
