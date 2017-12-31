////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : PID Control project
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////

#include <uWS/uWS.h>
#include <iostream>
#include "json.hpp"
#include "PID.h"
#include <math.h>
#include <cstdio>

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

double cte_sqd_sum = 0.0;
int num_cte_samples = 0;

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
std::string hasData(std::string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_last_of("]");
  if (found_null != std::string::npos) {
    return "";
  }
  else if (b1 != std::string::npos && b2 != std::string::npos) {
    return s.substr(b1, b2 - b1 + 1);
  }
  return "";
}

int main(int argc, char* argv[])
{
  uWS::Hub h;

  PID pid;
  // TODO: Initialize the pid variable.

  // CW: use command-line values if provided to save recompilation when experimenting
  // with new gain values.
  int good_conversions = 0;
  // Now updated to default to the values
  // found to be close to optimal in tuning (noting that I and D terms
  // computed allowing for real delta-time, not assumed dt=1).
  float k_p = 0.14, k_i = 2.0, k_d = 0.07;

  if (argc == 4) // program name, then P, I, D gain values if provided
  {
    good_conversions += sscanf(argv[1], "%g", &k_p);
    good_conversions += sscanf(argv[2], "%g", &k_i);
    good_conversions += sscanf(argv[3], "%g", &k_d);
  }
  if (good_conversions != 3)
  {
    cout << "Warning: no gains provided or conversion from string failed, using nominal defaults\n";
  }
  
  cout << "Using kP=" << k_p << " kI=" << k_i << " kD=" << k_d << endl;

  pid.Init(k_p, k_i, k_d);
  

  h.onMessage([&pid](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length, uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2')
    {
      auto s = hasData(std::string(data).substr(0, length));
      if (s != "") {
        auto j = json::parse(s);
        std::string event = j[0].get<std::string>();
        if (event == "telemetry") {
          // j[1] is the data JSON object
          double cte = std::stod(j[1]["cte"].get<std::string>());
          // Unused: double speed = std::stod(j[1]["speed"].get<std::string>());
          // Unused: double angle = std::stod(j[1]["steering_angle"].get<std::string>());
          double steer_value;
          /*
          * TODO: Calcuate steering value here, remember the steering value is
          * [-1, 1].
          * NOTE: Feel free to play around with the throttle and speed. Maybe use
          * another PID controller to control the speed!
          */

          // CW: update PID controller to get new steering command value
          pid.UpdateError(cte);
          steer_value = pid.TotalError();

          // CW: clip to allowable range (could do in controller class)
          if (steer_value < -1.0)
          {
            steer_value = -1.0;
          }
          else if (steer_value > 1.0)
          {
            steer_value = 1.0;
          }
          else
          {
            // No clipping required, in range already
          }

          // CW: keep track of filtered error to judge controller quality when
          // optimising gains
          double this_cte_sqd = cte * cte; // square to lose sign
          cte_sqd_sum += this_cte_sqd;
          num_cte_samples++;
          double avg_cte_sqd = cte_sqd_sum / num_cte_samples;

          // CW: output filtered CTE squared error to command line to allow
          // some quantitive measure of filter quality when tuning gains
          std::cout << "avg CTE^2: " << avg_cte_sqd << " CTE: " << cte << " Steering Value: " << steer_value << std::endl;

          json msgJson;
          msgJson["steering_angle"] = steer_value;
          msgJson["throttle"] = 0.3;
          auto msg = "42[\"steer\"," + msgJson.dump() + "]";
          //std::cout << msg << std::endl;
          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data, size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1)
    {
      res->end(s.data(), s.length());
    }
    else
    {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code, char *message, size_t length) {
    // CW: always find I have to comment out the close() call in these projects, else
    // it crashes with a core dump when the simulator disconnects
    //ws.close();
    std::cout << "Disconnected" << std::endl;

    // Reset accumulation of CTE error data
    cte_sqd_sum = 0.0;
    num_cte_samples = 0;
  });

  int port = 4567;
  if (h.listen(port))
  {
    std::cout << "Listening to port " << port << std::endl;
  }
  else
  {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
