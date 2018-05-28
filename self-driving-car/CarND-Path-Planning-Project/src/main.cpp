////////////////////////////////////////////////////////////////////////////////
// 
//    Udacity self-driving car course : Path Planning project.
//
//    Author : Charlie Wartnaby, Applus IDIADA
//    Email  : charlie.wartnaby@idiada.com
//
////////////////////////////////////////////////////////////////////////////////


#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
//#include "Eigen-3.3/Eigen/Core"   // Did not use this library in the end
//#include "Eigen-3.3/Eigen/QR"     // Did not use this library in the end
#include "json.hpp"

#include <chrono>   // Required for run-time adaptation to calling rate (depends on simulator quality)
#include "spline.h" // From http://kluge.in-chemnitz.de/opensource/spline/

using namespace std;
using namespace std::chrono;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

void check_for_greatest_s(double s, double& max_s)
{
  if (s > max_s)
  {
    max_s = s;
  }
}

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }


  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy]
                  (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // Static variables
    static bool   first_time         = true;
    static double current_speed_m_s  = 0.1; // Slightly nonzero start avoids trying to fit spline to set of identical points
    static double current_accel_m_s2 = 0.0;
    static int desired_lane_idx      = 1; // start in middle lane of 3 (leftmost is 0)
    static double max_s              = 1000.0; // Max Frenet distance seen, to account for wraparound

    static steady_clock::time_point prev_time             = steady_clock::now(); // To measure real-time calling period
    static steady_clock::time_point time_last_lane_change = steady_clock::now(); // To avoid over-frequent lane changes
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Compute real-time calling period and time since last lane change.
    // The frequency with which we are called varies quite a lot with simulator mode;
    // fast when it is in low-resolution, low-quality mode, slow when it is in
    // high-resolution, high-quality mode. We need the real period to regulate
    // speed changes appropriately later.
    steady_clock::time_point time_now = steady_clock::now();
    double delta_t_program = first_time ? 0.05 : chrono::duration_cast<chrono::microseconds>(time_now - prev_time).count() / 1e6;
    double time_since_last_lane_change = chrono::duration_cast<chrono::microseconds>(time_now - time_last_lane_change).count() / 1e6;
    prev_time = time_now;

    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
            
          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;


          	// TODO: define a path made up of (x,y) points that the car will visit sequentially every .02 seconds


            ////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Start by reusing the unconsumed path points we submitted to the simulator last time, that have been
            // handed back to us, following the method in the walkthrough Q&A video
            int prev_size = previous_path_x.size();

            vector<double> ptsx, ptsy;

            // "Ref" here is origin for car-local coordinates. Using ego car relative coordinates (x-axis extending
            // out from front of car) ensures that the immediate path (over tens of metres say) is always
            // single-valued, which is essential for making and using a spline fit from. In global coords,
            // the track would become multi-valued at given x when it happens to lie northwards or southwards
            // for example, so no sensible fit of y(x) could be made.
            double ref_x   = car_x;
            double ref_y   = car_y;
            double ref_yaw = deg2rad(car_yaw);

            double our_projected_s; // Frenet s-coordinate of last waypoint, or current position if we have none

            if (prev_size < 2)
            {
              // To start with use current state as reference
              double distance_back_from_current = 1.0; // implicit in tutor code
              double prev_car_x = car_x - distance_back_from_current * cos(ref_yaw); // bug in video, used car_yaw, but that in deg?
              double prev_car_y = car_y - distance_back_from_current * sin(ref_yaw);

              ptsx.push_back(prev_car_x);
              ptsy.push_back(prev_car_y);
              ptsx.push_back(car_x);
              ptsy.push_back(car_y);
              // Now have first 2 points in global coords being where we were some distance ago, and 
              // where we are now

              our_projected_s = car_s;
            }
            else
            {
              // Use last two unconsumed path points rather than car current state; we will
              // append to that previously planned path to avoid discontinuities
              ref_x = previous_path_x[prev_size - 1];
              ref_y = previous_path_y[prev_size - 1];
              double ref_x_prev = previous_path_x[prev_size - 2];
              double ref_y_prev = previous_path_y[prev_size - 2];
              ref_yaw = atan2((ref_y - ref_y_prev), (ref_x - ref_x_prev));

              ptsx.push_back(ref_x_prev);
              ptsy.push_back(ref_y_prev);
              ptsx.push_back(ref_x);
              ptsy.push_back(ref_y);

              our_projected_s = end_path_s;
            }


            // Some parameters for the highway we're on
            double lane_width = 4.0;
            int num_lanes = 3;
            double target_speed_m_s = 49.5 / 2.237; // mph to m/s
            double delta_t_path_points = 0.02;


            //////////////////////////////////////////////////////////////////////////////////////
            // Behaviour planning
            //
            // Here we'll analyse where the other vehicles are around us, and decide whether to
            // stay in our current lane or instigate a lane change
            /////////////////////////////////////////////////////////////////////////////////////

            // Start by assuming we are free to switch left or right or continue unimpeded
            // in our current lane. As we process the list of other vehicles, we will find
            // out if any are blocking us from those options.
            bool left_available = true, right_available = true, ahead_limiting = false;

            // We hope to stay close to the road speed limit, but if we are stuck behind
            // another car we will have to slow down to avoid hitting it
            double this_lane_max_speed = target_speed_m_s;

            // To allow for wraparound on the circular track, keep track of maximum s
            // coordinate seen for any vehicle
            check_for_greatest_s(car_s, max_s);

            // If we are close to the 
            for (auto other_veh : sensor_fusion)
            {
              // "The data format for each car is: [ id, x, y, vx, vy, s, d]"
              double vx    = other_veh[3];
              double vy    = other_veh[4];
              double speed = sqrt(vx * vx + vy * vy);
              double s     = other_veh[5];
              double d     = other_veh[6];

              // Again, keep track of max s coord seen
              check_for_greatest_s(s, max_s);

              // Project where the other car will be at the start of our planned path
              // extension. Obviously it is only an approximation that the other car
              // is continuing at constant speed, which will get worse the further out
              // we reuse our old planned path. Hence the number of reused path points
              // has been kept reasonably small here.
              double their_projected_s = s + delta_t_path_points * prev_size * speed;

              // Some adjustable parameters to tune safe lane changes. Given the
              // uncertainties in extrapolating current states into the future, we
              // allow for the fact that a car in our lane that is numerically just
              // behind us might be there because we are on course to drive into
              // it, so it should not be neglected.
              double safe_gap_ahead             = 30;
              double safe_gap_behind            = 20;
              double same_lane_behind_epsilon   = 10;
              double safe_following_speed_propn = 0.95;
              double safe_overtake_speed_propn  = 1.10;

              // Calculate fractional lane index of other vehicle
              double lane_idx = (d - (lane_width / 2.0)) / lane_width;

              // Figure out how far ahead or behind we are relative to the other car, allowing
              // for track wraparound (Frenet coordinate goes back to zero each lap, at what
              // we assume is max_s as that is the largest s-value we have seen):
              double we_are_ahead_by = our_projected_s - their_projected_s; // normal case
              if (fabs(we_are_ahead_by) > 0.5 * max_s)
              {
                // We are closer than we think because of wraparound, as the difference is greater
                // than half the track circumference. Recompute our delta s modulo the track circumference.
                // Note the result is signed.
                if (we_are_ahead_by >= 0.0)
                {
                  we_are_ahead_by = (our_projected_s - max_s) - their_projected_s;
                }
                else
                {
                  we_are_ahead_by = our_projected_s - (their_projected_s - max_s);
                }
              }

              bool we_are_in_front = (we_are_ahead_by >= 0.0);

              bool car_close_if_in_adjacent_lane = ((we_are_in_front &&  we_are_ahead_by < safe_gap_behind) ||
                                                    (!we_are_in_front && -we_are_ahead_by < safe_gap_ahead)   );

              bool car_close_if_in_our_lane      = ((we_are_in_front &&  we_are_ahead_by < same_lane_behind_epsilon) ||
                                                    (!we_are_in_front && -we_are_ahead_by < safe_gap_ahead         )   );


              if (fabs(lane_idx - desired_lane_idx) < 0.5)
              {
                // Other vehicle is in our lane
                if (car_close_if_in_our_lane)
                {
                  // Fairly closely ahead of us, reduce speed to be a bit slower than this car in front
                  double safe_speed = speed * safe_following_speed_propn;
                  if (safe_speed < this_lane_max_speed) this_lane_max_speed = safe_speed;
                  ahead_limiting = true;
                }
              }
              else if (lane_idx >= (desired_lane_idx + 0.5) && lane_idx <= (desired_lane_idx + 1.5))
              {
                // Vehicle immediately to our right
                if (car_close_if_in_adjacent_lane)
                {
                  // Vehicle is blocking us from changing lanes to the right
                  right_available = false;
                  // Avoid passing too fast as simulator may make other vehicle cut in front of us
                  double safe_speed = speed * safe_overtake_speed_propn;
                  if (safe_speed < target_speed_m_s) target_speed_m_s = safe_speed;
                }
              }
              else if (lane_idx <= (desired_lane_idx - 0.5) && lane_idx >= (desired_lane_idx - 1.5))
              {
                // Vehicle immediately to our left
                if (car_close_if_in_adjacent_lane)
                {
                  // Vehicle is blocking us from changing lanes to the left
                  left_available = false;
                  // Avoid passing too fast as simulator may make other vehicle cut in front of us
                  double safe_speed = speed * safe_overtake_speed_propn;
                  if (safe_speed < target_speed_m_s) target_speed_m_s = safe_speed;
                }
              }
              else
              {
                // Other vehicle may be in non-adjacent lane, not of interest
              }
            }

            // Avoid rapid consecutive lane change decisions, otherwise could end up triggering
            // out of lane warning if keep changing mind due to similar traffic ahead in
            // two lanes which might leave us driving between them for a while
            double min_time_between_lane_changes = 4.0;
            bool ok_to_change_lanes = (time_since_last_lane_change >= min_time_between_lane_changes);

            // Note: by only allowing lane changes separated in time by that amount, we have
            // effectively baked in a finite state machine; the vehicle is locked into a lane-changing
            // state once a lane change is triggered, and held there until it has settled into its
            // new lane.

            if (ok_to_change_lanes && right_available && desired_lane_idx < num_lanes - 1)
            {
              // Just for fun, prefer inner ("slow") lane if available, which is what we do
              // in Europe at least. This also covers the case where we happen to be stuck
              // behind another car in this lane but the right lane is free.
              desired_lane_idx++;
            }
            else if (!ahead_limiting)
            {
              // Stay in current lane, as there is nothing in our way
            }
            else if (ok_to_change_lanes && left_available && desired_lane_idx > 0)
            {
              // We are stuck behind another car but the lane to our left is free, so switch
              desired_lane_idx--;
            }
            else
            {
              // Stuck behind another vehicle in this lane so just regulate speed to avoid it
              target_speed_m_s = this_lane_max_speed;
            }

            // Are we currently actually in the lane we wish to end up in? If not, we're in
            // the process of changing lanes, which will take a while
            double current_lane_idx_dbl = (int) (car_d - (lane_width / 2.0)) / lane_width;
            int current_lane_idx        = (int)(current_lane_idx_dbl + 0.5); // round to nearest integer
            bool changing_lanes         = (current_lane_idx != desired_lane_idx);

            if (changing_lanes) time_last_lane_change = steady_clock::now();

            //////////////////////////////////////////////////////////////////////////////////////
            // Path planning -- now based again on method in project walkthrough video
            //
            // Add future path points to add so we always have required number ahead of current state


            // To get a smooth future path, we deliberately take points from the map that are
            // quite widely spaced, and then fit a spline through them. If the spacing is
            // *too* wide, we may not follow the road accurately enough, and get out of lane
            // warnings. But if they are not widely spaced enough, we will start to get the
            // jinks in the path defined by the map waypoints, leading to unwanted lateral
            // accelerations or jerk. As a slight cheat, here I go for a wider spacing if
            // we are changing lanes (to keep lane changes really smooth), but a narrower
            // spacing if we are just following the current lane (to avoid possibly going
            // off track):
            double dist_between_distant_points = changing_lanes ? 60 : 30;

            // How many distant points to set up spline fit for -- needs to be more than
            // enough to ensure we don't try and interpolate beyond the end of our fit:
            int num_distant_points = 3;

            // Compute the few desired waypoints well out in front from the map
            double x_current = 0.0; // relative to ego car hence starts at zero
            for (int i = 0; i < num_distant_points; i++)
            {
              double s = car_s + dist_between_distant_points * (i + 1);
              double d = lane_width * (desired_lane_idx + 0.5); // middle of desired lane
              vector<double> x_y = getXY(s, d, map_waypoints_s, map_waypoints_x, map_waypoints_y);
              ptsx.push_back(x_y[0]);
              ptsy.push_back(x_y[1]);
            }

            // So now we have the last couple of path points returned to us by the simulator
            // from last time (or a synthetic version if we've just started), and the future
            // points well out ahead in this lane, all in global x-y coordinates.
            // Convert all those points to car's reference frame to avoid getting multi-valued
            // spline lookup as a function of x in near future:
            for (int i = 0; i < ptsx.size(); i++)
            {
              double x_global   = ptsx[i];
              double y_global   = ptsy[i];
              double x_shift    = x_global - ref_x; // translate to origin of car
              double y_shift    = y_global - ref_y;
              double x_relative = x_shift * cos(-ref_yaw) - y_shift * sin(-ref_yaw); // rotate to ref frame of car
              double y_relative = y_shift * cos(-ref_yaw) + x_shift * sin(-ref_yaw);
              ptsx[i]           = x_relative;
              ptsy[i]           = y_relative;
            }

            // Create a spline to fit those points now they should be single-valued as a
            // function of x
            tk::spline spline_fit;
            spline_fit.set_points(ptsx, ptsy);

            // First use previously unconsumed global x-y points for future path
            for (int i = 0; i < prev_size; i++)
            {
              next_x_vals.push_back(previous_path_x[i]);
              next_y_vals.push_back(previous_path_y[i]);
            }

            ////////////////////////////////////////////////////////////////////////////////
            // Speed control
            //
            // Adjust speed towards target, limiting acceleration and jerk.
            // To do this, I've maintained a value of the current acceleration, which we do
            // not want to change too abruptly (i.e. cause jerk). This is smoothly pushed
            // towards our currently desired acceleration via a first-order filter to
            // avoid abrupt changes. The filter takes account of the real-time rate at which
            // this function is being called, which depends quite a lot on simulator mode.
            //
            // Start with desired raw acceleration; needs to be quite strong to avoid collisions.
            // Want to accelerate if going too slowly, decelerate if going too fast, compared
            // with currently desired speed:
            double desired_accel = (target_speed_m_s - current_speed_m_s) * 0.25; // at zero to get to 20 m/s ~4 m/s2

            // First-order filter on acceleration to avoid sudden changes (jerk):
            double accel_filter_const = 2.0; // by experiment
            current_accel_m_s2 += (desired_accel - current_accel_m_s2) * delta_t_program * accel_filter_const;

            // Clip acceleration to avoid exceeding allowed maximum:
            double max_accel = 3.0; // too close to limit of 5 and risk acceleration warnings
            if (current_accel_m_s2 > max_accel) current_accel_m_s2 = max_accel;
            if (current_accel_m_s2 < -max_accel) current_accel_m_s2 = -max_accel;

            // Now acceleration is computed, update current speed, allowing for real-time
            // calling rate of this code:
            current_speed_m_s += current_accel_m_s2 * delta_t_program;

            //cout << "dt=" << delta_t_program << " target_speed_m_s=" << target_speed_m_s << " current_speed_m_s=" << current_speed_m_s << " desired_accel=" << desired_accel << " current_accel_m_s2=" << current_accel_m_s2 << endl;


            // 50 was suggested in walkthrough video as number of path points to compute, but
            // makes less responsive to future changes if lock in too much path too early;
            // but also need reasonable number in case simulator run in fastest low-res mode. By
            // experiment, this works pretty well:
            int num_total_path_points = 30;

            // As per walkthrough video, draw future path points from the spline fit we have
            // prepared:
            double target_x = 30.0;
            double target_y = spline_fit(target_x);
            double target_dist = sqrt(target_x * target_x + target_y + target_y); // approximation on hypotenuse to curve

            for (int i = 0; i < num_total_path_points - prev_size; i++)
            {
              // Unlike the walkthrough video, our current acceleration is accounted for here, so the
              // future points will not be quite evenly spaced:
              double N = target_dist / (delta_t_path_points * current_speed_m_s); // Number of sample time segments to achieve target distance

              double x_rel = x_current + target_x / N;
              double y_rel = spline_fit(x_rel);
              x_current = x_rel;

              // Convert back to global coords
              double x_abs = ref_x + (x_rel * cos(ref_yaw) - y_rel * sin(ref_yaw));
              double y_abs = ref_y + (x_rel * sin(ref_yaw) + y_rel * cos(ref_yaw));

              next_x_vals.push_back(x_abs);
              next_y_vals.push_back(y_abs);

              // Update speed for current desired acceleration as we extrapolate these points
              current_speed_m_s += delta_t_path_points * current_accel_m_s2;
            }

          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }

    first_time = false;
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
    // As with previous projects, this call in provided code in Windows 10 bash
    // environment just causes segmentation fault, so commented out now:
    //   ws.close();
    // Simulator can then connect and disconnect without crashing this program.

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
