
GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):

    # Starting with walkthrough video code
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                 accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # Gains from walkthrough video
        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0.0
        max_throttle = 0.2
        self.throttle_controller = PID(kp, ki, kd, min_throttle, max_throttle)
        
        # Walkthrough video: velocity comes in a bit noisy so needs filtering
        tau = 0.5 # time constant where 1/(2.pi.tau) = cutoff frequency
        sample_time = 1.0 / 50.0
        self.vel_lpf = LowPassFilter(tau, sample_time)
        
        # Cache other parameters, all starting from walkthrough video
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity # can allow for mass of this if really want to!
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit # Comfort limits
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        
        self.last_time - rospy.get_time()


    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # Initially from walkthrough video
        
        if not dbw_enabled:
            # Avoid integral wind-up when in manual mode/stopped at lights etc
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
            
        current_vel = self.vel_lpf.filt(current_vel)
        
        # Debug output
        # rospy.logwarn("Angular vel: %f" % angular_vel)
        # rospy.logwarn("Target vel: %f" % linear_vel)
        # rospy.logwarn("Target angular vel: %f" % angular_vel)  # CW don't have separate target
        # rospy.logwarn("Current vel: %f" % current_vel)
        # rospy.logwarn("Filtered vel: %f" % self.vel.lpf.get()) # CW: does calling this advance filter?
        
        steering = self.yaw_controller.get_steering(linear_vel, angular_vel, current_vel)
        
        vel_error = linear_vel - current_vel
        self.last_vvel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.0
        
        if linear_vel == 0.0 and current_vel < 0.1:
            # Hold at a stop against creep torque of automatic transmission
            throttle = 0.0
            brake = 700 # sufficient torque (in Nm) to keep stopped
                        # (Note in classroom has to be larger than 400 in walkthrough video)
            # CW: shouldn't we be resetting the throttle PID controller?
        elif throttle < 0.1 and vel_error < 0.0:
            # Going faster than we want so need to do some braking
            throttle = 0.0
            decel = max(vel_error, self.decel_limit) # CW units m/s and m/s2 though??
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius # torque in Nm
        else:
            # Keep controller value
            pass
            
        return throttle, brake, steering
