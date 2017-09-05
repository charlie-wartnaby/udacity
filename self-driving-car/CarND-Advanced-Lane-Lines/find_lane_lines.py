"""
Self-driving car course: advanced lane line finding project

Author:  Charlie Wartnaby, Applus IDIADA
Email:   charlie.wartnaby@idiada.com
"""

import cv2
import glob
import math
from   moviepy.editor import VideoFileClip
import numpy as np
import os.path
import sys


def poly_result(y, poly_coeffs):
    """Just evaluate second-order polynomial at point provided"""

    return poly_coeffs[0] * y * y + poly_coeffs[1] * y + poly_coeffs[2]

def first_order_filter(old_value, raw_value, filter_const):
    """First-order infinite impulse response filter for smoothing"""

    if old_value is None:
        return raw_value
    else:
        return filter_const * raw_value + (1.0 - filter_const) * old_value

def calibrate_camera_from_images(cam_cal_path_pattern, nx, ny):
    """Matches all image files against path pattern provided, and uses those
     to perform camera calibration. Translation and rotation vectors are
     discarded, as it is assumed that different images are taken with different
     camera positions and orientations, so those vectors will not be common."""

    print("Calibrating camera expecting %d by %d corners in images matching '%s'" % (nx, ny, cam_cal_path_pattern))

    cal_image_filenames = glob.glob(cam_cal_path_pattern, recursive=True)
    if len(cal_image_filenames) <= 0:
        print("Error: no camera calibration images found matching '" + cam_cal_path_pattern + "'")
        sys.exit(1)

    # Use the same real-world coordinates for the corners in every case, from
    # (x,y,z)=(0,0,0) in the top-left corner to (nx-1,ny-1,0) in the bottom-
    # right corner, i.e. a fixed grid of integers. The depth coordinate z
    # can always be zero.
    common_obj_points = np.zeros((nx*ny, 3), np.float32) # flat array of 3 elements per entry
    common_obj_points[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # only write x and y elements, leave z alone
    #print("Debug: common_obj_points=%s" % str(common_obj_points))

    img_points = []
    obj_points = []
    good_files = 0
    first_image = None

    for cal_image_filename in cal_image_filenames:
        # Load image from disk
        raw_image = cv2.imread(cal_image_filename)

        # Need to convert to greyscale to use findChessboardCorners. As OpenCV was used to
        # read the image in, colour channels are in order BGR not RGB
        greyscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

        # Try and find chessboard grid intersections (corners)
        ret, corners = cv2.findChessboardCorners(greyscale_image, (nx, ny), cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not ret or len(corners) != nx * ny:
            print("Warning: expected corners not found in '%s', skipping that file" % cal_image_filename)
            #print ("Debug: ret False" if not ret else "len(corners)=" + str(len(corners)))
        else:
            # Found corners OK, so use data from this image in the calibration set
            #print("Debug: corners found OK in %s" % cal_image_filename)
            img_points.append(corners)
            obj_points.append(common_obj_points)
            good_files += 1
            if first_image is None:
                first_image = raw_image # need this for size below

    print("%d image files had usable chessboard corner points" % good_files)

    if good_files <= 0:
        print("Error: no suitable chessboard image points found")
        sys.exit(1)

    img_shape = np.shape(first_image)
    #print("Debug: for camera cal img_shape=" + str(img_shape))
    retval, camera_matrix, distortion_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_shape[:-1], None, None)
    #print("Debug: retval, camera_matrix, distortion_coeffs = %s, %s, %s" % (str(retval), str(camera_matrix), str(distortion_coeffs)))
    print("Camera calibration complete")

    return camera_matrix, distortion_coeffs

def undistort_one_image(raw_image, camera_matrix, distortion_coeffs):
    """Apply undistorting transform to image to correct for camera distortion"""

    # Just use library function
    return cv2.undistort(raw_image, camera_matrix, distortion_coeffs, None, camera_matrix)

def create_interest_region_mask(undistorted_image):
    """Create a binary mask which can be ANDd or multiplied with camera images to leave only
    the regions likely to contain lane lines surviving. Use the size of the image
    provided as a template, with a mask polygon scaled to that image assuming the 
    camera is facing horizontally forwards"""

    # We only want the image to be 2D (i.e. no colour channel), so use only first
    # two dimensions of image supplied to size it
    required_dimensions = np.shape(undistorted_image)[:2]
    mask = np.zeros(required_dimensions)
    #print("Debug: shape(img)=%s shape(mask)=%s" %(np.shape(undistorted_image), np.shape(mask)))

    # Figure out pixel coordinates of mask quadrilateral for this image size
    x_size = required_dimensions[1] # second dimension as first are rows i.e. y
    y_size = required_dimensions[0]

    # Set cropping region using proportions of current image size.
    # These have been adjusted approximately to ensure the useful part of lane lines
    # in all the test images is retained, but not too much outside that area, without
    # risking throwing away useful parts

    # If vanishing point is ideally half way up, crop a bit below that, as the lines
    # get unrecognisable too far into the distance
    crop_top_proportion = 0.6

    # The very bottom of the images is occupied by the car bonnet (hood) so get
    # rid of that too
    crop_bottom_proportion = 0.1

    # At the sides, the lane lines intersect the bottom edge of the image with a
    # fair bit of extraneous image to either side, so remove that amount
    # from each side
    crop_bottom_side_proportion = 0.1

    # As lines converge perspective-wise, mask region gets thinner towards the
    # top
    crop_top_side_proportion = 0.45

    # Compute vertices of trapezium we want to retain
    top_y = y_size * crop_top_proportion
    bottom_y = y_size * (1.0 - crop_bottom_proportion)
    bottom_y = min(y_size-1, bottom_y) # ensure on-image

    bottom_left_x = crop_bottom_side_proportion * x_size
    bottom_right_x = x_size - bottom_left_x # make symmetric
    bottom_right_x = min(x_size-1, bottom_right_x) # ensure on-image

    top_left_x = crop_top_side_proportion * x_size
    top_right_x = x_size - top_left_x # make symmetric
    top_right_x = min(x_size-1, top_right_x) # ensure on-image

    # Now walk around the 4 vertices of the required trapezium
    # to define its coordinates:
    vertices = np.array([[(bottom_left_x,  bottom_y),
                          (top_left_x,     top_y   ),
                          (top_right_x,    top_y   ),
                          (bottom_right_x, bottom_y)]],
                        dtype=np.int32)
    #print("Debug: vertices=%s" % vertices)

    cv2.fillPoly(mask, vertices, 255)

    return mask

def colour_trans_one_image(raw_image):
    """Apply colour transformations to make yellow and white lane lines stand
    out better from the road colour"""

    # Transform to hue/luminosity/saturation encoding
    hls = cv2.cvtColor(raw_image, cv2.COLOR_RGB2HLS)

    # Tried the three channels with the test images. S-channel makes both white and yellow
    # lines stand out nicely. White are good in L-channel but not yellow. H channel
    # not useful at all in this context! So stick with S-channel:
    s_channel = hls[:,:,2]

    return s_channel

def find_useful_gradient(greyscale_image, interest_region_mask):
    """Return an image map where high value reflects useful intensity gradient
    in the supplied monochrome image. This actually performs line direction
    processing too in an approximate way, by highlighting lines that run at
    about 45 deg or -45 deg in the image as viewed, over other lines that
    are closer to horizontal or vertical (see detailed comments)"""

    # Apply some smoothing across pixels; quite a big kernel size seems to
    # work out best for retaining weaker originally-yellow lines
    smoothing_kernel_size = 15

    # Use Sobel operator to identify pixels with intensity trending in desired
    # direction. But what direction? Before birds-eye distortion, the left
    # lane line will be roughly parallel to x=y (with origin in bottom left)
    # or x=-y (in graphics convention with origin at top left). Conversely
    # the right lane line will be roughly parallel to x=-y (origin bottom left).
    # (Note: the cv2 Sobel function seems to work as if y coordinates increase
    # upward, i.e. positive when brighter pixel above darker one, so from
    # now on talking about bottom-left origin here!)
    # Then consider the intensity I as a function of x and y:
    # On the left edge of the left line dI/dx = +ve and dI/dy = -ve
    # On the right edge of the left line dI/dx = -ve and dI/dy = +ve
    # So to emphasize both edges, we should *subtract* the dI/dx and dI/dy
    # maps from each other (i.e. Sobel x and Sobel y) to highlight both
    # edges of the left lane line.
    # On the left edge of the right line dI/dx = +ve and dI/dy = +ve
    # On the right edge of the right line dI/dx = -ve but dI/dy = -ve
    # So to emphasize both edges, we should *add* the Sobel x and y
    # maps from each other to highlight the right lane line.
    # So instead of taking the Sobel gradient in the x or y
    # direction, or both, and just adding it, my trick here is to both
    # add those components (highlighting left line at approx 45 deg)
    # and subtract them from each other (highlighting right line at approx
    # -45 deg), take the absolute magnitude for each of those individually, and
    # finally sum the two to highlight where we think
    # both left or right lane lines are. That will find these diagonal
    # lines more strongly than (unwanted) horizontal or vertical lines.
    # So this achieves both direction processing and edge detection at the same time.

    sobel_x = cv2.Sobel(greyscale_image, cv2.CV_64F, 1, 0, ksize=smoothing_kernel_size)
    sobel_y = cv2.Sobel(greyscale_image, cv2.CV_64F, 0, 1, ksize=smoothing_kernel_size)

    # Take abs for left and right lanes individually first; if we add them
    # without having done abs first, y will just cancel out!
    sobel_left_lane_line_highlights  = np.absolute(sobel_x + sobel_y)
    sobel_right_lane_line_highlights = np.absolute(sobel_x - sobel_y)
    
    # Only now add them together.
    sobel_sum = sobel_left_lane_line_highlights + sobel_right_lane_line_highlights

    # Up to now, a line that is horizontal or vertical will also do quite well, because
    # it will have a strong signal in either the dx or dy, and zero in the other direction.
    # But if we multiply what we have so far by the abs magnitude of both dI/dx and dI/dy,
    # a horizontal or vertical line will be wiped out as one of those will be zero;
    # while ~45deg lane lines should survive nicely as both directions will be
    # strongly non-zero
    abs_x = np.absolute(sobel_x)
    abs_y = np.absolute(sobel_y)
    max_abs_x = np.max(abs_x)
    max_abs_y = np.max(abs_y)
    allowance = 2 # to avoid wiping out weak/patchy lines or curves towards vertical inadvertently
    sobel_diagonal_filtered = sobel_sum * (abs_x / max_abs_x + allowance) * (abs_y / max_abs_y + allowance)

    # Before normalisation, multiply by region of interest mask so that we discard
    # any accidental bright regions elsewhere in image that might dominate overall
    # brightness otherwise
    masked_sobel_sum = sobel_diagonal_filtered * interest_region_mask

    # Rescale to make nice greyscale image for viewing/debugging purposes (which
    # we can then apply thresholding to, etc, for actual processing)
    scaled_sobel = np.uint8(255 * masked_sobel_sum / np.max(masked_sobel_sum))

    return scaled_sobel

def apply_binary_threshold(image, low_thresh, high_thresh):
    """Retain only pixels in supplied greyscale image with intensity in specified
    interval, returning binary image with intensities of only 0 (black) or
    255 (white)"""

    # Start with blank canvas of zeroes (black)
    binary = np.zeros_like(image)

    # Set pixels in required range to 1
    binary[(image > low_thresh) & (image <= high_thresh)] = 255

    return binary

def highlight_window_and_peak(searched_area_image, x_min, bottom_y, x_max, top_y, peak_x):
    """ Paint box to show an area searched for lane line peaks, and a line
    where the centre of the lane line was determined to be"""

    cv2.rectangle(searched_area_image, (x_min,top_y), (x_max, bottom_y), (0,255,0), 10)
    cv2.line(searched_area_image, (peak_x,top_y), (peak_x,bottom_y), (0,0,255), 10)


class LaneProcessor():
    def __init__(self, camera_matrix, distortion_coeffs):
        self.left_count_until_reinit = 0
        self.right_count_until_reinit = 0
        self.reinit_count_on_good_detection = 50
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.interest_region_mask = None
        self.warp_to_birdseye_matrix = None
        self.warp_from_birdseye_matrix = None
        self.x_size = 0
        self.y_size = 0
        self.gaussian_window_function = None
        self.min_convolution_detection_strength = 0.03
        self.left_poly_coeffs = []
        self.right_poly_coeffs = []
        self.min_conv_points_for_poly_fit = 4
        self.min_window_points_for_poly_fit = 100
        self.window_round_prev_fit_width_propn = 0.2 # proportion of total image width to look for new points within
        self.min_proportion_in_poly_fit_window = 0.7
        self.poly_filter_in_const = 0.1
        self.y_metres_of_road       = 30    # suggested number from tutorial, metres down the road
        self.y_metres_to_baseline   = 4     # guessing how far from camera to start of visible lanes as used in calibration
        self.x_metres_between_lanes = 3.7   # between lanes in straight-ahead calibration picture, again number from tutorial
        self.y_metres_per_pixel = 0 # calculated later
        self.x_metres_per_pixel = 0 # calculated later
        self.y_bottom_baseline_pixels = 0 # calculated later
        self.left_real_coeffs = None   # polynomial fit to lines in real space in metres
        self.right_real_coeffs = None   # polynomial fit to lines in real space in metres
        self.lateral_displacement_metres = None
        self.lane_width_metres = None
        self.mean_curvature_metres = None
        self.steering_angle_degrees = None
        self.real_data_filter_const = 0.05 # to smooth out displayed values
        self.birdseye_pip_relative_size = 0.5
        self.text_panel_relative_size = 0.5

    def reset(self):
        """ Forget any previous data, so we start from scratch with next image/frame"""
        self.left_count_until_reinit = 0
        self.right_count_until_reinit = 0
        # Display real-world data as unavailable until computed
        self.lateral_displacement_metres = None
        self.lane_width_metres = None
        self.mean_curvature_metres = None
        self.steering_angle_degrees = None

    def compute_warp_matrices(self):
        """Compute matrices for perspective transforms to and from birds-eye view"""

        # This was 'calibrated' using the straight_lines1.jpg test image after camera undistortion.
        # See project write-up for a picture highlighting the required rectangle.

        # I've laid out the coordinates in the source code here roughly as they appear in the image,
        # perspective-wise! So the src points are in normal view, dst transformed to a rectangle
        # So: source points, lower part of frame only, converging at the top perspective-wise:
        src_lane_points = np.float32(
                           [
                                      [574,466],      [708,466],
                            [275,672],                           [1033,672]])
        # destination points, now evenly spaced horizontally and extending to near top of frame vertically:
        dst_lane_points = np.float32(
                           [[275,50 ],                           [1033,50 ],

                            [275,672],                           [1033,672]])

        self.warp_to_birdseye_matrix   = cv2.getPerspectiveTransform(src_lane_points, dst_lane_points)
        self.warp_from_birdseye_matrix = cv2.getPerspectiveTransform(dst_lane_points, src_lane_points)

        # Also keep a note of the x-coords where we most expect to find the lines at the bottom
        # from that calibration image
        lane_centre_bottom_left_x  = int(dst_lane_points[2][0])
        lane_centre_bottom_right_x = int(dst_lane_points[3][0])
        self.base_expected_lane_x = [lane_centre_bottom_left_x, lane_centre_bottom_right_x]
        #print("Debug: expected x-coords of lane centres at bottom=%s" % self.base_expected_lane_x)

        # While we're here, figure out the conversion from metres to pixels in x and y
        # directions in the birdseye image
        inter_lane_dist_pixels  = lane_centre_bottom_right_x  - lane_centre_bottom_left_x
        road_distance_pixels    = dst_lane_points[2][1]       - dst_lane_points[1][1]
        self.x_metres_per_pixel = self.x_metres_between_lanes / inter_lane_dist_pixels
        self.y_metres_per_pixel = self.y_metres_of_road       / road_distance_pixels
        self.y_bottom_baseline_pixels  = dst_lane_points[2][1]

    def warp_image(self, image):
        """Apply perspective transform to warp image to or from birds-eye view"""

        # We're just wrapping OpenCV library with appropriate options here

        # May as well make warped image have same dimensions as original
        (self.y_size, self.x_size) = np.shape(image)

        img_size = (self.x_size, self.y_size)
        return cv2.warpPerspective(image, self.warp_to_birdseye_matrix, img_size, flags=cv2.INTER_LINEAR)

    def update_poly_fit_if_enough_points(self, x_points, y_points, num_available_points, min_points_for_conv_poly_fit):
        """Fit 2nd order polynomial to supplied list of [x,y] coords and update coefficients
        array, but only if we have been given enough points to reasonably do so,
        and (optionally) a high enough proportion of the points are in the window we're looking at.
        Return new coefficients if OK, else None"""

        if len(x_points) < min_points_for_conv_poly_fit:
            # Not enough points
            return None

        if num_available_points is not None and (num_available_points <= 0 or
                                                (float(len(x_points)) / num_available_points < self.min_proportion_in_poly_fit_window)):
            # Not a high enough proportion of points within the detection window,
            # must be other noise in the image or we've gone off track
            return None

        # Do polynomial fit of x(y), where x is the dependent variable because (as explained in
        # the course) the lane line may be multi-valued if consisdered as y(x)
        new_coeffs = np.polyfit(y_points, x_points, 2)
        return new_coeffs

    def draw_poly_fit(self, image, poly_coeffs_x_as_fn_y, confirmed_this_time):
        """Draw polynomial fit curve using 2nd order polynomial coefficients provided for x(y)"""

        if len(poly_coeffs_x_as_fn_y) < 3:
            # Don't have any fit yet, give up
            return

        unconfirmed_colour = (128,128,128)
        confirmed_colour   = (0,255,255)
        colour_this_time   = confirmed_colour if confirmed_this_time else unconfirmed_colour

        for y in range(self.y_size):
            x = int(poly_result(y, poly_coeffs_x_as_fn_y))
            cv2.circle(image, (x,y), 6, colour_this_time, 2)

    def find_lines_by_sliced_convolution(self, birdseye_image):
        """Search from scratch for lane lines in provided binary image
        using peaks in convolution with a shaped peak function on successive
        slices, working up from the bottom of the image where we have most
        confidence in where the lines are likely to be"""

        # As we are looking from scratch, use generously thick slabs of the
        # image to look in vertically, and look quite widely
        # around the base expected lane positions, to maximise our chance
        # of finding something useful:
        detection_window_x_size = int(self.x_size * 0.2)
        detection_window_y_size = int(self.y_size / 10)

        # From test images, after birdseye warp the lane lines can get pretty
        # thick (e.g. 130 pixels out of 1280), and want our convolution function
        # to span a whole line to find its centre properly, so use that kind
        # of thickness (but as ratio of image size so this works in future
        # with other resolutions)
        convolution_width = int(self.x_size * (130.0/1280.0))

        # Start with a copy of the lane positions from the calibration image
        # as our best guess of where the lanes might be at the image bottom
        current_lane_centre_x = self.base_expected_lane_x[:]

        # The tutorial example code showed convolution with a 'top hat' function
        # (i.e. just a flat step). The problem with that is that if it completely
        # envelops an area of bright pixels, the resulting convolution does not
        # specifically peak when those pixels are centred in the window; the
        # convolution value will be flat as the window slides over the lane line.
        # And as we used a modified Sobel gradient method to find the lines, we
        # will tend to have a double peak in the intensity (corresponding to the
        # left and right edges of each line).
        # Instead use a convolution shape that has a peak, so it gives more
        # emphasis to bright pixels in the window centre than at the edges. That
        # way, the convolution curve will peak when the bright pixels are *centred*
        # in the convolution window specifically.
        if self.gaussian_window_function is None or len(self.gaussian_window_function) != convolution_width:
            # Compute nice Gaussian (bell curve) shape to fit our desired detection
            # window size, just one standard deviation wide so it still has quite
            # strong magnitude right to the edges
            gaussian_centre = convolution_width / 2
            gaussian_stddev = convolution_width / 2
            gaussian_height = 1.0
            curve = []
            sum = 0.0
            for x in range(convolution_width):
                exponent = - (x-gaussian_centre)*(x-gaussian_centre) / (2 * gaussian_stddev * gaussian_stddev)
                y = gaussian_height * math.exp(exponent)
                curve.append(y)
                sum += y
            self.gaussian_window_function = np.array(curve, dtype=float)
            self.gaussian_window_sum = sum
            #print ("Debug: recomputed Gaussian window func: " + str(self.gaussian_window_function))

        # Image provided was greyscale with white (255) or black (0) pixels. No need
        # to normalise that mathematically; I just wanted to get numbers that I could
        # judge more easily:
        normalised_image = birdseye_image / 255

        # Keep track of good left and right lane point coordinates so we can do a fit
        # through all the good ones
        good_left_points_x  = []
        good_left_points_y  = []
        good_right_points_x = []
        good_right_points_y = []

        # Work upwards from bottom of image, centring the area scanned by convolution
        # on our best convolution peaks from the previous slice
        for slice_bottom_y in range(self.y_size-1, 0, -detection_window_y_size):
            slice_top_y = (slice_bottom_y - detection_window_y_size) + 1
            slice_avg_y = (slice_bottom_y + slice_top_y) / 2

            # Compute vertical sum of bright pixels in this slice, so we have a
            # 1-D curve to convolve with
            slice_1d_sum = np.sum(normalised_image[slice_top_y : slice_bottom_y-1, :], axis=0)
            # Normalising that so that 1 means pixel was set in every row in slice,
            # just to make it easier to understand what a 'good' number is
            slice_1d_sum = slice_1d_sum / detection_window_y_size
            #print ("Debug: part of normalised slice sum from y=%d to y=%d = %s" % (slice_bottom_y, slice_top_y, slice_1d_sum[200:340]))

            # Do convolution with the Gaussian shape, using 'same' mode so we get a
            # vector of the same width as the input image (being the wider of the two
            # inputs), meaning values near edge bound to trail off towards zero; has
            # the convenience of keeping x-coords the same as original
            slice_convolution = np.convolve(slice_1d_sum, self.gaussian_window_function, mode='same')
            # Again normalise that so 1.0 is what we'd get if all pixels below the curve were set,
            # to make it easier to judge whether we found something valid
            slice_convolution /= self.gaussian_window_sum
            #print ("Debug: part of convolution from y=%d to y=%d = %s" % (slice_bottom_y, slice_top_y, slice_convolution[200:340]))

            # Look for peak intensities in convolution output on the left and right hand sides,
            # within the x-detection window width from the previous slice we processed (or the
            # base values if we didn't have a previous slice)
            left_x_min = int(current_lane_centre_x[0] - detection_window_x_size/2)
            left_x_max = left_x_min + detection_window_x_size
            right_x_min = int(current_lane_centre_x[1] - detection_window_x_size/2)
            right_x_max = right_x_min + detection_window_x_size
            # Keep those sensible
            left_x_min = max(left_x_min, 0)
            left_x_max = min(left_x_max, int(self.x_size/2) - 1)
            right_x_min = max(right_x_min, int(self.x_size/2))
            right_x_max = min(right_x_max, self.x_size-1)
            min_width = 20
            if (left_x_max - left_x_min < min_width or right_x_max - right_x_min < min_width or left_x_min >= right_x_min):
                print("Error: convolution detection windows not sensible")
                print("left_x_min=%d left_x_max=%d right_x_min=%d right_x_max=%d" % (left_x_min, left_x_max, right_x_min, right_x_max))
                return None, None
            #print ("Debug: from y=%d to y=%d, windows at %d,%d to %d,%d" % (slice_bottom_y, slice_top_y, left_x_min, left_x_max, right_x_min, right_x_max))
            left_peak_x  = int(np.argmax(slice_convolution[left_x_min:left_x_max+1]))   + left_x_min
            try:
                right_peak_x = int(np.argmax(slice_convolution[right_x_min:right_x_max+1])) + right_x_min
            except:
                print("Debug: right_x_min=%s right_x_max=%s" % (right_x_min,right_x_max))
                sys.exit(1)
            left_peak_height = slice_convolution[left_peak_x]
            right_peak_height = slice_convolution[right_peak_x]
            #print ("Debug: from y=%d to y=%d, peaks at %d=%f, %d=%f" % (slice_bottom_y, slice_top_y, left_peak_x, left_peak_height, right_peak_x, right_peak_height))

            # Update our 'found' centre positions only if we got something. With dashed lane lines,
            # even with a good picture, we may well have found nothing. And we don't want to accept
            # a little patch of noise as a valid line either.
            if left_peak_height >= self.min_convolution_detection_strength:
                current_lane_centre_x[0] = left_peak_x
                highlight_window_and_peak(self.searched_area_image, left_x_min, slice_bottom_y, left_x_max, slice_top_y, left_peak_x)
                good_left_points_x.append(left_peak_x)
                good_left_points_y.append(slice_avg_y)
            if right_peak_height >=self.min_convolution_detection_strength:
                current_lane_centre_x[1] = right_peak_x
                highlight_window_and_peak(self.searched_area_image, right_x_min, slice_bottom_y, right_x_max, slice_top_y, right_peak_x)
                good_right_points_x.append(right_peak_x)
                good_right_points_y.append(slice_avg_y)

        # Considering the left and right lines, do a polynomial fit to the points found
        # only if we got a reasonable number of points this time.
        new_left_coeffs  = self.update_poly_fit_if_enough_points(good_left_points_x,  good_left_points_y,  None, self.min_conv_points_for_poly_fit)
        new_right_coeffs = self.update_poly_fit_if_enough_points(good_right_points_x, good_right_points_y, None, self.min_conv_points_for_poly_fit)

        return new_left_coeffs, new_right_coeffs, len(good_left_points_x), len(good_right_points_x)

    def keep_points_close_to_fit_and_show(self, nonzero_x, nonzero_y, fit_coeffs):
        """Given the arrays of x and corresponding y coordinates, keep only those within
        a reasonable distance of the 2nd order polynomial curve defined by the
        coefficients provided. Highlight acceptable search area on search image to visualise."""

        pixel_width = self.window_round_prev_fit_width_propn * self.x_size
        max_delta = pixel_width / 2

        indices = ((nonzero_x > poly_result(nonzero_y, fit_coeffs) - max_delta) &
                   (nonzero_x < poly_result(nonzero_y, fit_coeffs) + max_delta)  )

        plot_y = np.linspace(0, self.y_size-1, self.y_size)
        centre_fit_x = poly_result(plot_y, fit_coeffs)

        window_boundary_ascending_left = np.array([np.transpose(np.vstack([centre_fit_x-max_delta, plot_y]))])
        window_boundary_descending_right = np.array([np.flipud(np.transpose(np.vstack([centre_fit_x+max_delta, 
                              plot_y])))])
        window_boundary_complete = np.hstack((window_boundary_ascending_left, window_boundary_descending_right))

        window_img = np.zeros_like(self.searched_area_image)
        cv2.fillPoly(window_img, np.int_([window_boundary_complete]), (0,255, 0))
        self.searched_area_image = cv2.addWeighted(self.searched_area_image, 1, window_img, 0.3, 0)

        return indices

    def find_lines_near_existing_fit(self, birdseye_image):
        """Try and fit polynomial to points that lie within a reasonable distance
        of our current best estimate lane line positions from last time round,
        updating that estimate if the fit looks reasonable"""

        # Get all the coordinate pairs of bright pixels in the image, regardless
        # of their position:
        nonzero_coords = birdseye_image.nonzero()
        nonzero_x = nonzero_coords[1]
        nonzero_y = nonzero_coords[0]

        good_left_point_indices  = self.keep_points_close_to_fit_and_show(nonzero_x, nonzero_y, self.left_poly_coeffs)
        good_right_point_indices = self.keep_points_close_to_fit_and_show(nonzero_x, nonzero_y, self.right_poly_coeffs)

        good_left_points_x  = nonzero_x[good_left_point_indices]
        good_left_points_y  = nonzero_y[good_left_point_indices]
        good_right_points_x = nonzero_x[good_right_point_indices]
        good_right_points_y = nonzero_y[good_right_point_indices]
        
        # Compare with how many points we have *outside* the detection area. If it's too high, we'll
        # probably be fitting nonsense.
        midpoint_x = self.x_size / 2
        total_num_left_points  = np.count_nonzero(nonzero_x < midpoint_x)
        total_num_right_points = np.count_nonzero(nonzero_x >= midpoint_x)

        new_left_coeffs  = self.update_poly_fit_if_enough_points(good_left_points_x,  good_left_points_y,  total_num_left_points,  self.min_window_points_for_poly_fit)
        new_right_coeffs = self.update_poly_fit_if_enough_points(good_right_points_x, good_right_points_y, total_num_right_points, self.min_window_points_for_poly_fit)

        return new_left_coeffs, new_right_coeffs, len(good_left_points_x), len(good_right_points_x)

    def identify_lanes(self, birdseye_image):
        """Identify the most likely position of the lane lines on the provided
        birdseye image, annotating the image to show where we checked window-
        wise and where the final fit is currently. Make use of data from
        previous frame(s) if available, updating our best estimate of the
        current lane curves. Return annotated birdseye frame to visualise the
        detection"""

        # Initialise blank canvas for image we'll use to visualise where
        # we looked for the lines, in colour (hence 3rd dimension)
        self.searched_area_image = np.stack((birdseye_image, birdseye_image, birdseye_image), axis=-1)

        if self.left_count_until_reinit <= 0 or self.right_count_until_reinit <= 0:
            # We have not had a recent frame with good detections, or this is
            # our first time in, so start from scratch looking for lane positions
            (new_left_coeffs, new_right_coeffs, left_weight, right_weight) = self.find_lines_by_sliced_convolution(birdseye_image)
        else:
            # Base our search on lane positions we derived from recent frames
            (new_left_coeffs, new_right_coeffs, left_weight, right_weight) = self.find_lines_near_existing_fit(birdseye_image)

        # Only draw curves and update our coeffs if we made a new fit this time;
        # if so reset counters so we're good for the next few frames (in terms
        # of having a fit to work with)
        if (new_left_coeffs is not None):
            if self.left_count_until_reinit <= 0:
                self.left_poly_coeffs = new_left_coeffs
            else:
                self.left_poly_coeffs = first_order_filter(self.left_poly_coeffs, new_left_coeffs, self.poly_filter_in_const)
            self.left_count_until_reinit = self.reinit_count_on_good_detection
        else:
            self.left_count_until_reinit -= 1
            left_weight = 0

        if (new_right_coeffs is not None):
            if self.right_count_until_reinit <= 0:
                self.right_poly_coeffs = new_right_coeffs
            else:
                self.right_poly_coeffs = first_order_filter(self.right_poly_coeffs, new_right_coeffs, self.poly_filter_in_const)
            self.right_count_until_reinit = self.reinit_count_on_good_detection
        else:
            self.right_count_until_reinit -= 1
            right_weight = 0

        # Average together the polynomial fit coeffs to keep the best estimate
        # lines parallel, keeping mid-point of each line in its own place
        if len(self.left_poly_coeffs) == 3 and len (self.right_poly_coeffs) == 3:
            left_weight = np.sqrt(left_weight)
            right_weight = np.sqrt(right_weight)
            total_weight = left_weight + right_weight
            if (total_weight >= 1.0):
                mid_height_y = self.y_size / 2
                left_mid_offset = poly_result(mid_height_y, self.left_poly_coeffs)
                right_mid_offset = poly_result(mid_height_y, self.right_poly_coeffs)
                fit_coeff_0 = (left_weight * self.left_poly_coeffs[0] + right_weight * self.right_poly_coeffs[0]) / total_weight
                fit_coeff_1 = (left_weight * self.left_poly_coeffs[1] + right_weight * self.right_poly_coeffs[1]) / total_weight
                self.left_poly_coeffs[0]  = fit_coeff_0
                self.right_poly_coeffs[0] = fit_coeff_0
                self.left_poly_coeffs[1]  = fit_coeff_1
                self.right_poly_coeffs[1] = fit_coeff_1
                self.left_poly_coeffs[2]  -= (poly_result(mid_height_y, self.left_poly_coeffs) - left_mid_offset)
                self.right_poly_coeffs[2] -= (poly_result(mid_height_y, self.right_poly_coeffs) - right_mid_offset)

        self.draw_poly_fit(self.searched_area_image, self.left_poly_coeffs, new_left_coeffs is not None)
        self.draw_poly_fit(self.searched_area_image, self.right_poly_coeffs, new_right_coeffs is not None)
        
        return self.searched_area_image

    def convert_pixel_to_real_poly(self, pixel_poly_coeffs):
        """Given 2nd order polynomial coefficients fitting x(y) in the birdseye
        pixel space, transform to the coefficients of the same curve plotted
        on the real road in metres, where the origin is the camera position"""

        # We have to do a bit of maths here. Let:
        #   ymp = y metres per pixel
        #   y0m = metres from camera to nearest end of lines being fitted
        #   y0p = pixels from bottom of image to nearest end of lines being fitted
        #   x0m = metres from camera to centre of lane when vehicle centred (assume this is zero though)
        #   x0p = pixels from left edge of image to centre of lane in image (assume half image width)
        #
        # We have a polynomial in pixels relating xp (x in pixels) to yp (y in pixels) thus:
        #
        #   xp = c0p.yp^2 + c1p.yp + c2p
        #
        # where the coefficients in pixel space are c0p, c1p and c2p. We want the coeffs
        # in metre (real) space, c0m, c1m and c2m, which relate xm (x in metres) to ym (y in metres),
        # i.e.
        #
        #  xm = c0m.ym^2 + c1m.ym + c2m
        #
        # We have to watch out too because our pixel fit has y going positive towards the bottom
        # of the image (pixel coordinate), but in reality we want y to increase going further
        # away from the camera (up the image). We can relate the two coordinate systems:
        #
        #  xm = xmp(xp - x0p) + x0m   (but we'll ignore x0m assuming camera is central)
        #  ym = ymp(y0p - yp) + y0m   (flipped the subtraction because of inverting vertically)
        #
        # I started to do this algebraically but it got surprisingly messy. So as a cop-out,
        # like the tutorial code, let's just plot a few points in the old system, transform
        # them to the new system, and then fit a new polynomial!

        # So first work in pixels:
        plot_y_p = np.linspace(0, 100, num=5)
        plot_x_p = poly_result(plot_y_p, pixel_poly_coeffs)

        # Convert x in pixels to x in metres
        centre_x_pixels = self.x_size / 2
        x_m = self.x_metres_per_pixel * (plot_x_p - centre_x_pixels)

        # Convert y in pixels to y in metres
        y_m = self.y_metres_per_pixel * (self.y_bottom_baseline_pixels - plot_y_p) + self.y_metres_to_baseline

        # Extract new polynomial fit in metres space
        metres_poly_coeffs = np.polyfit(y_m, x_m, deg=2)
        return metres_poly_coeffs

    def calc_real_world_data(self):
        """Compute real-world curvature/position/angle data from polynomial
        fit already computed giving x (distance left from camera) as a function
        of y (distance away down road) in metres; smooth for display"""

        # As our process involved forcing the lines to be parallel, it shouldn't
        # matter whether we use the left or right line, but may as well average
        # results where we can.

        # Left or right offset of lane from camera is just last term in polynomial
        left_lane_offset_from_camera_metres = self.left_real_coeffs[2]
        right_lane_offset_from_camera_metres = self.right_real_coeffs[2]
        centre_lane_offset_from_camera_metres = (left_lane_offset_from_camera_metres + right_lane_offset_from_camera_metres) / 2.0
        self.lateral_displacement_metres = first_order_filter(self.lateral_displacement_metres, -centre_lane_offset_from_camera_metres, self.real_data_filter_const)
        width_metres = right_lane_offset_from_camera_metres - left_lane_offset_from_camera_metres
        self.lane_width_metres = first_order_filter(self.lane_width_metres, width_metres, self.real_data_filter_const)

        # Curvature using tutorial code
        y_eval = 0  # calculate curvature at camera position
        left_curverad = ((1 + (2*self.left_real_coeffs[0]*y_eval + self.left_real_coeffs[1])**2)**1.5) / np.absolute(2*self.left_real_coeffs[0])
        right_curverad = ((1 + (2*self.right_real_coeffs[0]*y_eval + self.right_real_coeffs[1])**2)**1.5) / np.absolute(2*self.right_real_coeffs[0])
        avg_curvature_radius = (left_curverad + right_curverad) / 2.0  # should be same anyway
        self.mean_curvature_metres = first_order_filter(self.mean_curvature_metres, avg_curvature_radius, self.real_data_filter_const)

        # More useful in a real car is steering angle. Assuming the camera is mounted
        # pretty close to the front wheels longitudinally, we just want the
        # angle between our polynomial fit and the straight-ahead direction
        # at y=0 (the origin of our fit in the y direction is where the camera is).
        # The angle we want is that of the tangent to our fit curve:
        #
        #    tan(angle) = dx / dy
        #
        #  We have our polynomial fit:
        #
        #    x = k0.y^2 + k1.y + k2
        #
        #  Differentiating,
        #
        #    dx/dy = 2.k0.y + k1
        #
        # Evaluating at y=0 just leaves us dx/dy = k1, so
        #
        #   angle = atan(k1)
        #
        # Or we can include the baseline 'y' to get the angle at the start of the lane fit.
        dx_by_dy = 2.0 * self.left_real_coeffs[0] * self.y_metres_to_baseline + self.left_real_coeffs[1] 
        angle_left = np.arctan(dx_by_dy)
        angle_right = np.arctan(self.right_real_coeffs[1]) # should be the same barring numerical errors
        mean_angle = (angle_left + angle_right) / 2.0
        # Convert to degrees for display/control system
        angle_degrees = mean_angle / np.pi * 180.0
        self.steering_angle_degrees = first_order_filter(self.steering_angle_degrees, angle_degrees, self.real_data_filter_const)

    def superimpose_lane_fit(self, birdseye_image, undistorted_image):
        """Paint the detected inter-lane area ahead onto the original-perspective
        image in a highlight colour"""

        # If we don't have a fit yet, we can't plot anything
        if len(self.left_poly_coeffs) != 3 or len(self.right_poly_coeffs) != 3:
            return undistorted_image

        # Based on tutorial code here
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(birdseye_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Get curves in birdseye pixel coordinates
        ploty = np.linspace(0, self.y_size-1, num=self.y_size)
        left_fitx = poly_result(ploty, self.left_poly_coeffs)
        right_fitx = poly_result(ploty, self.right_poly_coeffs)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.warp_from_birdseye_matrix, (undistorted_image.shape[1], undistorted_image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
        return result

    def superimpose_birdseye(self, birdseye_image, real_image):
        """Superimpose birdseye detection visualisation image, shrunk down,
        over the top left part of the real-world image, to show how the detection
        process is working"""

        shrunk_x_size = int(self.birdseye_pip_relative_size * self.x_size)
        shrunk_y_size = int(self.birdseye_pip_relative_size * self.y_size)
        shrunken_birdseye = cv2.resize(birdseye_image, (shrunk_x_size, shrunk_y_size), 
                                       fx=self.birdseye_pip_relative_size, fy=self.birdseye_pip_relative_size)

        real_image[0:shrunk_y_size, 0:shrunk_x_size] = shrunken_birdseye

        return real_image

    def superimpose_real_data(self, real_image):
        """Paint a panel on the top-right part of image with real-world
        data on lane curvature, steering angle etc"""

        # Make a dark (but not black) panel to ensure light text will show up OK
        panel_x_size = int(self.x_size * self.text_panel_relative_size)
        panel_y_size = int(self.y_size * self.text_panel_relative_size)
        # Position in top right
        panel_x_offset = self.x_size - panel_x_size
        panel_y_offset = 0
        real_image[panel_y_offset:panel_y_offset+panel_y_size, panel_x_offset:panel_x_offset+self.x_size, :] = \
           real_image[panel_y_offset:panel_y_offset+panel_y_size, panel_x_offset:panel_x_offset+self.x_size, :] / np.uint8(4)

        text_vertical_space_px = 60
        text_bottom_y = panel_y_offset + text_vertical_space_px # start near top
        colour = (127,127,255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        value = "%.2f" % self.lane_width_metres if self.lane_width_metres is not None else "--"
        cv2.putText(real_image, "Lane width = %s m" % value, (panel_x_offset,text_bottom_y), font, scale, colour)
        text_bottom_y += text_vertical_space_px
        value = "%.2f" % self.lateral_displacement_metres if self.lateral_displacement_metres is not None else "--"
        cv2.putText(real_image, "Lateral displacement = %s m" % value, (panel_x_offset,text_bottom_y), font, scale, colour)
        text_bottom_y += text_vertical_space_px
        value = "%d" % self.mean_curvature_metres if self.mean_curvature_metres is not None else "--"
        cv2.putText(real_image, "Lane curvature = %s m" % value, (panel_x_offset,text_bottom_y), font, scale, colour)
        text_bottom_y += text_vertical_space_px
        value = "%.2f" % self.steering_angle_degrees if self.steering_angle_degrees is not None else "--"
        cv2.putText(real_image, "Steering angle = %s deg" % value, (panel_x_offset,text_bottom_y), font, scale, colour)

        # Add ID/title
        text_bottom_y += text_vertical_space_px
        scale = 0.75
        text_vertical_space_px = 20
        cv2.putText(real_image, "charlie.wartnaby@idiada.com", (panel_x_offset,text_bottom_y), font, scale, colour)
        text_bottom_y += text_vertical_space_px
        cv2.putText(real_image, "Udacity self-driving car: advanced lane line project", (panel_x_offset,text_bottom_y), font, scale, colour)

        
        return real_image

    def process_next_image(self, raw_image):
        """ For one image/frame, do whole process going through to identified
        lane lines and return frame with colour and text annotation, making use
        of preceding frames if appropriate """

        # Start by correcting for camera distortion, if possible
        if self.camera_matrix is not None and self.distortion_coeffs is not None:
            undistorted_image = undistort_one_image(raw_image, self.camera_matrix, self.distortion_coeffs)
            #print("Debug: undistorted image OK")
        else:
            # Undistort skipped during development of other steps in pipeline
            undistorted_image = raw_image
            #print("Debug: skipped undistortion of image")

        # When we come to highlight lane lines, we will want to mask out areas
        # of the image that are not of interest before normalising intensity.
        # Otherwise, areas of the image that happen to give strong signals may
        # dominate the normalised version, to the detriment of the lines we
        # are after. The mask would always be the same in video, but not
        # necessarily for test images, so check if we already have one of the
        # right size
        if self.interest_region_mask is None or np.shape(self.interest_region_mask)[:2] != np.shape(undistorted_image)[:2]:
            self.interest_region_mask = create_interest_region_mask(undistorted_image)

        # It makes sense to do colour transformations first, so that the lane lines
        # stand out from the background, and only then check for gradients etc
        # (the gradient measurement should then work better).
        colour_trans_image = colour_trans_one_image(undistorted_image)

        # Now look for intensity gradient aiming in useful direction
        gradient_use_image = find_useful_gradient(colour_trans_image, self.interest_region_mask)

        # Apply intensity threshold to leave binary image. Processed image so far can have
        # quite weak signal for dim yellow lines (though distinct against near black background),
        # so low threshold pretty low:
        binary_image = apply_binary_threshold(gradient_use_image, 10, 255)

        # Project rubric requires examples of binary image, so save for later
        self.binary_image = binary_image

        # Apply perspective transform such that the image is recast as a bird's eye view.
        if self.warp_to_birdseye_matrix is None:
            # Avoid doing this every iteration to save CPU bandwidth, only compute first time
            self.compute_warp_matrices()
        birdseye_image = self.warp_image(binary_image)

        # Identify lanes and annotate birdseye image visually to show how we did it
        lanes_on_birdseye_image = self.identify_lanes(birdseye_image)

        # Compute polynomial coefficients for left and right lanes in terms of
        # metres from the camera, once we have them
        if len(self.left_poly_coeffs) == 3 and len(self.right_poly_coeffs) == 3:
            self.left_real_coeffs  = self.convert_pixel_to_real_poly(self.left_poly_coeffs )
            self.right_real_coeffs = self.convert_pixel_to_real_poly(self.right_poly_coeffs)

            # Now we have real-world polynomial fits, extract steering/position/curvature
            self.calc_real_world_data()

        # Paint the detected inter-lane area ahead on the original image as a highlight
        lanes_superimposed_image = self.superimpose_lane_fit(birdseye_image, undistorted_image)

        # To show how it is working, superimpose the birdseye image showing the
        # fitting process in the top-left corner of the frame
        birdseye_superimposed_image = self.superimpose_birdseye(lanes_on_birdseye_image, lanes_superimposed_image)

        # Display real-world numerical data in top right panel
        data_superimposed_image = self.superimpose_real_data(birdseye_superimposed_image)

        return data_superimposed_image

if __name__ == "__main__":
    # Validate command-line arguments and run appropriate action(s)
    import argparse
    parser = argparse.ArgumentParser()

    #-db DATABSE -u USERNAME -p PASSWORD -size 20
    parser.add_argument("-c", "--cam_cal_path_pattern", type=str, help="Path pattern to match camera calibration images"           )
    parser.add_argument("-x", "--cam_cal_nx",           type=int, help="Expected number of calibration image x-direction corners"  )
    parser.add_argument("-y", "--cam_cal_ny",           type=int, help="Expected number of calibration image x-direction corners"  )
    parser.add_argument("-e", "--cam_cal_eg_src",       type=str, help="Example image to process to demonstrate camera calibration")
    parser.add_argument("-f", "--cam_cal_eg_dst",       type=str, help="Output image to demonstrate camera calibration"            )
    parser.add_argument("-i", "--img_path_pattern",     type=str, help="Process single JPEG images matching this path pattern"     )
    parser.add_argument("-j", "--img_out_dir",          type=str, help="Output directory for processed single binary images"       )
    parser.add_argument("-v", "--video_in_file",        type=str, help="File path for input video to process"                      )
    parser.add_argument("-w", "--video_out_dir",        type=str, help="Output directory for processed video"                      )


    args = parser.parse_args()
    

    if args.cam_cal_path_pattern is None:
        print("Warning: no path provided for camera calibration images, distortion will not be corrected")
        camera_matrix = None
        distortion_coeffs = None
    elif args.cam_cal_nx is None or args.cam_cal_ny is None:
        print("Error: cannot calibrate camera without expected number of x and y corners (intersections) being specified")
        sys.exit(1)
    else:
        camera_matrix, distortion_coeffs = calibrate_camera_from_images(args.cam_cal_path_pattern, args.cam_cal_nx, args.cam_cal_ny)
 

    if args.cam_cal_eg_src is not None:
        # Load and undistort one (chessboard) image, saving output, as example of undistortion
        if args.cam_cal_eg_dst is None:
            print("Error: --cam_cal_eg_src option also needs --cam_cal_eg_dst for output")
            sys.exit(1)
        elif camera_matrix is None or distortion_coeffs is None:
            print("Error: cannot undistort example calibration image as no calibration was done")
            sys.exit(1)
        else:
            eg_image = cv2.imread(args.cam_cal_eg_src)
            undistorted_image = undistort_one_image(eg_image, camera_matrix, distortion_coeffs)
            cv2.imwrite(args.cam_cal_eg_dst, undistorted_image)
            print("Undistorted example %s output as %s" % (args.cam_cal_eg_src, args.cam_cal_eg_dst))

    # Lane-processing object will retain values from previous image(s) to aid lane detection
    # in successive images. It initialises itself with no previous image data to 'help':
    lane_processor = LaneProcessor(camera_matrix, distortion_coeffs)

    if args.img_path_pattern is not None:
        print("Processing single images matching %s" % args.img_path_pattern)
        if args.img_out_dir is None:
            print("Error: --img_path_pattern option also needs --img_out_dir for output")
            sys.exit(1)
        input_image_filenames = glob.glob(args.img_path_pattern)
        for image_filename in input_image_filenames:
            lane_processor.reset()
            single_image = cv2.imread(image_filename)
            root_part, filename = os.path.split(image_filename)
            #print("Debug: processing %s" % filename)
            lane_processor.process_next_image(single_image)
            output_path = os.path.join(args.img_out_dir, filename)
            cv2.imwrite(output_path, lane_processor.binary_image)
        print("%d processed single binary images written with same filenames to directory %s" % (len(input_image_filenames), args.img_out_dir))

    if args.video_in_file is not None:
        if args.video_out_dir is None:
            print("Error: --video_in_file option also needs --video_out_dir for output")
            sys.exit(1)
        root_part, filename = os.path.split(args.video_in_file)
        print("Processing video file: %s" % filename)
        input_clip = VideoFileClip(args.video_in_file) #.subclip(0,10)  # subclips used for speed in development/debug
        lane_processor.reset()
        processed_clip = input_clip.fl_image(lane_processor.process_next_image)
        output_path = os.path.join(args.video_out_dir, filename)
        processed_clip.write_videofile(output_path, audio=False)
        print("Processed video written to: %s" % output_path)
