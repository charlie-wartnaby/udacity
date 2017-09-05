REM Camera calibration:
REM --cam_cal_path_pattern camera_cal/calibration*.jpg --cam_cal_nx 9 --cam_cal_ny 6 --cam_cal_eg_src camera_cal/calibration1.jpg --cam_cal_eg_dst output_images/undistorted_calibration1.jpg
REM Single image processing:
REM --img_path_pattern test_images/*.jpg --img_out_dir output_images
REM Video processing:     project_video.mp4 harder_challenge_video.mp4
REM --video_in_file project_video.mp4 --video_out_dir output
python find_lane_lines.py --cam_cal_path_pattern camera_cal/calibration*.jpg --cam_cal_nx 9 --cam_cal_ny 6 --video_in_file project_video.mp4 --video_out_dir output_images