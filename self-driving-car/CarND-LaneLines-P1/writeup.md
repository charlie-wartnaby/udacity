# **Finding Lane Lines on the Road** 

Student: Charlie Wartnaby, IDIADA UK Ltd

<charlie.wartnaby@idiada.com>

06 June 2017

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

I left the `draw_lines()` function much as it was (just making it robust to being passed an empty list of lines). I did not want it to be involved in processing the actual line vectors, because then it would
have needed many additional parameters such as the geometry of the original region-of-interest
quadrilateral.

Instead I created a new function `average_viable_lines()` to perform the line processing, which
worked only on line vectors and not bitmaps, which I felt was a better architectural
division. This new function does the following:
- Accepts the raw set of lines as input, together with gradient and region parameters.
- For each line, rejects any that are horizontal, vertical, or had zero length.
- For surviving lines, it keeps only those with a gradient that might be reasonable for a left or
  right lane line using the parameters provided, separating left from right.
- For all the lines categorised as left lane lines, the gradients and offsets are averaged,
  and similarly for all the right lines, giving just one average gradient and offset for the left
  and one for the right.
- The averages are weighted according to the length of the input lines, so long lines carry
  large weight and short segments carry only a little weight.
- A single line is then constructed using the averaged gradient and offset, which extends
  from the bottom to the top of the region of interest defined.
- Hence the final output is a pair of lines, one for the left and one for the right,
  extrapolated as required to extend fully across the region of interest.

#### General comments on approach

* I avoided using absolute numbers of pixels in the adjustable parameters in my pipeline,
  instead defining these in terms of proportions of the total image width or size. I hope that
  this would make the algorithms robust against changes in resolution (e.g. to handle 4K or
  VGA video equally well).
 
* I re-ordered the cells which work on test images in the notebook provided so that the
  whole notebook could be run successfully doing Run... All Cells. (As provided, some Python
  functions and imports were out of order, requiring manual execution in the correct order
  the first time.)

* I altered the code to create the output folders if required, so that it would run on
  a clean installation of the project.

* I tried to make the code robust against corner conditions such as no lines
  being detected in a frame, zero-length lines being included, and so-on.

* In FireFox I had a problem with one of the videos being truncated in its inline view,
  which I fixed by removing some of the size directives in the HTML.

* I took care to use meaningful identifiers and to comment code as clearly as possible
  to aid maintenance and review/assessment.

### 2. Identify potential shortcomings with your current pipeline

* Currently the region of interest quadrilateral assumes that the camera has a fixed
  focal length; for example, it would not work well with a fisheye lens.

* The region of interest also assumes the camera is directly forward-facing. It would
  not work well if the camera view was dipping above or below the horizon, or pointing
  to the left or right of centre.

* The 'challenge' video showed that strongly curved lines are not captured quite so
  well. A fitting process that allowed for curved lines to be matched directly, instead
  of only the straight lines allowed by Hough processing, might improve this.

### 3. Suggest possible improvements to your pipeline

Dealing with a different camera focal length or orientation might be solved by
allowing the system to experiment with different assumed orientations and perspectives,
ideally self-learning which appears to give the best match for a given camera
installation.

A more fundamental architectural problem is that each frame of the video is currently
treated individually. No use is made of the fact that a preceding or following frame
should have much the same lane lines, just moved a little on the image if at all.
Working across several frames in time, not just across the image in space, should make
for a more sensitive and robust detection system. Yet a real-time system could of course not make
use of future frames, so it would have to work with only on a few preceding frames.
