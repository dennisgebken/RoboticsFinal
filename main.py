# Written by Dennis Gebken
# Some parts of the code are written with help from https://www.pyimagesearch.com
# and http://opencv-python-tutroals.readthedocs.io. There may be similarities.
# This script is able to identify fruits and vegetables from a webcam capture.
# the webcam needs to be in a fixed position for it to work properly.
# After starting, leave the background empty and press 'b' to calibrate.
# Then place a fruit or vegetable in the background and press 'c' to identify.
# Output is given in the command prompt window.
# Press 'z' to exit the program.

# Currently only works on single Bananas, Oranges, Cucumbers, and Tomatoes.


import numpy as np
import cv2

#Blur factor for the gaussian filter.
BLUR_FACTOR = 7
calibrated = False

#Threshold used to set the minimum black-level to be included in the binary
#image.
THRESHOLD = 15

#Databases with recorded values for sample fruits.
dbLong = {"Banana": (6.0, 26.0, 36.0, 0), "Cucumber": (12.0, 39.0, 10.4, 0.0)}
dbRound = {"Tomato": (5.0, 5.6, 56.0, 0.0), "Orange": (4.6, 15.7, 59.6, 0.0)}


#Grabs a capture of the background and blurs it with gaussian filter.
def calibrate(frame):
	background = cv2.GaussianBlur(frame, (BLUR_FACTOR,BLUR_FACTOR), 0)
	global calibrated
	calibrated = True
	return background

# Grabs a capture with the fruit/produce and starts the processing procedure
def capture(background, frame):
	if not calibrated:
		print "Capture empty background first by pressing 'b'."
		return
	process_image(background, frame)

# Create binary image from diffed image
def get_cutout(diff):
	_,thresh = cv2.threshold(diff, THRESHOLD, 255, 0)
	return thresh

#Create contours from binary image.
def get_contours(cutout):
	_,contours,_ = cv2.findContours(cutout.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return contours

#Use contours to create bounding rectangle. Return rectangle and its width/height
def get_bounding_rect(contours):
	rect = cv2.minAreaRect(contours[-1])
	box = cv2.boxPoints(rect)
	w = max(rect[1])
	h = min(rect[1])
	return np.int0(box), w, h

# Create mask to be able to get mean color from within these boundaries.
# Shape refers to the image size used for webcam captures.
def get_mask(shape, contours):
	mask = np.zeros(shape, np.uint8)
	mask = cv2.drawContours(mask, contours[-1], -1, 255, -1)
	return mask

#Check one of the databases (long or round fruit) and get the closest color match
# Then print the result to the console.
def checkDB(db, color):
	minDiff = 9999.9
	result = None
	for entry in db:
		meanDiff = abs(np.mean(np.subtract(db[entry],color)))
		if meanDiff < minDiff:
			result = entry
			minDiff = meanDiff
	percentage = (np.mean(color) - np.mean(db[result])) / np.mean(color)
	percentage = int(100 * (1-abs(percentage)))
	print "Item: "+result+"."

# Run all processing functions on the captured frame and background.
# Uncomment last line to see images at various stages during the process.
def process_image(background, frame):
	capture = cv2.GaussianBlur(frame, (BLUR_FACTOR,BLUR_FACTOR), 0)
	diff = cv2.absdiff(capture, background)
	diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	cutout = get_cutout(diff)
	contours = get_contours(cutout)
	if contours:
		boundingRect, w, h = get_bounding_rect(contours)
		mask = get_mask(diff.shape, contours)
		if h > 0: aspectRatio = (w/h)
		else: aspectRatio = 999.9
		if aspectRatio > 1.3: db = dbLong
		else: db = dbRound
		meanColor = cv2.mean(frame, mask)
		result = checkDB(db, meanColor)
		#show_process(frame, diff, cutout, contours, mask, boundingRect, meanColor)


#If called, shows images of some intermediate steps to make it easier to adjust
#some values if needed.
def show_process(frame, diff, cutout, contours, mask, boundingRect, meanColor):
	cv2.imshow("Diff Background/Capture", diff)
	cv2.imshow("Cutout", cutout)
	cv2.imshow("Mask", mask)
	result = cv2.drawContours(frame, contours, -1, (0,255,0), 3)
	result = cv2.drawContours(result, [boundingRect], 0, (0,0,255), 3)
	cv2.imshow("Result", result)
	color = np.ones([432,576,3],'uint8')
	color[:,:] = meanColor[:-1]


if __name__ == "__main__":
	cam = cv2.VideoCapture(0)
	while(True):
		_,frame = cam.read()
		frame = cv2.resize(frame, (0,0), fx=0.9,fy=0.9)
		cv2.imshow("Feed",frame)
		keyPress = cv2.waitKey(1)

		if keyPress & 0xFF == ord('b'):
			background = calibrate(frame)

		elif keyPress & 0xFF == ord('c'):
			capture(background, frame)

		elif keyPress & 0xFF == ord('z'):
			break

	cam.release()
	cv2.destroyAllWindows()
