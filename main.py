from math import ceil
import RPi.GPIO as GPIO
from time import sleep
import math

# import requests
import cv2
# import serial
import numpy as np
import time
start_time = time.time()

def pump(j):
    led_pin = 3
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pin, GPIO.OUT)
    if(j==0):
        GPIO.output(led_pin, GPIO.HIGH)
    else:
        GPIO.output(led_pin, GPIO.LOW)
        
def SetAngle(i,angle):
    
	GPIO.setmode(GPIO.BOARD)
	GPIO.setup(i, GPIO.OUT)
	pwm=GPIO.PWM(i, 50)
	pwm.start(0)
	duty = angle / 18 + 2
	GPIO.output(i, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(i, False)
	pwm.ChangeDutyCycle(0)
	pwm.stop()
	GPIO.cleanup()


        

class FireDetection:
    # value initialization label
    
        
        
    def __init__(self):
        self.subtractBackground = cv2.createBackgroundSubtractorKNN(
            detectShadows=False)
        # change the COM as per the auduino
        # self.sendSerial = serial.Serial('COM3', 9600)
        o_kern = ceil(3)
        c_kern = ceil(30)
        self.openKernel = np.ones((o_kern, o_kern), np.uint8)
        self.closeKernel = np.ones((c_kern, c_kern), np.uint8)
        self.xCentroid = None
        self.yCentroid = None
        self.prevRatio = None
        self.currRatio = None
        self.prevHist = None
        self.currHist = None
        self.prevHist1 = None
        self.prevHist2 = None
        self.prevHist3 = None
        self.prevHist4 = None
        self.prevHist5 = None

        self.prevVar = None
        self.currVar = None
        self.history = []
        self.countFireFrames = 0
        self.fps = 29.97  # 2000 frames
        self.totalFrame = 0
        self.totalFrameCount1 = 0
        self.totalFrameCount2 = 0
        self.totalFrameCount3 = 0
        self.totalFrameCount4 = 0
        self.totalFrameCount5 = 0
        self.CalcFrameCount1 = 0
        self.CalcFrameCount2 = 0
        self.CalcFrameCount3 = 0
        self.CalcFrameCount4 = 0
        self.CalcFrameCount5 = 0
        self.prevNoOfPixels = 1
        self.histogram_flag = False
        self.variance_flag = False

    def volume_analysis(self, threshold_image):
        gamma = 0.05
        flag = False
        no_of_pixels = cv2.compare(threshold_image, 0, cv2.CMP_GT).sum() / 255
        if self.prevNoOfPixels is not None:
            if abs(self.prevNoOfPixels - no_of_pixels) / self.prevNoOfPixels > gamma:
                flag = True
        self.prevNoOfPixels = no_of_pixels
        return flag

    def dimension_analysis(self, width, height):
        gamma = 0.1
        flag = False
        self.currRatio = width / height
        if self.prevRatio is not None:
            if abs(self.prevRatio - self.currRatio) > gamma:
                flag = True
        self.prevRatio = self.currRatio
        return flag

    def background_subtraction(self, capture_image):
        mask = self.subtractBackground.apply(capture_image)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.openKernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closeKernel)
        threshold = cv2.medianBlur(closing, 5)
        foreground = cv2.bitwise_and(
            capture_image, cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))
        return threshold, foreground

    def color_analysis_org(self, capture_image, threshold_image):
        div = 210
        mul = 1.3
        self.totalFrameCount1 += 1
        img_ycrcb = cv2.cvtColor(capture_image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)

        y_mean = y.mean()
        cr_mean = cr.mean()
        cb_mean = cb.mean()

        cr_mean = cr_mean if cr_mean > 150 else 150
        #print("cr_mean", cr_mean)
        #cb_mean = 120 if cb_mean > 120 else cb_mean
        #print("cb_mean", cb_mean)
        # if cb_mean == 120:
        # print("No Fire Detected with Color Analysis")
       # else:
        #self.CalcFrameCount1 += 1
       # print('ymean',y_mean)
        y_mean = y_mean if div < y_mean else div

        #_, y_mat = cv2.threshold(y, div, 255, cv2.THRESH_TOZERO_INV)
        _, y_mat = cv2.threshold(y, y_mean, 255, cv2.THRESH_TOZERO)

        y_mat = cv2.compare(y_mat, cb, cv2.CMP_GT)
        ##cv2.imshow("y>Cb", y_mat)
        cr_mat = cv2.compare(cr, cb, cv2.CMP_GE)
        cb_mat = cv2.compare(cb, cb_mean, cv2.CMP_LE)

        light_fire = cv2.bitwise_and(y_mat, cr_mat)
        #light_fire = cv2.bitwise_and(light_fire, cb_mat)

        y_mean = y_mean if div < y_mean else div

        y_mat = cv2.compare(y, y_mean, cv2.CMP_GT)
        cr_mat = cv2.compare(cr, cb, cv2.CMP_GE)

        heavy_fire = cv2.bitwise_and(y_mat, cr_mat)
        fire = cv2.bitwise_or(light_fire, heavy_fire)
        fire = cv2.bitwise_and(fire, threshold_image)

        opening = cv2.morphologyEx(fire, cv2.MORPH_OPEN, self.openKernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.closeKernel)

        closing = cv2.medianBlur(closing, 5)
        
        return_img = cv2.bitwise_and(
            capture_image, cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR))

        return return_img
    
    def fire_blob_detection(self, capture_image, fire_pixel_image, threshold_image):
        print("fire frame", self.CalcFrameCount5)
        print('total frame', self.totalFrame)
        min_pixel_length = 10
        self.totalFrame += 1
        self.totalFrameCount2 += 1
        self.totalFrameCount3 += 1
        self.totalFrameCount4 += 1
        self.totalFrameCount5 += 1
        barrier_frame_limit_hist = 2
        barrier_frame_limit_var = 4
        cam_input = capture_image.copy()
        interval = 10
        if self.totalFrame % interval == 0:
            self.alarm_decision(False)
            self.prevVar = None
            self.prevHist = None
            self.xCentroid = None
        contours, _ = cv2.findContours(cv2.cvtColor(fire_pixel_image, cv2.COLOR_BGR2GRAY),
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
            x_coord, y_coord, width, height = cv2.boundingRect(contours[0])

###################### Color Analysis ###################################

            if width > min_pixel_length and height > min_pixel_length:
                self.CalcFrameCount1 += 1
                ExecTime1 = (time.time()-start_time)
                print("FC for Color Analysis:  ", self.CalcFrameCount1)
                print("Exec time for Color Analysis:  ", ExecTime1, 'secs')
                print("Fire Detected with Color Analysis")

###################### Centroid Analysis ###############################

                if self.centroid_analysis(x_coord, y_coord, width, height):

                    self.CalcFrameCount2 += 1
                    ExecTime2 = self.totalFrameCount2 / self.fps
                    print(" FC for Centroid Analysis : ", self.CalcFrameCount2)
                    print('Exec time for Centroid:  ',
                          (time.time()-start_time), 'secs')
                    print('Probable Fire with centroid true')

###################### Histogram #########################################

                    if self.countFireFrames < barrier_frame_limit_hist:
                        self.histogram_flag = self.histogram_analysis(
                            capture_image, threshold_image)
                    else:
                        self.histogram_flag = True

                    if self.histogram_flag:

                        self.CalcFrameCount3 += 1
                        ExecTime3 = (time.time()-start_time)
                        print("FC for Hist Analysis:  ", self.CalcFrameCount3)
                        print('Exec time for Histogram:  ', ExecTime3, 'secs')
                        print('Probable Fire with histogram true')

######################### Variance ###########################################
                        if self.countFireFrames < barrier_frame_limit_var:
                            self.variance_flag = self.variance_analysis(fire_pixel_image[y_coord:y_coord + height,x_coord:x_coord + width])
                            #self.variance_flag = True
                        else:
                            self.variance_flag = True

                        if self.variance_flag:

                            self.CalcFrameCount4 += 1
                            ExecTime4 = (time.time()-start_time)
                            print("FC for Variance Analysis:  ",
                                  self.CalcFrameCount4)
                            print('Exec time for Variance:  ',
                                  ExecTime4, 'secs')
                            print('Probable Fire with variance true')
                            cam_input = cv2.rectangle(cam_input,
                                                      (x_coord, y_coord),
                                                      ((x_coord + width),
                                                       (y_coord + height)),
                                                      (255, 255, 255),
                                                      2)

############################### Alarm ############################################
                            if self.alarm_decision(True):
                                print("Alarm done",time.time())
                                self.CalcFrameCount5 += 1
                                ExecTime5 = (time.time()-start_time)
                                print("FC for Alarm Analysis:  ",
                                      self.CalcFrameCount5)
                                print('Exec time for Alarm:  ',
                                      ExecTime5, 'secs')
                                print('Fire Alert! {} xFrames / {} Total Frames--------------'.format(self.CalcFrameCount5,
                                                                                                      self.totalFrame))
                                # self.sendSerial.write(str.encode('f'))
                                cam_input = cv2.rectangle(cam_input,
                                                          (x_coord, y_coord),
                                                          ((x_coord + width),
                                                           (y_coord + height)),
                                                          (0, 0, 255),
                                                          2)
                                x1 = x_coord + width
                                y1 = y_coord + height
                                area = height * width
                                cv2.drawContours(
                                    cam_input, contours[0], -1, (0, 0, 255), 2)
                                print('x = ', x_coord, 'y = ', y_coord, 'x1 = ', x_coord + width, 'y1 = ',
                                      y_coord + height, 'width = ', width, 'ht = ', height, 'area = ', area)
                                deltax=(x_coord)
                                deltay=(y_coord)
                                thetabase=math.atan2(deltay,deltax)
                                print("thetabase")
                                thetabase=thetabase*180/(3.14)
                                print(thetabase)
                                SetAngle(7,thetabase)
                                if (x_coord < 480 and y_coord < 270):
                                    
                                    #self.sendSerial.write(str.encode('a'))
                                    print('Sprinklers Activated')
                                    
                                elif (x_coord > 480 and y_coord < 270):
                                    #self.sendSerial.write(str.encode('b'))
                                    print('Sprinklers Activated')
                                elif (x_coord > 480 and y_coord > 270):
                                    #self.sendSerial.write(str.encode('c'))
                                    print('Sprinklers Activated')
                                elif (x_coord < 480 and y_coord > 270):
                                    #self.sendSerial.write(str.encode('d'))
                                    print('Sprinklers Activated')

                                print('Area Safe from alarm decision')
                                #self.sendSerial.write(str.encode('s'))
                        else:
                            print('Area Safe from Variance Analysis')
                            #self.sendSerial.write(str.encode('s'))
                    else:
                        print('Area Safe from Histogram Analysis')
                        #self.sendSerial.write(str.encode('s'))
            else:
                print('Area Safe from Centroid Analysis')
                #self.sendSerial.write(str.encode('s'))
        else:
            print('Area Safe')
            # self.sendSerial.write(str.encode('s'))
        return cam_input

    def centroid_analysis(self, x_coord, y_coord, width, height):
        range_limit = 30
        min_pixel_length = 5
        flag = False
        x_center = x_coord + width / 2
        y_center = y_coord + height / 2
        if self.xCentroid is not None:
            if abs(x_center - self.xCentroid) < range_limit and abs(y_center - self.yCentroid) < range_limit:

                # if width > min_pixel_length and height > min_pixel_length:
                flag = True
        self.xCentroid = x_center
        self.yCentroid = y_center
        # print(flag)
        return flag

    def histogram_analysis(self, capture_image, threshold_image_mask):
        omega = .8
        flag = False
        self.currHist = cv2.calcHist([capture_image],
                                     [0, 1, 2],
                                     threshold_image_mask,
                                     [256, 256, 256],
                                     [0, 255, 0, 255, 0, 255])
        self.currHist = cv2.normalize(self.currHist, None).flatten()
        if self.prevHist2 is not None:
            correl = cv2.compareHist(
                self.currHist, self.prevHist2, cv2.HISTCMP_CORREL)
            print(correl)

            if correl < omega:
                flag = True

        self.prevHist2 = self.prevHist
        self.prevHist1 = self.prevHist
        self.prevHist = self.currHist
        return flag

    def variance_analysis(self, image):
        sigma = 50
        flag = False
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        comp_img = cv2.compare(image, 0, cv2.CMP_GT)
        ones = comp_img.sum() / 255
        total = comp_img.size
        var = np.var(cr)
        new_var = ((total * var) - ((total - ones) *
                   abs(128-cr.mean()) * abs(128-cr.mean()))) / ones
        self.currVar = new_var
        print(new_var)
        if self.prevVar is not None:
            if (self.currVar > sigma):
                flag = True
        self.prevVar = self.currVar
        return flag

    def alarm_decision(self, flag):
        history_storage_limit = 10
        min_fire_frames = 6
        alert = False
        if flag:
            self.history.append(1)
        else:
            self.history.append(0)
        if sum(self.history) > min_fire_frames:
            self.countFireFrames += 1
            alert = True
        else:
            self.countFireFrames = 0
        if len(self.history) > history_storage_limit:
            self.history = self.history[1:]
        return alert


# main part
if __name__ == '__main__':
    Fire = FireDetection()
    url = "http://25.59.12.198:8080/video"
    # cap = cv2.VideoCapture('E:/Fire1/Movie 1.mp4') #uncomment it if want to use downloaded video
    cap = cv2.VideoCapture(0)  # for video through webcam
    length = 406
    breadth = 720
    m = length / 2
    n = breadth / 2
    int_m = int(m)
    int_n = int(n)
    while True:
        grabbed, imBGR = cap.read()
        if not grabbed:
            break
        factor = 2
        # zone1 = cv2.rectangle(imBGR,(0,0),(int_m,int_n),(0,0,255),3) # Quad 1 red
        # zone2 = cv2.rectangle(imBGR,(0,int_n),(int_m,breadth),(0,255,255),3) # Quad 2 yellow
        # zone3 = cv2.rectangle(imBGR,(int_m,0),(length,int_n),(255,255,255),3) # Quad 3 white
        # zone4 = cv2.rectangle(imBGR,(int_m,int_n),(length,breadth),(255,255,0),3) # Quad 4 blue
        print(imBGR.shape[1], imBGR.shape[0])
        imBGR = cv2.resize(
            imBGR, (ceil(imBGR.shape[1] / factor), ceil(imBGR.shape[0] / factor)))
        print(imBGR.shape[1], imBGR.shape[0])
        thresholdImage, foregroundImage = Fire.background_subtraction(imBGR)
        firePixelImage = Fire.color_analysis_org(imBGR, thresholdImage)
        ##cv2.imshow('color_segment', firePixelImage)
        fireBlob = Fire.fire_blob_detection(
           imBGR, firePixelImage, thresholdImage)

        print("Fresh Analysis>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        ##cv2.imshow('OUTPUT', fireBlob)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
