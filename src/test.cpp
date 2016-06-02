//Code to check the OpenCV installation on Raspberry Pi and measure frame rate of frames captured by the OpenCV - RPi wrapper

//Author: Samarth Manoj Brahmbhatt, University of Pennsylvania, with help of low-level code written by Tasanakorn

#include "cap.h"
#include <ctime>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const int BUFFER_SIZE = 256;

void put_time(char buffer[]){
  time_t now = time(NULL);
  strftime(buffer, BUFFER_SIZE, "%Y%m%d%H%M%S", localtime(&now));
}

int main() {
  //    namedWindow("Video");

    // Create capture object, similar to VideoCapture
    // PiCapture(width, height, color_flag);
    // color_flag = true  => color images are captured,
    // color_flag = false => greyscale images are captured
    PiCapture cap(320, 240, false);

    Mat im, flipped, cropped;
    double tickcount = 0;
    unsigned int frames = 0;

    char buffer[BUFFER_SIZE];
    char file_name[BUFFER_SIZE];
    
    String face_cascade_name = "./haarcascade_frontalface_default.xml";
    CascadeClassifier face_cascade;
    if( !face_cascade.load( face_cascade_name ) ){ cout << "--(!)Error loading face cascade\n"; return -1; };
    
    cout << "Press 'q' to quit" << endl;
    while(char(waitKey(1)) != 'q') {
        double t0 = getTickCount();
        im = cap.grab();
        frames++;
        if(!im.empty()) {
	  flip(im, flipped, 0);
	  vector<Rect> faces;
	  face_cascade.detectMultiScale( im, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(30, 30) );
	  if (faces.size() > 0) {
	    put_time(buffer);
	    cout << buffer;
	    cout << " SUCCESS" << endl;
	    cout << faces.size() << endl;
	    cout << faces[0].x << endl;

	    for(int i = 0; i < faces.size(); i++){
	      sprintf(file_name, "faces/%s_%i.jpg", buffer, i);
	      imwrite(file_name, im(faces[i]));
	    }
		
	    rectangle(im, faces[0], 255);
	  }
	  //	  imshow("Flipped", flipped);
	  //	  imshow("Flipped", im);
	}
        else cout << "Frame dropped" << endl;

	//        tickcount += (getTickCount() - t0) / getTickFrequency();
	//        cout << frames / tickcount << " fps" << endl;
    }

    return 0;
}
