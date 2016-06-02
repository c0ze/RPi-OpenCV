#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <dirent.h>

using namespace cv;
using namespace std;

vector<Mat> images;
vector<int> labels;

DIR *dir;
char filename[256];
Mat image;
Mat image_resized;

const int IM_WIDTH = 70;
const int IM_HEIGHT = 70;

int load_images(){
  // These vectors hold the images and corresponding labels:
  struct dirent *ent;
  if ((dir = opendir ("faces")) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if (!(strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)) {
	printf ("loading... %s\n", ent->d_name);

	sprintf(filename, "faces/%s", ent->d_name);
	printf ("loading... %s\n", filename);
	image = imread(filename, 0);
	cv::resize(image, image_resized, Size(IM_WIDTH, IM_HEIGHT), 1.0, 1.0, INTER_CUBIC);
	images.push_back(image_resized);
	labels.push_back(1);
      }
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("");
    return EXIT_FAILURE;
  }
  return 0;
}

Ptr<FaceRecognizer> model;

void train_model(){

  // Create a FaceRecognizer and train it on the given images:
  model = createFisherFaceRecognizer();
  model->train(images, labels);
}

void check_pictures(){
  // These vectors hold the images and corresponding labels:
  struct dirent *ent;
  if ((dir = opendir ("faces2")) != NULL) {
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      if (!(strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)) {
	sprintf(filename, "faces2/%s", ent->d_name);
	image = imread(filename, 0);
	cv::resize(image, image_resized, Size(IM_WIDTH, IM_HEIGHT), 1.0, 1.0, INTER_CUBIC);
	// Now perform the prediction, see how easy that is:
	int prediction = model->predict(image_resized);
	printf ("file : %s prediction : %d\n", filename, prediction);
      }
    }
    closedir (dir);
  } else {
    /* could not open directory */
    perror ("could not open directory");
  }
}

int main(int argc, const char *argv[]) {

  int err = load_images(); 
  if (err != 0) {
    return err;
  }

  train_model();
  check_pictures();
  return 0;
}
