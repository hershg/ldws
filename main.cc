/*
 * Copyright 2016 Konsulko Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 */

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <string>

#include "config_store.h"
#include "fps.h"
#include "lane_detector.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[])
{
	ofstream myfile;
  	myfile.open ("timeperframe.txt", std::ios_base::app);

	// Get a config store and parse options
	ConfigStore *cs = ConfigStore::GetInstance();
	cs->ParseConfig(argc, argv);

	// Open video input file/device
	VideoCapture capture(cs->video_in);
	// If file open fails, try finding a camera indicated by an integer argument
	if (!capture.isOpened())
	{capture.open(atoi(cs->video_in.c_str()));}

	// Toggle OpenCL on/off
	if (!cs->cuda_enabled) {
		cv::ocl::setUseOpenCL(cs->opencl_enabled);

		if (cs->opencl_enabled) {
			cv::ocl::Context context;
			cv::ocl::Device(context.device(0)); //OpenCl accel with GPU
		}

		if (!cv::ocl::haveOpenCL()) {
			cout << "OpenCL is not available..." << endl;
			//return;
		}

		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_ALL)) {
			cout << "Failed creating the context..." << endl;
			//return;
		}

		cout << context.ndevices() << " Following devices are detected." << endl;
		//This bit provides an overview of the OpenCL devices you have in your computer

		for (int i = 0; i < context.ndevices(); i++) {
			cv::ocl::Device device = context.device(i);
			cout << "name:              " << device.name() << endl;
			cout << "ID:                " << i << endl;
			cout << "available:         " << device.available() << endl;
			cout << "imageSupport:      " << device.imageSupport() << endl;
			cout << "OpenCL_C_Version:  " << device.OpenCL_C_Version() << endl;
			cout << endl;
		}
	}

	string mode = "CPU";
	if (cs->cuda_enabled)
		mode = "CUDA GPU";
	else if (cs->opencl_enabled)
		mode = "FPGA";	// mode = "OpenCL GPU";
	cout << "Mode: " << mode << endl;
	myfile << mode << '\n';

	// Report video specs
	double width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	double height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
	char fourcc[] = {(char)(ex & 0XFF),(char)((ex & 0XFF00) >> 8),(char)((ex & 0XFF0000) >> 16),(char)((ex & 0XFF000000) >> 24),0};
	Size frame_size(static_cast<int>(width), static_cast<int>(height));
	cout << "Video: frame size " << width << "x" << height << ", codec " << fourcc << endl;

	// Create output window
	string window_name = "Full Video";
	if (cs->display_enabled) {
		namedWindow(window_name, CV_WINDOW_KEEPRATIO);
	}

	// FIXME this should be conditional
	VideoWriter output_writer(cs->video_out, CV_FOURCC('P','I','M','1'), 30, frame_size, true);

	Mat frame, edge;
	Mat temp = Mat(height, width, CV_8UC3);
	cv::cuda::GpuMat gpu_frame, gpu_gray, gpu_edge, gpu_lines;
	UMat u_frame, u_gray, u_edge;
	// FIXME need to error check for valid roi
	Rect roi_rect = Rect(cs->roi.x, cs->roi.y, cs->roi.w, cs->roi.h);
	cv::Ptr<cv::cuda::Filter> blur = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, Size(5, 5), 1.5);
	cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(cs->canny_min_thresh, cs->canny_max_thresh, 3, false);
	double rho = 1;
	double theta = CV_PI/180;
	vector<Vec4i> lines;
	cv::Ptr<cuda::HoughSegmentDetector> hough = cuda::createHoughSegmentDetector(rho, theta, cs->hough_min_length, cs->hough_max_gap);
	LaneDetector ld;

	frame_avg_init();

	while (true)
	{
		capture >> frame;
		if (frame.empty())
			break;

		// Display original frame
		if (cs->intermediate_display) {
			namedWindow("Original Video", WINDOW_AUTOSIZE);
			imshow("Original Video", frame);
		}

		frame_begin();

		if (cs->cuda_enabled) {
			// CUDA implementation
			gpu_frame.upload(frame);

			// Set ROI to reduce workload
			cv::cuda::GpuMat gpu_roi(gpu_frame, roi_rect);

			// Convert to grayscale and blur
			cv::cuda::cvtColor(gpu_roi, gpu_gray, CV_BGR2GRAY);
			blur->apply(gpu_gray, gpu_gray);

			// Canny edge detection
			canny->detect(gpu_gray, gpu_edge);

			// Probabilistic Hough line detection
			hough->detect(gpu_edge, gpu_lines);
			lines.resize(gpu_lines.cols);
			Mat temp(1, gpu_lines.cols, CV_32SC4, &lines[0]);
			gpu_lines.download(temp);
		} else {
			// TAPI implementation
			frame.copyTo(u_frame);

			// Set ROI to reduce workload
			UMat u_roi(u_frame, roi_rect);

			// Convert to grayscale and blur
			cvtColor(u_roi, u_gray, CV_BGR2GRAY);

            //setenv("OPENCV_OPENCL_DEVICE", ":ACCELERATOR:", 1);
            //cout << "OPENCV_OPENCL_DEVICE SET: " << getenv("OPENCV_OPENCL_DEVICE") << endl << endl;

			GaussianBlur(u_gray, u_gray, Size(5, 5), 1.5);

            //unsetenv("OPENCV_OPENCL_DEVICE");
            //cout << "OPENCV_OPENCL_DEVICE UNSET: " << getenv("OPENCV_OPENCL_DEVICE") << endl << endl;

			// Canny edge detection
			Canny(u_gray, u_edge, cs->canny_min_thresh, cs->canny_max_thresh);

			// Probabilistic Hough line detection
			HoughLinesP(u_edge, lines, rho, theta, cs->hough_thresh, cs->hough_min_length, cs->hough_max_gap);
		}

		if (cs->cuda_enabled) {
			gpu_edge.download(edge);
		} else {
			edge = u_edge.getMat(ACCESS_READ);
		}

		ld.ProcessLanes(lines, frame, edge, temp);

		// Release the reference taken in getMat()
		if (!cs->cuda_enabled)
			edge.release();

		frame_end();

		// Display Canny image
		if (cs->intermediate_display) {
			namedWindow("Edges");
			if (cs->cuda_enabled) {
				imshow("Edges", edge);
			} else {
				imshow("Edges", u_edge);
			}
		}

		// Display FPS
		putText(frame, "Mode: " + mode, Point(5, 30), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);
		putText(frame, "FPS: " + frame_fps_str(), Point(5,60), FONT_HERSHEY_SIMPLEX, 1., Scalar(255, 100, 0), 2);

		// Display full image
		if (cs->display_enabled)
			imshow(window_name, frame);

		// Write frame to output file
		if (cs->file_write)
			output_writer << frame;

		if (waitKey(1) == 27) break;
	}

	cout << "Average FPS: " << frame_fps_avg_str() << endl;
	cout << "Time per frame (ms): " << time_per_frame() << endl;
	//cout << "OpenCV version : " << CV_VERSION << endl;
  	//cout << "Major version : " << CV_MAJOR_VERSION << endl;
  	//cout << "Minor version : " << CV_MINOR_VERSION << endl;
  	//cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;
	myfile << time_per_frame() << "\n\n";
}
