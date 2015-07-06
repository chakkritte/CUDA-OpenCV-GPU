#define TEST_IMAGE "MyImages\\test1.jpg"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <iomanip>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/core/ocl.hpp"

#include "tick_meter.hpp"
#define haarcascades "haarcascades/haarcascade_frontalface_alt2.xml"
#define haarcascades_cuda "haarcascades_cuda/haarcascade_frontalface_alt.xml"


using namespace std;
using namespace cv;
using namespace cv::cuda;

/* Function Headers */
void detectAndDisplay(Mat frame);
/* Global variables */
String face_cascade_name = haarcascades;
//String eyes_cascade_name = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";
cv::CascadeClassifier face_cascade;
//cv::CascadeClassifier eyes_cascade;

//Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(haarcascades_cuda);

String window_name = "Capture - Face detection";

double TakeTime;
int64 Atime, Btime;

static void convertAndResize(const Mat& src, Mat& gray, Mat& resized, double scale)
{
	if (src.channels() == 3)
	{
		cv::cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = src;
	}

	Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

	if (scale != 1)
	{
		cv::resize(gray, resized, sz);
	}
	else
	{
		resized = gray;
	}
}

static void convertAndResize(const GpuMat& src, GpuMat& gray, GpuMat& resized, double scale)
{
	if (src.channels() == 3)
	{
		cv::cuda::cvtColor(src, gray, COLOR_BGR2GRAY);
	}
	else
	{
		gray = src;
	}

	Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

	if (scale != 1)
	{
		cv::cuda::resize(gray, resized, sz);
	}
	else
	{
		resized = gray;
	}
}


int detectAndDisplay(Mat image, bool useGPU , string name)
{

	Ptr<cuda::CascadeClassifier> cascade_gpu = cuda::CascadeClassifier::create(haarcascades_cuda);

	cv::CascadeClassifier cascade_cpu;
	if (!cascade_cpu.load(haarcascades_cuda))
	{
		return cerr << "ERROR: Could not load cascade classifier \"" << haarcascades << "\"" << endl, -1;
	}

	Mat frame, frame_cpu, gray_cpu, resized_cpu, frameDisp;
	vector<Rect> faces;

	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

	//facesBuf_gpu.create(1, 10000, cv::DataType<cv::Rect>::type);

	const int defaultObjSearchNum = 100000;
	if (facesBuf_gpu.empty())
	{
		facesBuf_gpu.create(1, defaultObjSearchNum, DataType<Rect>::type);
	}
	if (resized_gpu.empty())
	{
		resized_gpu.create(1, defaultObjSearchNum, DataType<Rect>::type);
	}


	/* parameters */
	double scaleFactor = 1.0;
	bool findLargestObject = false;
	bool filterRects = true;

	(image.empty() ? frame : image).copyTo(frame_cpu);
	frame_gpu.upload(image.empty() ? frame : image);

	convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
	convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);

	Atime = getTickCount();

	if (useGPU)
	{
		cascade_gpu->setFindLargestObject(findLargestObject);
		cascade_gpu->setScaleFactor(1.1);
		cascade_gpu->setMinNeighbors((filterRects || findLargestObject) ? 2 : 0);
		cascade_gpu->detectMultiScale(resized_gpu, facesBuf_gpu);
		cascade_gpu->convert(facesBuf_gpu, faces);
	}
	else
	{

		Size minSize = cascade_gpu->getClassifierSize();
		cascade_cpu.detectMultiScale(resized_cpu, faces, 1.1,
			(filterRects || findLargestObject) ? 2 : 0,
			(findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
			| CASCADE_SCALE_IMAGE,
			minSize);
	}

	Btime = getTickCount();

	for (size_t i = 0; i < faces.size(); ++i)
	{
		if (useGPU)
		{
			rectangle(image, faces[i], CV_RGB(255, 0, 0), 2);
		}
		else
		{
			rectangle(image, faces[i], CV_RGB(0, 0, 255), 2);
		}
	}
	
	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(%s version) = %d / %lf sec take.\n", name.c_str(), faces.size(), TakeTime);

	//printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), tm.getTimeSec());
	/*
	//print detections to console
	cout << setfill(' ') << setprecision(2);
	cout << setw(6) << fixed << fps << " FPS, " << faces.size() << " det";
	if ((filterRects || findLargestObject) && !faces.empty())
	{
		for (size_t i = 0; i < faces.size(); ++i)
		{
			cout << ", [" << setw(4) << faces[i].x
				<< ", " << setw(4) << faces[i].y
				<< ", " << setw(4) << faces[i].width
				<< ", " << setw(4) << faces[i].height << "]";
		}
	}
	cout << endl;
	*/

	//cv::cvtColor(resized_cpu, frameDisp, COLOR_GRAY2BGR);
	//displayState(frameDisp, helpScreen, useGPU, findLargestObject, filterRects, fps);
	//imshow(name, image);
	imwrite(name + ".jpg", image);
	return 0;

}


int main(){

	if (getCudaEnabledDeviceCount() == 0)
	{
		return cerr << "No GPU found or the library is compiled without CUDA support" << endl, -1;
	}

	cout << cv::getBuildInformation() << endl;

	//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
	//cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());

	//cv::ocl::setUseOpenCL(true);

	/*
	if (!cv::ocl::haveOpenCL())
	{
		cout << "OpenCL is not avaiable..." << endl;
		return -1;
	}
	cv::ocl::Context context;
	if (!context.create(cv::ocl::Device::TYPE_GPU))
	{
		cout << "Failed creating the context..." << endl;
		return -1;
	}

	// In OpenCV 3.0.0 beta, only a single device is detected.
	cout << context.ndevices() << " GPU devices are detected." << endl;
	for (int i = 0; i < context.ndevices(); i++)
	{
		cv::ocl::Device device = context.device(i);
		cout << "name                 : " << device.name() << endl;
		cout << "available            : " << device.available() << endl;
		cout << "imageSupport         : " << device.imageSupport() << endl;
		cout << "OpenCL_C_Version     : " << device.OpenCL_C_Version() << endl;
		cout << "localMemSize     : " << device.localMemSize() << endl;
		cout << endl;
	}

	// http://stackoverflow.com/questions/28529458/how-to-launch-custom-opencl-kernel-in-opencv-3-0-0-ocl

	// Select the first device
	cv::ocl::Device(context.device(0));

	*/
	
	Mat frame;
	//-- 1. Load the cascades
	//if (!face_cascade.load(face_cascade_name)){ printf("--(!)CPU Error loading\n"); return -1; };
	//if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading eyes cascade\n"); return -1; };
	Mat img = imread(TEST_IMAGE);
	Mat imgCPU = img.clone();
	Mat imgGPUOpenCL = img.clone();
	Mat imgGPUCUDA = img.clone();


	detectAndDisplay(imgCPU, false, "CPU");

	
	try
	{
		detectAndDisplay(imgGPUCUDA, true, "CUDA");
	}
	catch (const cv::Exception& ex)
	{
		std::cout << "Error: " << ex.what() << std::endl;
	}

	
	
	waitKey(0);
	return 0;
}
