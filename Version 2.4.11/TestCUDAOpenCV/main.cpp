#define BUILDALL true
#define RUNCPU true
#define RUNOCL true
#define haarcascade_DIR "haarcascades\\haarcascade_frontalface_alt2.xml"
#define haarcascade_DIR_CUDA "haarcascades_cuda\\haarcascade_frontalface_alt.xml"
#define TEST_IMAGE "MyImages\\test1.jpg"
#define SIZE 20

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ocl/ocl.hpp"
#include <stdio.h>
#include <vector>
#include <iomanip>
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2411d.lib")
#pragma comment(lib, "opencv_imgproc2411d.lib")   //MAT processing
#pragma comment(lib, "opencv_objdetect2411d.lib") //HOGDescriptor
#pragma comment(lib, "opencv_gpu2411d.lib")
#pragma comment(lib, "opencv_highgui2411d.lib")
#else
#pragma comment(lib, "opencv_core2411.lib")
#pragma comment(lib, "opencv_imgproc2411.lib")
#pragma comment(lib, "opencv_objdetect2411.lib")
#pragma comment(lib, "opencv_gpu2411.lib")
#pragma comment(lib, "opencv_highgui2411.lib")
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::ocl;


/** Global variables */
String face_cascade_name = haarcascade_DIR_CUDA;
CascadeClassifier face_cascade;

//CUDA
CascadeClassifier_GPU face_cascade_gpu;
String face_cascade_name_CUDA = haarcascade_DIR_CUDA;

//OCL
OclCascadeClassifier cascade_ocl;
CascadeClassifier  nestedCascade;

//CascadeClassifier eyes_cascade;
string window_name = "CPU - Face detection";
string window_namegpu = "GPU OpenCL- Face detection";
string window_namegpu_cuda = "GPU CUDA- Face detection";
RNG rng(12345);

double TakeTime;
int64 Atime, Btime;


template<class T>
void convertAndResize(const T& src, T& gray, T& resized, double scale)
{
	if (src.channels() == 3)
	{
		cvtColor(src, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = src;
	}

	Size sz(cvRound(gray.cols * scale), cvRound(gray.rows * scale));

	if (scale != 1)
	{
		resize(gray, resized, sz);
	}
	else
	{
		resized = gray;
	}
}


struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };

/** @function detectAndDisplay */
void detectAndDisplayCPU(Mat& frame)
{
	vector<Rect> faces;
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	Atime = getTickCount();

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(SIZE, SIZE), Size(0, 0));

	Btime = getTickCount();
	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(cpu version) = %d / %lf sec take.\n", faces.size(), TakeTime);

	if (faces.size() >= 1)
	{
		for (int ji = 0; ji < faces.size(); ++ji)
		{
			rectangle(frame, faces[ji], CV_RGB(0, 0, 255), 4);
		}
	}
	//-- Show what you got
	//namedWindow(window_name, WINDOW_AUTOSIZE);
	//imshow(window_name, frame);
	imwrite("cpu.jpg", frame);
}

void detectAndDisplayGPUCUDA(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	GpuMat faceBuf_gpu;
	GpuMat GpuImg;
	GpuImg.upload(frame_gray);

	const int defaultObjSearchNum = 10000;
	if (faceBuf_gpu.empty())
	{
		faceBuf_gpu.create(1, defaultObjSearchNum, DataType<Rect>::type);
	}

	Atime = getTickCount();
	int detectionNumber = face_cascade_gpu.detectMultiScale(GpuImg, faceBuf_gpu, 1.1, 2, Size(SIZE, SIZE));
	Btime = getTickCount();
	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(cuda version) = %d / %lf sec take.\n", detectionNumber, TakeTime);

	Mat faces_downloaded;
	if (detectionNumber >= 1)
	{
		faceBuf_gpu.colRange(0, detectionNumber).download(faces_downloaded);
		Rect* faces = faces_downloaded.ptr< Rect>();

		for (int ji = 0; ji < detectionNumber; ++ji)
		{
			rectangle(frame, Point(faces[ji].x, faces[ji].y), Point(faces[ji].x + faces[ji].width, faces[ji].y + faces[ji].height), CV_RGB(0, 255, 0), 4);
		}
	}

	imwrite("cuda.jpg", frame);
	//imshow(window_namegpu_cuda, frame);
}

void detectAndDisplayOpenCL(Mat& img, cv::ocl::OclCascadeClassifier& cascade, CascadeClassifier&, double scale)
{

	int i = 0;
	double t = 0;
	vector<Rect> faces;

	cv::ocl::oclMat image(img);
	cv::ocl::oclMat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);

	cv::ocl::cvtColor(image, gray, CV_BGR2GRAY);
	cv::ocl::resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	cv::ocl::equalizeHist(smallImg, smallImg);

	CvSeq* _objects;
	MemStorage storage(cvCreateMemStorage(0));
	Atime = getTickCount();
	_objects = cascade.oclHaarDetectObjects(smallImg, storage, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(SIZE, SIZE), Size(0, 0));
	Btime = getTickCount();

	vector<CvAvgComp> vecAvgComp;
	Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
	faces.resize(vecAvgComp.size());
	std::transform(vecAvgComp.begin(), vecAvgComp.end(), faces.begin(), getRect());

	TakeTime = (Btime - Atime) / getTickFrequency();
	printf("detected face(ocl version) = %d / %lf sec take.\n", faces.size(), TakeTime);

	if (faces.size() >= 1)
	{
		for (int ji = 0; ji < faces.size(); ++ji)
		{
			rectangle(img, faces[ji], CV_RGB(255, 0, 0), 4);
		}
	}
	//namedWindow(window_namegpu, WINDOW_AUTOSIZE);
	//imshow(window_namegpu, img);
	imwrite("ocl.jpg", img);
}

void detectAndDisplayOpenCL(Mat& img, vector<Rect>& faces, ocl::OclCascadeClassifier& cascade, double scale, bool calTime)
{
	ocl::oclMat image(img);
	ocl::oclMat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
	ocl::cvtColor(image, gray, CV_BGR2GRAY);
	ocl::resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	ocl::equalizeHist(smallImg, smallImg);

	cascade.detectMultiScale(smallImg, faces, 1.1,
		3, 0
		| CV_HAAR_SCALE_IMAGE
		, Size(30, 30), Size(0, 0));
}

int detectAndDisplay(bool useGPU, Mat image, string name)
{

	CascadeClassifier_GPU cascade_gpu;
	if (!cascade_gpu.load(haarcascade_DIR_CUDA))
	{
		return cerr << "ERROR: Could not load cascade classifier \"" << haarcascade_DIR_CUDA << "\"" << endl, -1;
	}

	CascadeClassifier cascade_cpu;
	if (!cascade_cpu.load(haarcascade_DIR_CUDA))
	{
		return cerr << "ERROR: Could not load cascade classifier \"" << haarcascade_DIR_CUDA << "\"" << endl, -1;
	}

	Mat frame, frame_cpu, gray_cpu, resized_cpu, faces_downloaded, frameDisp;
	vector<Rect> facesBuf_cpu;

	GpuMat frame_gpu, gray_gpu, resized_gpu, facesBuf_gpu;

	const int defaultObjSearchNum = 10000;
	if (facesBuf_gpu.empty())
	{
		facesBuf_gpu.create(1, defaultObjSearchNum, DataType<Rect>::type);
	}

	double scaleFactor = 1.0;
	bool findLargestObject = false;
	bool filterRects = true;
	int detections_num;

	(image.empty() ? frame : image).copyTo(frame_cpu);
	frame_gpu.upload(image.empty() ? frame : image);

	convertAndResize(frame_gpu, gray_gpu, resized_gpu, scaleFactor);
	convertAndResize(frame_cpu, gray_cpu, resized_cpu, scaleFactor);

	Atime = getTickCount();

	if (useGPU)
	{
		//cascade_gpu.visualizeInPlace = true;

		cascade_gpu.findLargestObject = findLargestObject;

		detections_num = cascade_gpu.detectMultiScale(resized_gpu, facesBuf_gpu, 1.2,
			(filterRects || findLargestObject) ? 4 : 0);
		//facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);
	}
	else
	{
		Size minSize = cascade_gpu.getClassifierSize();
		cascade_cpu.detectMultiScale(resized_cpu, facesBuf_cpu, 1.2,
			(filterRects || findLargestObject) ? 4 : 0,
			(findLargestObject ? CASCADE_FIND_BIGGEST_OBJECT : 0)
			| CV_HAAR_SCALE_IMAGE,
			minSize);
		detections_num = (int)facesBuf_cpu.size();
	}

	Btime = getTickCount();
	TakeTime = (Btime - Atime) / getTickFrequency();

	facesBuf_gpu.colRange(0, detections_num).download(faces_downloaded);

	if (!useGPU && detections_num)
	{
		for (int i = 0; i < detections_num; ++i)
		{
			rectangle(image, facesBuf_cpu[i], CV_RGB(255, 0, 0), 2);
		}
		printf("detected face(%s version) = %d / %lf sec take.\n", "CPU", detections_num, TakeTime);
	}

	if (useGPU)
	{
		resized_gpu.download(resized_cpu);

		for (int i = 0; i < detections_num; ++i)
		{
			rectangle(image, faces_downloaded.ptr<Rect>()[i], CV_RGB(0, 0, 255), 2);
		}
		printf("detected face(%s version) = %d / %lf sec take.\n", "CUDA", detections_num, TakeTime);
	}

	imshow(name, image);
	imwrite(name + ".jpg", image);
	return 0;

}


/** @function main */
int main(int argc, const char *argv[])
{

	//cout << cv::getBuildInformation() << endl;

	bool useCUDA = true;

	if (getCudaEnabledDeviceCount() == 0)
	{
		useCUDA = false;
		return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
	}

	cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	printf("\n");
	//CvCapture* capture;

	Mat img = imread(TEST_IMAGE);
	Mat imgCPU = img.clone();
	Mat imgOCL = img.clone();
	Mat imgCUDA = img.clone();

	detectAndDisplay(false, imgCPU, "CPU");
	detectAndDisplay(true, imgCUDA, "CUDA");

	/*

	//Code use CPU
	if (RUNCPU && BUILDALL)
	{
		if (!face_cascade.load(haarcascade_DIR_CUDA)){ printf("--(!)CPU Error loading\n"); return -1; };
		detectAndDisplayCPU(imgCPU);
	}

	//Code use CUDA
	if (useCUDA && BUILDALL)
	{
		if (!face_cascade_gpu.load(haarcascade_DIR_CUDA)){ printf("--(!)GPU CUDA Error loading\n"); return -1; };
		detectAndDisplayGPUCUDA(imgCUDA);

	}

	if (!useCUDA || RUNOCL)
	{
		//Code use OpenCL
		DevicesInfo devices;
		getOpenCLDevices(devices);
		setDevice(devices[0]);
		if (!cascade_ocl.load(haarcascade_DIR_CUDA)){ printf("--(!)GPU OpenCL Error loading\n"); return -1; };
		detectAndDisplayOpenCL(imgOCL, cascade_ocl, nestedCascade, 1.0);

	}


	*/




	cv::waitKey(0);
	/*
	//-- 2. Read the video stream
	capture = cvCaptureFromCAM(0);
	if (capture)
	{
	while (true)
	{
	frame = cvQueryFrame(capture);

	//-- 3. Apply the classifier to the frame
	if (!frame.empty())
	{
	detectAndDisplay(frame);
	}
	else
	{
	printf(" --(!) No captured frame -- Break!"); break;
	}

	int c = waitKey(10);
	if ((char)c == 'c') { break; }
	}
	}
	*/
	return 0;
}
