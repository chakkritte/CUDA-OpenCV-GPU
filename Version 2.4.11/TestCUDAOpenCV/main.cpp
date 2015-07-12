#define BUILDALL true
#define RUNCPU true
#define RUNOCL true
#define haarcascade_DIR "haarcascades\\haarcascade_frontalface_alt2.xml"
#define haarcascade_DIR_CUDA "haarcascades_cuda\\haarcascade_frontalface_alt2.xml"
#define TEST_IMAGE "MyImages\\got1.jpg"
#define SIZE 20

#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/gpu/gpu.hpp"
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
//#pragma comment(lib, "opencv_gpu2411.lib")
#pragma comment(lib, "opencv_highgui2411.lib")
#endif

using namespace std;
using namespace cv;
using namespace cv::gpu;
using namespace cv::ocl;


static void detectFaces(Mat&, vector<Rect_<int> >&, string);
static void detectEyes(Mat&, vector<Rect_<int> >&, string);
static void detectNose(Mat&, vector<Rect_<int> >&, string);
static void detectMouth(Mat&, vector<Rect_<int> >&, string);
static void detectFacialFeaures(Mat&, const vector<Rect_<int> >, string, string, string);

String eye_cascade_path = "haarcascades\\haarcascade_eye.xml";
String nose_cascade_path = "haarcascades\\haarcascade_mcs_nose.xml";
String mouth_cascade_path = "haarcascades\\haarcascade_mcs_mouth.xml";

/** Global variables */
String face_cascade_name = haarcascade_DIR_CUDA;
CascadeClassifier face_cascade;

//CUDA
//CascadeClassifier_GPU face_cascade_gpu;
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
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(SIZE, SIZE), Size(0, 0));

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
	imshow(window_name, frame);
	//imwrite("cpu.jpg", frame);
}

/*
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
*/


Mat GreyAndEqualizeHist(Mat& img)
{
	Mat frame_gray;
	cvtColor(img, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	return frame_gray;
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
	_objects = cascade.oclHaarDetectObjects(smallImg, storage, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(SIZE, SIZE), Size(0, 0));
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
			//rectangle(img, faces[ji], CV_RGB(255, 0, 0), 4);
			//Mat crop_img = img[new Scalar(faces[ji].x, faces[ji].y, faces[ji].width, faces[ji].height)];
			Mat cropedImage = img(Rect(faces[ji].x, faces[ji].y, faces[ji].width, faces[ji].height));
			String name = "Crop\\";
			String namefile = ".jpg";
			//String filename = name + namefile;

			Mat imageFace = GreyAndEqualizeHist(cropedImage);

			imwrite(name + to_string(ji + 1) + namefile, imageFace);
		}
	}
	//Mat result = GreyAndEqualizeHist(img);
	//detectFacialFeaures(result, faces, eye_cascade_path, nose_cascade_path, mouth_cascade_path);
	//imshow("Result", result);

	//namedWindow(window_namegpu, WINDOW_AUTOSIZE);
	//imshow(window_namegpu, img);
	//imwrite("ocl.jpg", img);
}


/*
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

*/



static void detectFacialFeaures(Mat& img, const vector<Rect_<int> > faces, string eye_cascade,
	string nose_cascade, string mouth_cascade)
{
	for (unsigned int i = 0; i < faces.size(); ++i)
	{
		// Mark the bounding box enclosing the face
		Rect face = faces[i];
		rectangle(img, Point(face.x, face.y), Point(face.x + face.width, face.y + face.height),
			Scalar(255, 0, 0), 1, 4);

		// Eyes, nose and mouth will be detected inside the face (region of interest)
		Mat ROI = img(Rect(face.x, face.y, face.width, face.height));

		// Check if all features (eyes, nose and mouth) are being detected
		bool is_full_detection = false;
		if ((!eye_cascade.empty()) && (!nose_cascade.empty()) && (!mouth_cascade.empty()))
			is_full_detection = true;

		// Detect eyes if classifier provided by the user
		if (!eye_cascade.empty())
		{
			vector<Rect_<int> > eyes;
			detectEyes(ROI, eyes, eye_cascade);

			// Mark points corresponding to the centre of the eyes
			for (unsigned int j = 0; j < eyes.size(); ++j)
			{
				Rect e = eyes[j];
				circle(ROI, Point(e.x + e.width / 2, e.y + e.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				/* rectangle(ROI, Point(e.x, e.y), Point(e.x+e.width, e.y+e.height),
				Scalar(0, 255, 0), 1, 4); */
			}
		}

		// Detect nose if classifier provided by the user
		double nose_center_height = 0.0;
		if (!nose_cascade.empty())
		{
			vector<Rect_<int> > nose;
			detectNose(ROI, nose, nose_cascade);

			// Mark points corresponding to the centre (tip) of the nose
			for (unsigned int j = 0; j < nose.size(); ++j)
			{
				Rect n = nose[j];
				circle(ROI, Point(n.x + n.width / 2, n.y + n.height / 2), 3, Scalar(0, 255, 0), -1, 8);
				nose_center_height = (n.y + n.height / 2);
			}
		}

		// Detect mouth if classifier provided by the user
		double mouth_center_height = 0.0;
		if (!mouth_cascade.empty())
		{
			vector<Rect_<int> > mouth;
			detectMouth(ROI, mouth, mouth_cascade);

			for (unsigned int j = 0; j < mouth.size(); ++j)
			{
				Rect m = mouth[j];
				mouth_center_height = (m.y + m.height / 2);

				// The mouth should lie below the nose
				if ((is_full_detection) && (mouth_center_height > nose_center_height))
				{
					rectangle(ROI, Point(m.x, m.y), Point(m.x + m.width, m.y + m.height), Scalar(0, 255, 0), 1, 4);
				}
				else if ((is_full_detection) && (mouth_center_height <= nose_center_height))
					continue;
				else
					rectangle(ROI, Point(m.x, m.y), Point(m.x + m.width, m.y + m.height), Scalar(0, 255, 0), 1, 4);
			}
		}

	}

	return;
}


static void detectEyes(Mat& img, vector<Rect_<int> >& eyes, string cascade_path)
{
	CascadeClassifier eyes_cascade;
	eyes_cascade.load(cascade_path);

	eyes_cascade.detectMultiScale(img, eyes, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectNose(Mat& img, vector<Rect_<int> >& nose, string cascade_path)
{
	CascadeClassifier nose_cascade;
	nose_cascade.load(cascade_path);

	nose_cascade.detectMultiScale(img, nose, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}

static void detectMouth(Mat& img, vector<Rect_<int> >& mouth, string cascade_path)
{
	CascadeClassifier mouth_cascade;
	mouth_cascade.load(cascade_path);

	mouth_cascade.detectMultiScale(img, mouth, 1.20, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
	return;
}



/** @function main */
int main(int argc, const char *argv[])
{

	//cout << cv::getBuildInformation() << endl;

	bool useCUDA = true;
	/*
	if (getCudaEnabledDeviceCount() == 0)
	{
	useCUDA = false;
	return cerr << "No GPU found or the library is compiled without GPU support" << endl, -1;
	}
	*/
	//cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());

	printf("\n");
	//CvCapture* capture;

	Mat img = imread(TEST_IMAGE);
	Mat imgCPU = img.clone();
	Mat imgOCL = img.clone();
	//Mat imgCUDA = img.clone();

	//detectAndDisplay(false, imgCPU, "CPU");
	//detectAndDisplay(true, imgCUDA, "CUDA");



	//Code use CPU
	if (RUNCPU && BUILDALL)
	{
		//if (!face_cascade.load(haarcascade_DIR_CUDA)){ printf("--(!)CPU Error loading\n"); return -1; };
		//detectAndDisplayCPU(imgCPU);
	}

	//Code use CUDA
	/*

	if (useCUDA && BUILDALL)
	{
	if (!face_cascade_gpu.load(haarcascade_DIR_CUDA)){ printf("--(!)GPU CUDA Error loading\n"); return -1; };
	detectAndDisplayGPUCUDA(imgCUDA);

	}
	*/
	if (!useCUDA || RUNOCL)
	{
		//Code use OpenCL
		DevicesInfo devices;
		getOpenCLDevices(devices);
		setDevice(devices[0]);
		if (!cascade_ocl.load(haarcascade_DIR_CUDA)){ printf("--(!)GPU OpenCL Error loading\n"); return -1; };
		detectAndDisplayOpenCL(imgOCL, cascade_ocl, nestedCascade, 1.0);

	}


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
