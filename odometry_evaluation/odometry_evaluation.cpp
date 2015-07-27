/*
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Willow Garage, Inc. nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/
/*
* this program is modified to show point cloud using viz and evaluate odometry using ICL-NUIM datasets.
*/
#include <opencv2/opencv.hpp>
#include <opencv_lib.hpp>

#include <opencv2/rgbd.hpp>
#include <opencv2/viz.hpp>

////Dependencies
////C:\Program Files (x86)\PCL 1.7.2\3rdParty\VTK\lib\vtk-5.10;
////C:\Program Files (x86)\PCL 1.7.2\3rdParty\VTK\include\vtk-5.10;
//#include <opencv_vtk_lib.hpp>
//#include <opencv2/viz/widget_accessor.hpp>

//#include <vtkPoints.h>
//#include <vtkTriangle.h>
//#include <vtkCellArray.h>
//#include <vtkPolyData.h>
//#include <vtkPolyDataMapper.h>
//#include <vtkIdList.h>
//#include <vtkActor.h>
//#include <vtkProp.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv::rgbd;

#define BILATERAL_FILTER 0// if 1 then bilateral filter will be used for the depth

class MyTickMeter
{
public:
	MyTickMeter() { reset(); }
	void start() { startTime = getTickCount(); }
	void stop()
	{
		int64 time = getTickCount();
		if (startTime == 0)
			return;
		++counter;
		sumTime += (time - startTime);
		startTime = 0;
	}

	int64 getTimeTicks() const { return sumTime; }
	double getTimeSec()   const { return (double)getTimeTicks() / getTickFrequency(); }
	int64 getCounter() const { return counter; }

	void reset() { startTime = sumTime = 0; counter = 0; }
private:
	int64 counter;
	int64 sumTime;
	int64 startTime;
};

static
void writeResults(const string& avetime, const string& filename, const vector<string>& timestamps, const vector<Mat>& Rt)
{
	CV_Assert(timestamps.size() == Rt.size());

	ofstream file(filename.c_str());
	if (!file.is_open())
		return;

	file << "#" << avetime << endl;
	cout.precision(4);
	for (size_t i = 0; i < Rt.size(); i++)
	{
		const Mat& Rt_curr = Rt[i];
		if (Rt_curr.empty())
			continue;

		CV_Assert(Rt_curr.type() == CV_64FC1);

		Mat R = Rt_curr(Rect(0, 0, 3, 3)), rvec;
		Rodrigues(R, rvec);
		double alpha = norm(rvec);
		if (alpha > DBL_MIN)
			rvec = rvec / alpha;

		double cos_alpha2 = std::cos(0.5 * alpha);
		double sin_alpha2 = std::sin(0.5 * alpha);

		rvec *= sin_alpha2;

		CV_Assert(rvec.type() == CV_64FC1);
		// timestamp tx ty tz qx qy qz qw
		file << timestamps[i] << " " << fixed
			<< Rt_curr.at<double>(0, 3) << " " << Rt_curr.at<double>(1, 3) << " " << Rt_curr.at<double>(2, 3) << " "
			<< rvec.at<double>(0) << " " << rvec.at<double>(1) << " " << rvec.at<double>(2) << " " << cos_alpha2 << endl;

	}
	file.close();
}

static
void setCameraMatrixFreiburg1(float& fx, float& fy, float& cx, float& cy)
{
	fx = 517.3f; fy = 516.5f; cx = 318.6f; cy = 255.3f;
}

static
void setCameraMatrixFreiburg2(float& fx, float& fy, float& cx, float& cy)
{
	fx = 520.9f; fy = 521.0f; cx = 325.1f; cy = 249.7f;
}

static
void setCameraMatrixIclnuim(float& fx, float& fy, float& cx, float& cy)
{
	fx = 481.20f; fy = -480.0f; cx = 319.5f; cy = 239.5f;
}

#include <sstream>
#include <string>
template<typename T_n>
string NumToString(T_n number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

/*
* This sample helps to evaluate odometry on TUM datasets and benchmark http://vision.in.tum.de/data/datasets/rgbd-dataset.
* At this link you can find instructions for evaluation. The sample runs some opencv odometry and saves a camera trajectory
* to file of format that the benchmark requires. Saved file can be used for online evaluation.
*/
/*
 * and also helps to evaluate odpmetry on ICL-NUIM datasets http://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
 */

int main(int argc, char** argv)
{
	int benchmark = 1; // 1: TUM 2: ICL-NUIM

	vector<string> timestamps;
	vector<Mat> Rts;

	string filename;
	ifstream file;

	string trajname;
	if (benchmark == 1){
		trajname = "fr1.xyz.";
	}
	else if (benchmark == 2){
		trajname = "livingRoom0.freiburg.";
	}

	string dirname;
	int timestampLength = 17;
	int rgbPathLehgth = 17 + 8;
	int depthPathLehgth = 17 + 10;
	if (benchmark == 1){
		dirname = "C:/Users/fujimoto/Documents/RGB-Ddataset/rgbd_dataset_freiburg1_xyz/";
	}else if(benchmark == 2){	
		dirname = "C:/Users/fujimoto/Documents/RGB-Ddataset/living_room_traj0n_frei_png/";
	}

	filename = dirname + "associations.txt";
	file.open(filename.c_str());
	if (!file.is_open())
		return -1;

	int startnum = 0;
	float fx = 525.0f, // default
		fy = 525.0f,
		cx = 319.5f,
		cy = 239.5f;
	if (filename.find("freiburg1") != string::npos)
		setCameraMatrixFreiburg1(fx, fy, cx, cy);
	if (filename.find("freiburg2") != string::npos)
		setCameraMatrixFreiburg2(fx, fy, cx, cy);
	if (filename.find("living_room") != string::npos)
		setCameraMatrixIclnuim(fx, fy, cx, cy);
	if (filename.find("office_room") != string::npos)
		setCameraMatrixIclnuim(fx, fy, cx, cy);
	if (filename.find("office_room_traj0") != string::npos) //なぜかtraj0だけ1から始まっている
		startnum = 1;

	Mat cameraMatrix = Mat::eye(3, 3, CV_32FC1);
	{
		cameraMatrix.at<float>(0, 0) = fx;
		cameraMatrix.at<float>(1, 1) = fy;
		cameraMatrix.at<float>(0, 2) = cx;
		cameraMatrix.at<float>(1, 2) = cy;
	}

	Ptr<OdometryFrame> frame_prev = Ptr<OdometryFrame>(new OdometryFrame()),
		frame_curr = Ptr<OdometryFrame>(new OdometryFrame());
	//odometry_name: Rgbd or ICP or RgbdICP
	Ptr<Odometry> odometry = Odometry::create(string(argv[1]) + "Odometry");
	if (odometry.empty())
	{
		cout << "Can not create Odometry algorithm. Check the passed odometry name." << endl;
		return -1;
	}
	odometry->setCameraMatrix(cameraMatrix);

	viz::Viz3d myWindow("Point Cloud");
	/// Pose of the widget in camera frame
	cv::Affine3d cloud_pose = cv::Affine3d().translate(cv::Vec3d(0.0f, 0.0f, 0.0f));
	/// Pose of the widget in global frame
	cv::Affine3d cloud_pose_global = cloud_pose;
	cv::Affine3d cam_pose;

	MyTickMeter gtm;
	int count = 0;
	for (int i = 0; !file.eof(); i++)
	{
		string str;
		if (benchmark == 2){
			if (i == 0 && startnum == 1){
				i++;
			}
		}
		
		std::getline(file, str);
		if (str.empty()) break;
		if (str.at(0) == '#') continue; /* comment */
		
		Mat image, depth;
		// Read one pair (rgb and depth)
		// example: 1305031453.359684 rgb/1305031453.359684.png 1305031453.374112 depth/1305031453.374112.png
#if BILATERAL_FILTER
		MyTickMeter tm_bilateral_filter;
#endif
		{
			string rgbFilename;
			string timestap;
			string depthFilename;
			if (benchmark == 1){
				rgbFilename = str.substr(timestampLength + 1, rgbPathLehgth);
				timestap = str.substr(0, timestampLength);
				depthFilename = str.substr(2 * timestampLength + rgbPathLehgth + 3, depthPathLehgth);
			}
			else if (benchmark == 2){
				rgbFilename = NumToString(i) + ".png";
				timestap = NumToString(i);
				depthFilename = NumToString(i) + ".png";
			}

			if (benchmark == 1){
				image = imread(dirname + rgbFilename);
				depth = imread(dirname + depthFilename, -1);
			}
			else if (benchmark == 2){
				image = imread(dirname + "rgb/" + rgbFilename);
				depth = imread(dirname + "depth/" + depthFilename, -1);
			}

			CV_Assert(!image.empty());
			CV_Assert(!depth.empty());
			CV_Assert(depth.type() == CV_16UC1);

			cout << i << " " << rgbFilename << " " << depthFilename << endl;

			// scale depth
			Mat depth_flt;
			depth.convertTo(depth_flt, CV_32FC1, 1.f / 5000.f);
#if !BILATERAL_FILTER
			depth_flt.setTo(std::numeric_limits<float>::quiet_NaN(), depth == 0);
			depth = depth_flt;
#else
			tm_bilateral_filter.start();
			depth = Mat(depth_flt.size(), CV_32FC1, Scalar(0));
			const double depth_sigma = 0.03;
			const double space_sigma = 4.5;  // in pixels
			Mat invalidDepthMask = depth_flt == 0.f;
			depth_flt.setTo(-5 * depth_sigma, invalidDepthMask);
			bilateralFilter(depth_flt, depth, -1, depth_sigma, space_sigma);
			depth.setTo(std::numeric_limits<float>::quiet_NaN(), invalidDepthMask);
			tm_bilateral_filter.stop();
			cout << "Time filter " << tm_bilateral_filter.getTimeSec() << endl;
#endif
			timestamps.push_back(timestap);
		}

		{
			Mat gray;
			cvtColor(image, gray, COLOR_BGR2GRAY);
			frame_curr->image = gray;
			frame_curr->depth = depth;

			Mat Rt;
			if (!Rts.empty())
			{
				MyTickMeter tm;
				tm.start();
				gtm.start();
				bool res = odometry->compute(frame_curr, frame_prev, Rt);
				gtm.stop();
				tm.stop();
				count++;
				cout << "Time " << tm.getTimeSec() << endl;
#if BILATERAL_FILTER
				cout << "Time ratio " << tm_bilateral_filter.getTimeSec() / tm.getTimeSec() << endl;
#endif
				if (!res)
					Rt = Mat::eye(4, 4, CV_64FC1);
			}

			if (Rts.empty())
				Rts.push_back(Mat::eye(4, 4, CV_64FC1));
			else
			{
				Mat& prevRt = *Rts.rbegin();
				//cout << "Rt " << Rt << endl;
				Rts.push_back(prevRt * Rt);
			}

			//10フレームごと変換して表示
			Mat rot = Rts[count](Rect(0, 0, 3, 3)).t();
			Mat tvec = Rts[count](Rect(3, 0, 1, 3)).t();
			if (count % 10 == 0){
				int downSamplingNum = 4; //e.g. 4
				Mat image2(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_8UC3);
				resize(image, image2, image2.size(), 0, 0, INTER_LINEAR);
				Mat pCloud(image.rows / downSamplingNum, image.cols / downSamplingNum, CV_64FC3);

				for (int y = 0; y < 480; y += downSamplingNum){
					for (int x = 0; x < 640; x += downSamplingNum){
						if (depth.at<float>(y, x) < 8.0 && depth.at<float>(y, x) > 0.4){
							//RGB-D Dataset
							Mat pmat(1, 3, CV_64F);
							pmat.at<double>(0, 2) = (double)depth.at<float>(y, x);
							pmat.at<double>(0, 0) = (x - cx) * pmat.at<double>(0, 2) / fx;
							pmat.at<double>(0, 1) = (y - cy) * pmat.at<double>(0, 2) / fy;
							pmat = (pmat)*rot + tvec;
							Point3d p(pmat);
							pCloud.at<Point3d>(y / downSamplingNum, x / downSamplingNum) = p;
							pmat.release();
						}
						else{
							//RGB-D Dataset
							pCloud.at<Vec3d>(y / downSamplingNum, x / downSamplingNum) = Vec3d(0.f, 0.f, 0.f);
						}
					}
				}

				viz::WCloud wcloud(pCloud, image2);
				string myWCloudName = "CLOUD" + NumToString(count);
				myWindow.showWidget(myWCloudName, wcloud, cloud_pose_global);

				pCloud.release();
				image2.release();
			}

			cam_pose = cv::Affine3d(rot.t(), tvec);
			viz::WCameraPosition cpw(0.1); // Coordinate axes
			viz::WCameraPosition cpw_frustum(cv::Matx33d(cameraMatrix), /*image,*/ 0.1, viz::Color::white()); // Camera frustum
			string widgetPoseName = "CPW" + NumToString(count);
			string widgetFrustumName = "CPW_FRUSTUM" + NumToString(count);
			myWindow.showWidget(widgetPoseName, cpw, cam_pose);
			myWindow.showWidget(widgetFrustumName, cpw_frustum, cam_pose);
			myWindow.spinOnce(1, true);


			if (!frame_prev.empty())
				frame_prev->release();
			std::swap(frame_prev, frame_curr);

			rot.release();
			tvec.release();
		}
	}

	std::cout << "Average time " << gtm.getTimeSec() / count << std::endl;
	string avetime = NumToString(gtm.getTimeSec() / count);
	writeResults(avetime, trajname + string(argv[1]) + ".txt", timestamps, Rts);

	return 0;
}
