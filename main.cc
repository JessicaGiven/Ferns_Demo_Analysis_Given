/*
  Copyright 2007, 2008 Computer Vision Lab,
  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
  All rights reserved.

  Authors: Vincent Lepetit (http://cvlab.epfl.ch/~lepetit)
           Mustafa Ozuysal (http://cvlab.epfl.ch/~oezuysal)
           Julien  Pilet   (http://cvlab.epfl.ch/~jpilet)

  This file is part of the ferns_demo software.

  ferns_demo is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or (at your option) any later
  version.

  ferns_demo is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE. See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with
  ferns_demo; if not, write to the Free Software Foundation, Inc., 51 Franklin
  Street, Fifth Floor, Boston, MA 02110-1301, USA
*/
#include <cv.h>
#include <highgui.h>

#include <iostream>
#include <string>
using namespace std;

#include "mcv.h"
#include "planar_pattern_detector_builder.h"
#include "template_matching_based_tracker.h"

const int max_filename = 1000;
/*

枚举三个输入源：
摄像头、图片序列设备、视频文件

*/
enum source_type {webcam_source, sequence_source, video_source};

//定义一个新的二维特征检测器类
planar_pattern_detector * detector;
//定义一个新的模板匹配跟踪器
template_matching_based_tracker * tracker;

int mode = 2;
bool show_tracked_locations = true;
bool show_keypoints = true;
//初始化文字表示结构
CvFont font;

/*

draw_quadrangle函数

绘制四边形框

*/

void draw_quadrangle(IplImage * frame,	//输入图像
		     int u0, int v0,	
		     int u1, int v1,
		     int u2, int v2,
		     int u3, int v3,	//四边形四顶点坐标
		     CvScalar color, int thickness = 1)	//color:四边形颜色；thinckness:四边形线段粗细
{
  cvLine(frame, cvPoint(u0, v0), cvPoint(u1, v1), color, thickness);
  cvLine(frame, cvPoint(u1, v1), cvPoint(u2, v2), color, thickness);
  cvLine(frame, cvPoint(u2, v2), cvPoint(u3, v3), color, thickness);
  cvLine(frame, cvPoint(u3, v3), cvPoint(u0, v0), color, thickness);
}

/*

draw_detected_position函数

绘制检测位置的四边形框（白色）

*/

void draw_detected_position(IplImage * frame, planar_pattern_detector * detector)
{
  draw_quadrangle(frame,
		  detector->detected_u_corner[0], detector->detected_v_corner[0],
		  detector->detected_u_corner[1], detector->detected_v_corner[1],
		  detector->detected_u_corner[2], detector->detected_v_corner[2],
		  detector->detected_u_corner[3], detector->detected_v_corner[3],
		  cvScalar(255), 3);
}

/*

draw_initial_rectangle函数

初始化四边形框（灰色）

*/

void draw_initial_rectangle(IplImage * frame, template_matching_based_tracker * tracker)
{
  draw_quadrangle(frame,
		  tracker->u0[0], tracker->u0[1],
		  tracker->u0[2], tracker->u0[3],
		  tracker->u0[4], tracker->u0[5],
		  tracker->u0[6], tracker->u0[7],
		  cvScalar(128), 3);
}

/*

draw_tracked_position函数

绘制跟踪位置的四边形框（白色）

*/

void draw_tracked_position(IplImage * frame, template_matching_based_tracker * tracker)
{
  draw_quadrangle(frame,
		  tracker->u[0], tracker->u[1],
		  tracker->u[2], tracker->u[3],
		  tracker->u[4], tracker->u[5],
		  tracker->u[6], tracker->u[7],
		  cvScalar(255), 3);
}

/*

draw_tracked_locations函数

绘制跟踪位置的圆形框（白色）

*/

void draw_tracked_locations(IplImage * frame, template_matching_based_tracker * tracker)
{
  for(int i = 0; i < tracker->nx * tracker->ny; i++) {
    int x1, y1;
    tracker->f.transform_point(tracker->m[2 * i], tracker->m[2 * i + 1], x1, y1);
    cvCircle(frame, cvPoint(x1, y1), 3, cvScalar(255, 255, 255), 1);
  }
}

/*

draw_detected_keypoints函数

绘制特征点位置的圆形框（灰色）

*/

void draw_detected_keypoints(IplImage * frame, planar_pattern_detector * detector)
{
  for(int i = 0; i < detector->number_of_detected_points; i++)
    cvCircle(frame,
	     cvPoint(detector->detected_points[i].fr_u(),
		     detector->detected_points[i].fr_v()),
	     16 * (1 << int(detector->detected_points[i].scale)),
	     cvScalar(100), 1);
}

/*

draw_recognized_keypoints函数

绘制潜在匹配特征点位置的圆形框（白色）（雾？）

*/

void draw_recognized_keypoints(IplImage * frame, planar_pattern_detector * detector)
{
  for(int i = 0; i < detector->number_of_model_points; i++)
    if (detector->model_points[i].class_score > 0)
      cvCircle(frame,
	       cvPoint(detector->model_points[i].potential_correspondent->fr_u(),
		       detector->model_points[i].potential_correspondent->fr_v()),
	       16 * (1 << int(detector->detected_points[i].scale)),
	       cvScalar(255, 255, 255), 1);
}

/*

detect_and_draw函数

0: Detect when tracking fails or for initialization then track.	当追踪失败时进行检测，或者初始化追踪
1: Track only	//只追踪
2: Detect only (DEFAULT)	//只检测（默认）
3: Detect + track in every frame	//对每一帧进行都进行追踪和检测

The number  keys 4&5  can be  used to turn  on/off the  recognized and
 detected keypoints, respectively.

 键入4，5可以分别控制是否识别和检测特征点

*/

void detect_and_draw(IplImage * frame)
{
	static bool last_frame_ok=false;	//上一帧成功处理标记

	if (mode == 1 || ((mode==0) && last_frame_ok)) {
		bool ok = tracker->track(frame);	//对当前帧进行跟踪，返回跟踪是否成功标记
		last_frame_ok=ok;


		if (!ok) {
			if (mode==0) return detect_and_draw(frame);	//如果追踪失败，则重载该函数一次跳过跟踪阶段进入检测阶段
			else {
				draw_initial_rectangle(frame, tracker);	//初始化四边形框
				tracker->initialize();	//初始化跟踪器
			}
		} else {
			draw_tracked_position(frame, tracker);	//绘制跟踪位置
			if (show_tracked_locations) draw_tracked_locations(frame, tracker);	//绘制跟踪位置（变换）
		}
		cvPutText(frame, "template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//在视频上叠加template-based 3D tracking
	} else {
		detector->detect(frame);	//对当前帧进行检测
		
		if (detector->pattern_is_detected) {
			last_frame_ok=true;	//检测到特征就置位成功标记
			//利用检测器的结果对跟踪器进行初始化
			tracker->initialize(detector->detected_u_corner[0], detector->detected_v_corner[0],
					detector->detected_u_corner[1], detector->detected_v_corner[1],
					detector->detected_u_corner[2], detector->detected_v_corner[2],
					detector->detected_u_corner[3], detector->detected_v_corner[3]);

			if (mode == 3 && tracker->track(frame)) {
				//绘制特征点位置
				if (show_keypoints) {
					draw_detected_keypoints(frame, detector); //绘制随机蕨分类器检测到的当前帧关键点
					draw_recognized_keypoints(frame, detector); //绘制经过比对的当前帧关键点
				}
				draw_tracked_position(frame, tracker);	//绘制追踪位置
				if (show_tracked_locations) draw_tracked_locations(frame, tracker);	//绘制追踪位置(变换）

				cvPutText(frame, "detection+template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//在视频上叠加detection+template-based 3D tracking
			} else {//绘制特征点位置
				if (show_keypoints) {
					draw_detected_keypoints(frame, detector);	 
					draw_recognized_keypoints(frame, detector);
				}
				draw_detected_position(frame, detector);	//绘制检测位置
				cvPutText(frame, "detection", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//在视频上叠加template-based 3D tracking
			}
		} else {
			last_frame_ok=false;
			if (show_keypoints) draw_detected_keypoints(frame, detector);	//绘制特征点位置

			if (mode == 3)
				cvPutText(frame, "detection + template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//在视频上叠加detection + template-based 3D tracking
			else
				cvPutText(frame, "detection", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//在视频上叠加detection
		}
	}

	cvShowImage("ferns-demo", frame);	//显示标记之后的帧
}

//帮助输出内容
void help(const string& exec_name) {
  cout << exec_name << " [-m <model image>] [-s <image sequence format>]\n\n";
  cout << "   -m : specify the name of the model image depicting the planar \n";
  cout << "        object from a frontal viewpoint. Default model.bmp\n";
  cout << "   -s : image sequence format in printf style, e.g. image%04.jpg,\n";
  cout << "        to test detection. If not specified webcam is used as \n";
  cout << "        image source.\n";
  cout << "   -v : video filename to test detection. If not specified webcam\n";
  cout << "        is used as image source.\n";
  cout << "   -h : This help message." << endl;
}

//主函数
int main(int argc, char ** argv)
{
  string model_image     = "model.bmp";	//定义模型图片文件名
  string sequence_format = "";	//序列格式
  string video_file = "";	//视频文件名
  source_type frame_source = webcam_source;	//默认输入模式：摄像头模式

  //接受键盘输入的命令，根据命令索引help
  for(int i = 0; i < argc; ++i) {
    if(strcmp(argv[i], "-h") == 0) {
      help(argv[0]);
      return 0;
    }
	//读取模型文件名
    if(strcmp(argv[i], "-m") == 0) {
      if(i == argc - 1) {
        cerr << "Missing model name after -m\n";
        help(argv[0]);
        return -1;
      }
      ++i;
      model_image = argv[i];
    }//读取图片序列文件名
    else if(strcmp(argv[i], "-s") == 0) {
      if(i == argc - 1) {
        cerr << "Missing sequence format after -s\n";
        help(argv[0]);
        return -1;
      }
      ++i;
      sequence_format = argv[i];
      frame_source = sequence_source;
    }//读取视频文件名
    else if(strcmp(argv[i], "-v") == 0) {
      if(i == argc - 1) {
        cerr << "Missing  video filename after -v\n";
        help(argv[0]);
        return -1;
      }
      ++i;
      video_file = argv[i];
      frame_source = video_source;
    }
  }

  //初始化仿射变换参数
  affine_transformation_range range; 

  //初始化二维特征检测器
  detector = planar_pattern_detector_builder::build_with_cache(model_image.c_str(),
							       &range,
							       400,
							       5000,
							       0.0,
							       32, 7, 4,
							       30, 12,
							       10000, 200);

  if (!detector) {
    cerr << "Unable to build detector.\n";
    return -1;
  }

  //用检测器类中的方法设置最大检测点
  detector->set_maximum_number_of_points_to_detect(1000);

  tracker = new template_matching_based_tracker(); //初始化模板匹配跟踪器
  string trackerfn = model_image + string(".tracker_data");	//设置跟踪器文件名
  if (!tracker->load(trackerfn.c_str())) {
    cout << "Training template matching..."<<endl;	//读取跟踪器文件数据（模板数据？）
    tracker->learn(detector->model_image,
		   5, // number of used matrices (coarse-to-fine)用到的矩阵（由粗到细）
		   40, // max motion in pixel used to train to coarser matrix （训练粗矩阵的时候所用的最大动作数？）
		   20, 20, // defines a grid. Each cell will have one tracked point.定义一个矩形区域，每个单元里有一个跟踪点
		   detector->u_corner[0], detector->v_corner[1], //检测器roi区域
		   detector->u_corner[2], detector->v_corner[2],
		   40, 40, // neighbordhood for local maxima selection 计算极大值时对图像边缘的保护距离
		   10000 // number of training samples 训练样本数
		   );	//训练跟踪器
    tracker->save(trackerfn.c_str());	//存储跟踪器数据
  }
  tracker->initialize();	//跟踪器初始化
  //初始化文字标识
  cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX,
	     1.0, 1.0, 0.0,
	     3, 8);
 
  CvCapture * capture = 0;	//定义视频捕捉结构体
  IplImage * frame, * gray_frame = 0;
  int frame_id = 1;
  char seq_buffer[max_filename];	//定义图像序列存储空间

  cvNamedWindow("ferns-demo", 1 );
  //从摄像头读取图像
  if(frame_source == webcam_source) {
    capture = cvCaptureFromCAM(0);
  }//从视频文件读取图像
  else if(frame_source == video_source) {
    capture = cvCreateFileCapture(video_file.c_str());
  }

  //初始化计时器
  int64 timer = cvGetTickCount();

  bool stop = false;
  do {
    if(frame_source == webcam_source || frame_source == video_source) {
      if (cvGrabFrame(capture) == 0) break;
      frame = cvRetrieveFrame(capture);	//抓取一帧图像
    }
    else {
      snprintf(seq_buffer, max_filename, sequence_format.c_str(), frame_id);	//snprintf函数windows的标准输入输出库中没有，为Linux专用，从图像序列中读取一帧到缓冲区
      frame = cvLoadImage(seq_buffer, 1);	//从缓冲区加载图像
      ++frame_id;
    }

    if (frame == 0) break;

    if (gray_frame == 0)
      gray_frame = cvCreateImage(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);	//初始化灰度空白图像

    cvCvtColor(frame, gray_frame, CV_RGB2GRAY);	//将读取帧转为灰度图像
	//判断图像原点位置，若图像原点不是左上，则翻转图像
    if (frame->origin != IPL_ORIGIN_TL)
      cvFlip(gray_frame, gray_frame, 0);

    detect_and_draw(gray_frame);	//关键函数，检测跟踪并绘制结果函数

    int64 now = cvGetTickCount();	//取得计时结果
    double fps = 1e6 * cvGetTickFrequency()/double(now-timer);	//计算处理帧率
    timer = now;
    clog << "Detection frame rate: " << fps << " fps         \r";
	//选择工作模式
    int key = cvWaitKey(10);
    if (key >= 0) {
      switch(char(key)) {
      case '0': mode = 0; break;
      case '1': mode = 1; break;
      case '2': mode = 2; break;
      case '3': mode = 3; break;
      case '4': show_tracked_locations = !show_tracked_locations; break;
      case '5': show_keypoints = !show_keypoints; break;
      case 'q': stop = true; break;
      default: ;
      }
      cout << "mode=" << mode << endl;
    }
	//如果输入源是图像序列的话，则释放当前帧
    if(frame_source == sequence_source) {
      cvReleaseImage(&frame);
    }
  } while(!stop);

  clog << endl;
  delete detector;
  delete tracker;
  //清除缓存
  cvReleaseImage(&gray_frame);
  cvReleaseCapture(&capture);
  cvDestroyWindow("ferns-demo");

  return 0;
}
