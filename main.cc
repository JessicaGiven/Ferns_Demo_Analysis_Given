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

ö����������Դ��
����ͷ��ͼƬ�����豸����Ƶ�ļ�

*/
enum source_type {webcam_source, sequence_source, video_source};

//����һ���µĶ�ά�����������
planar_pattern_detector * detector;
//����һ���µ�ģ��ƥ�������
template_matching_based_tracker * tracker;

int mode = 2;
bool show_tracked_locations = true;
bool show_keypoints = true;
//��ʼ�����ֱ�ʾ�ṹ
CvFont font;

/*

draw_quadrangle����

�����ı��ο�

*/

void draw_quadrangle(IplImage * frame,	//����ͼ��
		     int u0, int v0,	
		     int u1, int v1,
		     int u2, int v2,
		     int u3, int v3,	//�ı����Ķ�������
		     CvScalar color, int thickness = 1)	//color:�ı�����ɫ��thinckness:�ı����߶δ�ϸ
{
  cvLine(frame, cvPoint(u0, v0), cvPoint(u1, v1), color, thickness);
  cvLine(frame, cvPoint(u1, v1), cvPoint(u2, v2), color, thickness);
  cvLine(frame, cvPoint(u2, v2), cvPoint(u3, v3), color, thickness);
  cvLine(frame, cvPoint(u3, v3), cvPoint(u0, v0), color, thickness);
}

/*

draw_detected_position����

���Ƽ��λ�õ��ı��ο򣨰�ɫ��

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

draw_initial_rectangle����

��ʼ���ı��ο򣨻�ɫ��

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

draw_tracked_position����

���Ƹ���λ�õ��ı��ο򣨰�ɫ��

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

draw_tracked_locations����

���Ƹ���λ�õ�Բ�ο򣨰�ɫ��

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

draw_detected_keypoints����

����������λ�õ�Բ�ο򣨻�ɫ��

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

draw_recognized_keypoints����

����Ǳ��ƥ��������λ�õ�Բ�ο򣨰�ɫ��������

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

detect_and_draw����

0: Detect when tracking fails or for initialization then track.	��׷��ʧ��ʱ���м�⣬���߳�ʼ��׷��
1: Track only	//ֻ׷��
2: Detect only (DEFAULT)	//ֻ��⣨Ĭ�ϣ�
3: Detect + track in every frame	//��ÿһ֡���ж�����׷�ٺͼ��

The number  keys 4&5  can be  used to turn  on/off the  recognized and
 detected keypoints, respectively.

 ����4��5���Էֱ�����Ƿ�ʶ��ͼ��������

*/

void detect_and_draw(IplImage * frame)
{
	static bool last_frame_ok=false;	//��һ֡�ɹ�������

	if (mode == 1 || ((mode==0) && last_frame_ok)) {
		bool ok = tracker->track(frame);	//�Ե�ǰ֡���и��٣����ظ����Ƿ�ɹ����
		last_frame_ok=ok;


		if (!ok) {
			if (mode==0) return detect_and_draw(frame);	//���׷��ʧ�ܣ������ظú���һ���������ٽ׶ν�����׶�
			else {
				draw_initial_rectangle(frame, tracker);	//��ʼ���ı��ο�
				tracker->initialize();	//��ʼ��������
			}
		} else {
			draw_tracked_position(frame, tracker);	//���Ƹ���λ��
			if (show_tracked_locations) draw_tracked_locations(frame, tracker);	//���Ƹ���λ�ã��任��
		}
		cvPutText(frame, "template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//����Ƶ�ϵ���template-based 3D tracking
	} else {
		detector->detect(frame);	//�Ե�ǰ֡���м��
		
		if (detector->pattern_is_detected) {
			last_frame_ok=true;	//��⵽��������λ�ɹ����
			//���ü�����Ľ���Ը��������г�ʼ��
			tracker->initialize(detector->detected_u_corner[0], detector->detected_v_corner[0],
					detector->detected_u_corner[1], detector->detected_v_corner[1],
					detector->detected_u_corner[2], detector->detected_v_corner[2],
					detector->detected_u_corner[3], detector->detected_v_corner[3]);

			if (mode == 3 && tracker->track(frame)) {
				//����������λ��
				if (show_keypoints) {
					draw_detected_keypoints(frame, detector); //�������ާ��������⵽�ĵ�ǰ֡�ؼ���
					draw_recognized_keypoints(frame, detector); //���ƾ����ȶԵĵ�ǰ֡�ؼ���
				}
				draw_tracked_position(frame, tracker);	//����׷��λ��
				if (show_tracked_locations) draw_tracked_locations(frame, tracker);	//����׷��λ��(�任��

				cvPutText(frame, "detection+template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//����Ƶ�ϵ���detection+template-based 3D tracking
			} else {//����������λ��
				if (show_keypoints) {
					draw_detected_keypoints(frame, detector);	 
					draw_recognized_keypoints(frame, detector);
				}
				draw_detected_position(frame, detector);	//���Ƽ��λ��
				cvPutText(frame, "detection", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//����Ƶ�ϵ���template-based 3D tracking
			}
		} else {
			last_frame_ok=false;
			if (show_keypoints) draw_detected_keypoints(frame, detector);	//����������λ��

			if (mode == 3)
				cvPutText(frame, "detection + template-based 3D tracking", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//����Ƶ�ϵ���detection + template-based 3D tracking
			else
				cvPutText(frame, "detection", cvPoint(10, 30), &font, cvScalar(255, 255, 255));	//����Ƶ�ϵ���detection
		}
	}

	cvShowImage("ferns-demo", frame);	//��ʾ���֮���֡
}

//�����������
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

//������
int main(int argc, char ** argv)
{
  string model_image     = "model.bmp";	//����ģ��ͼƬ�ļ���
  string sequence_format = "";	//���и�ʽ
  string video_file = "";	//��Ƶ�ļ���
  source_type frame_source = webcam_source;	//Ĭ������ģʽ������ͷģʽ

  //���ܼ�����������������������help
  for(int i = 0; i < argc; ++i) {
    if(strcmp(argv[i], "-h") == 0) {
      help(argv[0]);
      return 0;
    }
	//��ȡģ���ļ���
    if(strcmp(argv[i], "-m") == 0) {
      if(i == argc - 1) {
        cerr << "Missing model name after -m\n";
        help(argv[0]);
        return -1;
      }
      ++i;
      model_image = argv[i];
    }//��ȡͼƬ�����ļ���
    else if(strcmp(argv[i], "-s") == 0) {
      if(i == argc - 1) {
        cerr << "Missing sequence format after -s\n";
        help(argv[0]);
        return -1;
      }
      ++i;
      sequence_format = argv[i];
      frame_source = sequence_source;
    }//��ȡ��Ƶ�ļ���
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

  //��ʼ������任����
  affine_transformation_range range; 

  //��ʼ����ά���������
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

  //�ü�������еķ�������������
  detector->set_maximum_number_of_points_to_detect(1000);

  tracker = new template_matching_based_tracker(); //��ʼ��ģ��ƥ�������
  string trackerfn = model_image + string(".tracker_data");	//���ø������ļ���
  if (!tracker->load(trackerfn.c_str())) {
    cout << "Training template matching..."<<endl;	//��ȡ�������ļ����ݣ�ģ�����ݣ���
    tracker->learn(detector->model_image,
		   5, // number of used matrices (coarse-to-fine)�õ��ľ����ɴֵ�ϸ��
		   40, // max motion in pixel used to train to coarser matrix ��ѵ���־����ʱ�����õ������������
		   20, 20, // defines a grid. Each cell will have one tracked point.����һ����������ÿ����Ԫ����һ�����ٵ�
		   detector->u_corner[0], detector->v_corner[1], //�����roi����
		   detector->u_corner[2], detector->v_corner[2],
		   40, 40, // neighbordhood for local maxima selection ���㼫��ֵʱ��ͼ���Ե�ı�������
		   10000 // number of training samples ѵ��������
		   );	//ѵ��������
    tracker->save(trackerfn.c_str());	//�洢����������
  }
  tracker->initialize();	//��������ʼ��
  //��ʼ�����ֱ�ʶ
  cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX,
	     1.0, 1.0, 0.0,
	     3, 8);
 
  CvCapture * capture = 0;	//������Ƶ��׽�ṹ��
  IplImage * frame, * gray_frame = 0;
  int frame_id = 1;
  char seq_buffer[max_filename];	//����ͼ�����д洢�ռ�

  cvNamedWindow("ferns-demo", 1 );
  //������ͷ��ȡͼ��
  if(frame_source == webcam_source) {
    capture = cvCaptureFromCAM(0);
  }//����Ƶ�ļ���ȡͼ��
  else if(frame_source == video_source) {
    capture = cvCreateFileCapture(video_file.c_str());
  }

  //��ʼ����ʱ��
  int64 timer = cvGetTickCount();

  bool stop = false;
  do {
    if(frame_source == webcam_source || frame_source == video_source) {
      if (cvGrabFrame(capture) == 0) break;
      frame = cvRetrieveFrame(capture);	//ץȡһ֡ͼ��
    }
    else {
      snprintf(seq_buffer, max_filename, sequence_format.c_str(), frame_id);	//snprintf����windows�ı�׼�����������û�У�ΪLinuxר�ã���ͼ�������ж�ȡһ֡��������
      frame = cvLoadImage(seq_buffer, 1);	//�ӻ���������ͼ��
      ++frame_id;
    }

    if (frame == 0) break;

    if (gray_frame == 0)
      gray_frame = cvCreateImage(cvSize(frame->width,frame->height), IPL_DEPTH_8U, 1);	//��ʼ���Ҷȿհ�ͼ��

    cvCvtColor(frame, gray_frame, CV_RGB2GRAY);	//����ȡ֡תΪ�Ҷ�ͼ��
	//�ж�ͼ��ԭ��λ�ã���ͼ��ԭ�㲻�����ϣ���תͼ��
    if (frame->origin != IPL_ORIGIN_TL)
      cvFlip(gray_frame, gray_frame, 0);

    detect_and_draw(gray_frame);	//�ؼ������������ٲ����ƽ������

    int64 now = cvGetTickCount();	//ȡ�ü�ʱ���
    double fps = 1e6 * cvGetTickFrequency()/double(now-timer);	//���㴦��֡��
    timer = now;
    clog << "Detection frame rate: " << fps << " fps         \r";
	//ѡ����ģʽ
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
	//�������Դ��ͼ�����еĻ������ͷŵ�ǰ֡
    if(frame_source == sequence_source) {
      cvReleaseImage(&frame);
    }
  } while(!stop);

  clog << endl;
  delete detector;
  delete tracker;
  //�������
  cvReleaseImage(&gray_frame);
  cvReleaseCapture(&capture);
  cvDestroyWindow("ferns-demo");

  return 0;
}
