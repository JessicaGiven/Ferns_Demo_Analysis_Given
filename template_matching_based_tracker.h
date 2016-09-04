/*
  Copyright 2007 Computer Vision Lab,
  Ecole Polytechnique Federale de Lausanne (EPFL), Switzerland.
  All rights reserved.

  Author: Vincent Lepetit (http://cvlab.epfl.ch/~lepetit)

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
#ifndef template_matching_based_tracker_h
#define template_matching_based_tracker_h

#include <cv.h>
#include "homography06.h"
#include "homography_estimator.h"

//模板匹配跟踪器类
class template_matching_based_tracker
{
 public:
  template_matching_based_tracker(void);

  bool load(const char * filename);	//跟踪器文件读取函数
  void save(const char * filename);	//跟踪器文件存储函数
  //跟踪器训练函数
  void learn(IplImage * image,	//输入模板文件
	     int number_of_levels, int max_motion, int nx, int ny,	//用到的矩阵（由粗到细），（训练粗矩阵的时候所用的最大动作数？），定义一个框架，每个单元里有一个跟踪点
	     int xUL, int yUL,
	     int xBR, int yBR,	//检测器感兴趣区域（？？？）
	     int bx, int by,	//供选择的最大邻域
	     int Ns);	//训练样本数

  //初始化跟踪器函数
  void initialize(void);	
  //初始化跟踪器函数（带参数）
  void initialize(int u0, int v0,
		  int u1, int v1,
		  int u2, int v2,
		  int u3, int v3);
  //跟踪函数
  bool track(IplImage * input_frame);

  homography06 f;

  //private:
  void find_2d_points(IplImage * image, int bx, int by);
  void compute_As_matrices(IplImage * image, int max_motion, int Ns);
  void move(int x, int y, float & x2, float & y2, int amp);
  bool normalize(CvMat * V);
  void add_noise(CvMat * V);
  IplImage * compute_gradient(IplImage * image);
  void get_local_maximum(IplImage * G,
			 int xc, int yc, int w, int h,
			 int & xm, int & ym);

  homography_estimator he;

  int * m;
  CvMat ** As;
  CvMat * U0, * U, * I0, * DU, * DI, * I1;
  float * u0, * u, * i0, * du, * i1;
  int number_of_levels, nx, ny;
};


#endif
