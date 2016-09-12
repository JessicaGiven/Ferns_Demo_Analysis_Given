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
#include <fstream>
using namespace std;

#include "homography_estimator.h"

#include "template_matching_based_tracker.h"
#include "mcv.h"

template_matching_based_tracker::template_matching_based_tracker(void)
{
}

bool template_matching_based_tracker::load(const char * filename)
{

  ifstream f(filename);

  if (!f.good()) 
	  return false;

  cout << "Loading " << filename << "..." << endl;

  U0 = cvCreateMat(8, 1, CV_32F);
  u0 = U0->data.fl;
  for(int i = 0; i < 8; i++)
    f >> u0[i];
  f >> nx >> ny;
  m = new int[2 * nx * ny];
  for(int i = 0; i < nx * ny; i++)
    f >> m[2 * i] >> m[2 * i + 1];

  U = cvCreateMat(8, 1, CV_32F);
  u = U->data.fl;

  I0 = cvCreateMat(nx * ny, 1, CV_32F);
  i0 = I0->data.fl;

  for(int i = 0; i < nx * ny; i++)
    f >> i0[i];

  I1 = cvCreateMat(nx * ny, 1, CV_32F);
  i1 = I1->data.fl;
  DI = cvCreateMat(nx * ny, 1, CV_32F);
  DU = cvCreateMat(8, 1, CV_32F);
  du = DU->data.fl;

  f >> number_of_levels;

  As = new CvMat*[number_of_levels];
  for(int i = 0; i < number_of_levels; i++) {
    As[i] = cvCreateMat(8, nx * ny, CV_32F);
    for(int j = 0; j < 8; j++)
      for(int k = 0; k < nx * ny; k++) {
	float v;
	f >> v;
	cvmSet(As[i], j, k, v);
      }
  }

  if (!f.good()) 
	  return false;

  cout << "Done." << endl;
  return true;
}

void template_matching_based_tracker::save(const char * filename)
{
  ofstream f(filename);

  for(int i = 0; i < 8; i++)
    f << u0[i] << " ";
  f << endl;
  f << nx << " " << ny << endl;
  for(int i = 0; i < nx * ny; i++)
    f << m[2 * i] << " " << m[2 * i + 1] << endl;
  for(int i = 0; i < nx * ny; i++)
    f << i0[i] << " ";
  f << endl;
  f << number_of_levels << endl;
  for(int i = 0; i < number_of_levels; i++) {
    for(int j = 0; j < 8; j++) {
      for(int k = 0; k < nx * ny; k++)
	f << cvmGet(As[i], j, k) << " ";
      f << endl;
    }
  }
  f.close();
}

void template_matching_based_tracker::move(int x, int y, float & x2, float & y2, int amp)
{
  int d = rand() % amp;
  float a = float(rand() % 720) * 3.14159 * 2.0 / 720;

  x2 = x + d * cosf(a);
  y2 = y + d * sinf(a);
}

//极大值矩阵标准化函数（？？？）
bool template_matching_based_tracker::normalize(CvMat * V)
{
  float sum = 0.0, sum2 = 0.0;
  float * v = V->data.fl;

  for(int i = 0; i < V->rows; i++) {
    sum += v[i];	//对矩阵前行数个元素求和
    sum2 += v[i] * v[i];	//对矩阵前行数个元素平方并求和
  }

  // Not enough contrast,  better not put this sample into the training set:
  if (sum < (V->rows * 10))	//如果求和结果小于十倍行数，则认定其对比度不足，不将其加入训练集中
    return false;

  float mean = sum / V->rows;	//求和平均值
  float inv_sigma = 1.0 / sqrt(sum2 / V->rows - mean * mean);	//计算方差的倒数（？）

  // Not enough contrast,  better not put this sample into the training set:
  if (!finite(inv_sigma))	//判断方差倒数是否为无穷，如果是无穷，则不加入训练集
    return false;

  for(int i = 0; i < V->rows; i++)	
    v[i] = inv_sigma * (v[i] - mean);	//将极大值矩阵前行数个元素标准化

  return true;
}

//随机加噪函数
void template_matching_based_tracker::add_noise(CvMat * V)
{
  float * v = V->data.fl;

  float gamma = 0.5 + (3 - 0.7) * float(rand()) / RAND_MAX;
  for(int i = 0; i < V->rows; i++) {
    v[i] = pow(v[i], gamma) + rand() % 10 - 5;
    if (v[i] < 0) v[i] = 0;
    if (v[i] > 255) v[i] = 255;
  }
}

//图像梯度计算函数
IplImage * template_matching_based_tracker::compute_gradient(IplImage * image)
{
  IplImage * dx = cvCreateImage(cvSize(image->width, image->height),
				IPL_DEPTH_16S, 1);	//初始化输入图像
  IplImage * dy = cvCreateImage(cvSize(image->width, image->height),
				IPL_DEPTH_16S, 1);	//初始化输入图像
  IplImage * result = cvCreateImage(cvSize(image->width, image->height),
				    IPL_DEPTH_16S, 1);	//初始化输出图像
  cvSobel(image, dx, 1, 0, 3);	//用sobel算子计算图像x方向梯度
  cvSobel(image, dy, 0, 1, 3);	//计算y方向梯度
  cvMul(dx, dx, dx);	//G(x)^2
  cvMul(dy, dy, dy);	//G(y)^2
  cvAdd(dx, dy, result);	//G = (G(x)^2 + G(y)^2)^(1/2) (不开方？）

  cvReleaseImage(&dx);
  cvReleaseImage(&dy);	//释放内存

  return result;
}

/*极大值计算函数

通过遍历的方法，计算邻域极大值

*/
void template_matching_based_tracker::get_local_maximum(IplImage * G,
							int xc, int yc, int w, int h,
							int & xm, int & ym)
{
  int max = -1;	//初始化最大值缓存变量
  for(int v = yc - h / 2; v <= yc + h / 2; v++) {//纵向遍历
    short * row = mcvRow(G, v, short);	//取图像一行
    for(int u = xc - w / 2; u <= xc + w / 2; u++)//横向遍历
      if (row[u] > max) {
	max = row[u];
	xm = u;
	ym = v;	//记录最大值以其坐标
      }
  }
}

/*计算极大值点（？）坐标

u0是检测器输出的roi坐标。

nx,ny定义了一个矩形区域，nx和ny分别为矩形区域的长宽，这个区域中的每一个元素都是一个跟踪点。

函数首先计算了矩形区域需要平移的步长，然后依次平移该矩形区域，每一次平移都计算矩形区域中的极大值，总共平移nx*ny次。

bx,by为图像边缘保护距离，目的是防止算法计算图像边缘梯度。

*/
void template_matching_based_tracker::find_2d_points(IplImage * image, int bx, int by)
{
  IplImage * gradient = compute_gradient(image);	//计算图像梯度

  const float stepx = float(u0[2] - u0[0] - 2 * bx) / (nx - 1);	//计算矩形区域横向平移步长
  const float stepy = float(u0[5] - u0[1] - 2 * by) / (ny - 1);	//计算矩形区域纵向平移步长
  for(int j = 0; j < ny; j++)
    for(int i = 0; i < nx; i++)
      get_local_maximum(gradient,	//梯度图像
			int(u0[0] + bx + i * stepx + 0.5),	//求极大值的范围是一个矩形，此参数是矩形的左上角横坐标
			int(u0[1] + by + j * stepy + 0.5),	//上述矩形左上角纵坐标
			int(stepx), int(stepy),	//上述矩形长、宽
			m[2 * (j * nx + i)],
			m[2 * (j * nx + i) + 1]);	//存储求得的所有极大值

  cvReleaseImage(&gradient);
}

//以矩阵计算
void template_matching_based_tracker::compute_As_matrices(IplImage * image, int max_motion, int Ns)
{
  As = new CvMat*[number_of_levels];	//新建多个矩阵

  //新建矩阵
  CvMat * Y = cvCreateMat(8, Ns, CV_32F);
  CvMat * H = cvCreateMat(nx * ny, Ns, CV_32F);
  CvMat * HHt = cvCreateMat(nx * ny, nx * ny, CV_32F);
  CvMat * HHt_inv = cvCreateMat(nx * ny, nx * ny, CV_32F);
  CvMat * Ht_HHt_inv = cvCreateMat(Ns, nx * ny, CV_32F);

  //新建单应性类
  homography06 ft;

  //
  for(int level = 0; level < number_of_levels; level++) {

    int n = 0;
    while(n < Ns) {
      cout << "Level: " << level << " (" << n << "/" << Ns << " samples done)" << char(13) << flush;

      float u1[8];

      float k = exp(1. / (number_of_levels - 1) * log(5.0 / max_motion));
      float amp = pow(k, float(level)) * max_motion;

      for(int i = 0; i < 4; i++)
	move(u0[2 * i], u0[2 * i + 1], u1[2 * i], u1[2 * i + 1], amp);

      for(int i = 0; i < 8; i++)
	cvmSet(Y, i, n, u1[i] - u0[i]);

      he.estimate(&ft,
		  u0[0], u0[1], u1[0], u1[1],
		  u0[2], u0[3], u1[2], u1[3],
		  u0[4], u0[5], u1[4], u1[5],
		  u0[6], u0[7], u1[6], u1[7]);

      for(int i = 0; i < nx * ny; i++) {
	int x1, y1;

	ft.transform_point(m[2 * i], m[2 * i + 1], x1, y1);	//x1,y1为单应变换后的坐标
	i1[i] = mcvRow(image, y1, unsigned char)[x1];	//利用单应变换后的坐标取出输入图像对应部分
      }
      add_noise(I1);	//对输入图像加入随机噪声
      bool ok = normalize(I1);	//标准化
      if (ok) {
	for(int i = 0; i < nx * ny; i++)
	  cvmSet(H, i, n, i1[i] - i0[i]);	//将单应变换且加噪后的极大值图像与极大值图像相减，存入H
	n++;
      }
    }
    cout << "Level: " << level << "                                        " << endl;
    cout << " - " << n << " training samples generated." << endl;

    As[level] = cvCreateMat(8, nx * ny, CV_32F);

	//求矩阵H平方
    cout << " - computing HHt..." << flush;
    cvGEMM(H, H, 1.0, 0, 0.0, HHt, CV_GEMM_B_T);
    cout << "done." << endl;

	//求矩阵H平方的逆
    cout << " - inverting HHt..." << flush;
    if (cvInvert(HHt, HHt_inv, CV_SVD_SYM) == 0) {
      cerr << "> In template_matching_based_tracker::compute_As_matrices :" << endl;
      cerr << " Can't compute HHt matrix inverse!" << endl;
      cerr << " damn!" << endl;	//该死的！
      exit(-1);
    }
    cout << "done." << endl;

	//求矩阵H和矩阵H平方逆的乘积
    cout << " - computing H(HHt)^-1..." << flush;
    cvGEMM(H, HHt_inv, 1.0, 0, 0.0, Ht_HHt_inv, CV_GEMM_A_T);
    cout << "done." << endl;

    //求上述结果与Y的乘积
    cout << " - computing YH(HHt)^-1..." << flush;
    cvMatMul(Y, Ht_HHt_inv, As[level]);	//As为此函数输出结果
    cout << "done." << endl;
  }

  cvReleaseMat(&Y);
  cvReleaseMat(&H);
  cvReleaseMat(&HHt);
  cvReleaseMat(&HHt_inv);
  cvReleaseMat(&Ht_HHt_inv);
}
//跟踪器训练函数
void template_matching_based_tracker::learn(IplImage * image,
					    int number_of_levels, int max_motion, int nx, int ny,	////用到的矩阵（由粗到细），（训练粗矩阵的时候所用的最大动作数？），定义一个框架，每个单元里有一个跟踪点
					    int xUL, int yUL,
					    int xBR, int yBR,	//检测器感兴趣区域（？？？）
					    int bx, int by,	//供选择的最大邻域
					    int Ns)	//训练样本数
{
  this->number_of_levels = number_of_levels;	//类内部需要相互调用的时候用指针this
  this->nx = nx;
  this->ny = ny;

  m = new int[2 * nx * ny];
  U0 = cvCreateMat(8, 1, CV_32F);	//初始化一个矩阵，8*1
  u0 = U0->data.fl;	//指针指向图像数据，以32bits浮点数为单位
  u0[0] = xUL; u0[1] = yUL;
  u0[2] = xBR; u0[3] = yUL;
  u0[4] = xBR; u0[5] = yBR;
  u0[6] = xUL; u0[7] = yBR;	//检测器roi区域

  find_2d_points(image, bx, by);	//计算roi的中极大值

  U = cvCreateMat(8, 1, CV_32F);	//初始化一个矩阵，8*1
  u = U->data.fl;	//指针指向图像数据，以32bits浮点数为单位

  I0 = cvCreateMat(nx * ny, 1, CV_32F);	//初始化一个矩阵，nx*ny
  i0 = I0->data.fl;	//指针指向图像数据，以32bits浮点数为单位

  for(int i = 0; i < nx * ny; i++)
    i0[i] = mcvRow(image, m[2 * i + 1], unsigned char)[ m[2 * i] ];	//将极大值点数据导出至i0
  bool ok = normalize(I0);	//将极大值矩阵标准化
  if (!ok) {
    cerr << "> in template_matching_based_tracker::learn :" << endl;
    cerr << "> Template matching: image has not enough contrast." << endl;
    return ;
  }

  I1 = cvCreateMat(nx * ny, 1, CV_32F);
  i1 = I1->data.fl;
  DI = cvCreateMat(nx * ny, 1, CV_32F);
  DU = cvCreateMat(8, 1, CV_32F);
  du = DU->data.fl;

  compute_As_matrices(image, max_motion, Ns);
}

//跟踪器初始化函数
void template_matching_based_tracker::initialize(void)
{
  cvCopy(U0, U);

  // Set f to Id:对u0进行单应变换
  he.estimate(&f,
	      u0[0], u0[1], u[0], u[1],
	      u0[2], u0[3], u[2], u[3],
	      u0[4], u0[5], u[4], u[5],
	      u0[6], u0[7], u[6], u[7]);
}

void template_matching_based_tracker::initialize(int x0, int y0,
						 int x1, int y1,
						 int x2, int y2,
						 int x3, int y3)
{
  u[0] = x0;  u[1] = y0;
  u[2] = x1;  u[3] = y1;
  u[4] = x2;  u[5] = y2;
  u[6] = x3;  u[7] = y3;

  he.estimate(&f,
	      u0[0], u0[1], u[0], u[1],
	      u0[2], u0[3], u[2], u[3],
	      u0[4], u0[5], u[4], u[5],
	      u0[6], u0[7], u[6], u[7]);
}

// void homography_from_4pt(const float *x, const float *y, const float *z, const float *w, float cgret[8])
// {
//         double t1 = x[0];
//         double t2 = z[0];
//         double t4 = y[1];
//         double t5 = t1 * t2 * t4;
//         double t6 = w[1];
//         double t7 = t1 * t6;
//         double t8 = t2 * t7;
//         double t9 = z[1];
//         double t10 = t1 * t9;
//         double t11 = y[0];
//         double t14 = x[1];
//         double t15 = w[0];
//         double t16 = t14 * t15;
//         double t18 = t16 * t11;
//         double t20 = t15 * t11 * t9;
//         double t21 = t15 * t4;
//         double t24 = t15 * t9;
//         double t25 = t2 * t4;
//         double t26 = t6 * t2;
//         double t27 = t6 * t11;
//         double t28 = t9 * t11;
//         double t30 = 0.1e1 / (-t24 + t21 - t25 + t26 - t27 + t28);
//         double t32 = t1 * t15;
//         double t35 = t14 * t11;
//         double t41 = t4 * t1;
//         double t42 = t6 * t41;
//         double t43 = t14 * t2;
//         double t46 = t16 * t9;
//         double t48 = t14 * t9 * t11;
//         double t51 = t4 * t6 * t2;
//         double t55 = t6 * t14;
//         cgret[0] = -(-t5 + t8 + t10 * t11 - t11 * t7 - t16 * t2 + t18 - t20 + t21 * t2) * t30;
//         cgret[1] = (t5 - t8 - t32 * t4 + t32 * t9 + t18 - t2 * t35 + t27 * t2 - t20) * t30;
//         cgret[2] = t1;
//         cgret[3] = (-t9 * t7 + t42 + t43 * t4 - t16 * t4 + t46 - t48 + t27 * t9 - t51) * t30;
//         cgret[4] = (-t42 + t41 * t9 - t55 * t2 + t46 - t48 + t55 * t11 + t51 - t21 * t9) * t30;
//         cgret[5] = t14;
//         cgret[6] = (-t10 + t41 + t43 - t35 + t24 - t21 - t26 + t27) * t30;
//         cgret[7] = (-t7 + t10 + t16 - t43 + t27 - t28 - t21 + t25) * t30;
//         //cgret[8] = 1;
// 	}

//跟踪函数
bool template_matching_based_tracker::track(IplImage * input_frame)
{
  homography06 fs;

  for(int level = 0; level < number_of_levels; level++) {
    for(int iter = 0; iter < 5; iter++) {
      for(int i = 0; i < nx * ny; i++) {
	int x1, y1;

	f.transform_point(m[2 * i], m[2 * i + 1], x1, y1);
	//判断之前极大值矩阵的长宽，如果小于0或者其长宽大于等于输入帧时，则判定跟踪失败
	if (x1 < 0 || y1 < 0 || x1 >= input_frame->width || y1 >= input_frame->height)
	  return false;
	//取出输入帧对应范围的图像
	i1[i] = mcvRow(input_frame, y1, unsigned char)[x1];
      }
      normalize(I1);	//标准化
      cvSub(I1, I0, DI);	//输入帧-极大值矩阵

      cvMatMul(As[level], DI, DU);	//compute函数输出结果与上述结果进行相乘
	  //下面的代码看不太懂
      he.estimate(&fs,
		  u0[0],  u0[1],  u0[0] - du[0], u0[1] - du[1],
		  u0[2],  u0[3],  u0[2] - du[2], u0[3] - du[3],
		  u0[4],  u0[5],  u0[4] - du[4], u0[5] - du[5],
		  u0[6],  u0[7],  u0[6] - du[6], u0[7] - du[7]);

      cvMatMul(&f, &fs, &f);

      float norm = 0;
      for(int i = 0; i < 9; i++) norm += f.data.fl[i] * f.data.fl[i];
      norm = sqrtf(norm);
      for(int i = 0; i < 9; i++) f.data.fl[i] /= norm;
    }
  }
  //输出结果是指示跟踪目标的跟踪框坐标
  f.transform_point(u0[0], u0[1], u[0], u[1]);
  f.transform_point(u0[2], u0[3], u[2], u[3]);
  f.transform_point(u0[4], u0[5], u[4], u[5]);
  f.transform_point(u0[6], u0[7], u[6], u[7]);

  return true;
}
