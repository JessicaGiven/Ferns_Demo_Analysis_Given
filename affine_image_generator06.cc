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
#include <stdlib.h>
#include <math.h>

#include <iostream>
using namespace std;

#include "general.h"
#include "mcv.h"
#include "affine_image_generator06.h"

static const int prime = 307189;

//仿射变换图像生成器生成函数
affine_image_generator06::affine_image_generator06(void)
{
  original_image = 0;	//初始化原始图像
  generated_image = 0;	//初始化生成图像
  original_image_with_128_as_background = 0;	//背景图像（？）

  white_noise = new char[prime];	//白噪声（？）
  limited_white_noise = new int[prime];	//有限白噪声（？）

  set_default_values();	//设置仿射变换图像生成器初始参数

  save_images = false;	//存储图像标志
}

affine_image_generator06::~affine_image_generator06(void)
{
  if (original_image != 0)  cvReleaseImage(&original_image);
  if (generated_image != 0) cvReleaseImage(&generated_image);
  if (original_image_with_128_as_background) cvReleaseImage(&original_image_with_128_as_background);

  delete [] white_noise;
  delete [] limited_white_noise;
}

void affine_image_generator06::load_transformation_range(ifstream & f)
{
  transformation_range.load(f);
}

void affine_image_generator06::save_transformation_range(ofstream & f)
{
  transformation_range.save(f);
}

void affine_image_generator06::set_transformation_range(affine_transformation_range * range)
{
  transformation_range = *range;
}

void affine_image_generator06::generate_Id_image(void)
{
  generate_Id_affine_transformation();	//生成仿射变换矩阵
  generate_affine_image();	//生成仿射变换图像
}

void affine_image_generator06::generate_random_affine_image(void)
{
  generate_random_affine_transformation(); //生成随机仿射变换矩阵
  generate_affine_image();
}

void affine_image_generator06::save_generated_images(char * generic_name)
{
  save_images = true;
  strcpy(generic_name_of_saved_images, generic_name);
}

//仿射变换图像生成器默认参数设置函数
void affine_image_generator06::set_default_values(void)
{
  set_noise_level(20);	//设置噪声强度
  set_use_random_background(true);	//使用随机背景
  set_add_gaussian_smoothing(true);	//加入高斯平滑
  set_change_intensities(true);	//改变强度（？）
  add_noise = true;	//加入噪声
}

//噪声强度设置函数
void affine_image_generator06::set_noise_level(int noise_level)
{
  this->noise_level = noise_level;

  index_white_noise = 0;
  for(int i = 0; i < prime; i++) {
    limited_white_noise[i] = rand() % (2 * noise_level) - noise_level;
    white_noise[i] = char(rand() % 256);
  }
}

//设置原始图像函数
void affine_image_generator06::set_original_image(IplImage * p_original_image,
                                                  int generated_image_width,
                                                  int generated_image_height)
{
  if (original_image != 0) cvReleaseImage(&original_image);	//原始图像若已存在则直接释放
  original_image = cvCloneImage(p_original_image);	//从输入图像处克隆到原始图像

  if (generated_image != 0) cvReleaseImage(&generated_image);	//生成图像若已存在则直接释放
  if (generated_image_width < 0)	//生成图像宽度若小于0，则从摄入图像处克隆。反之，则新建原尺寸灰度图像
    generated_image = cvCloneImage(p_original_image);
  else
    generated_image = cvCreateImage(cvSize(generated_image_width, generated_image_height), IPL_DEPTH_8U, 1);
  
  if (original_image_with_128_as_background != 0) cvReleaseImage(&original_image_with_128_as_background);	//判断图像是否存在并释放
  original_image_with_128_as_background = cvCloneImage(p_original_image);	//从输入图象处克隆图像
  mcvReplace(original_image_with_128_as_background, 128, int(127));	//将图像中128的像素替换为127
}

/*设置mask函数

将指定矩形范围的图像覆盖为白色

*/

void affine_image_generator06::set_mask(int x_min, int y_min, int x_max, int y_max)
{
  for(int v = 0; v < original_image_with_128_as_background->height; v++) {
    unsigned char * row = mcvRow(original_image_with_128_as_background, v, unsigned char);
    for(int u = 0; u < original_image_with_128_as_background->width; u++)
      if (u < x_min || u > x_max || v < y_min || v > y_max)
	row[u] = 128;
  }
}

//仿射变换矩阵生成函数（变换坐标系，原理见笔记）
void affine_image_generator06::generate_affine_transformation(float a[6],
                                                              float initialTx, float initialTy,
                                                              float theta, float phi,
                                                              float lambda1, float lambda2,
                                                              float finalTx, float finalTy)
{
  float t1 = cos(theta);
  float t2 = cos(phi);
  float t4 = sin(theta);
  float t5 = sin(phi);
  float t7 = t1 * t2 + t4 * t5;
  float t8 = t7 * lambda1;
  float t12 = t1 * t5 - t4 * t2;
  float t13 = t12 * lambda2;
  float t15 = t8 * t2 + t13 * t5;
  float t18 = -t8 * t5 + t13 * t2;
  float t22 = -t12 * lambda1;
  float t24 = t7 * lambda2;
  float t26 = t22 * t2 + t24 * t5;
  float t29 = -t22 * t5 + t24 * t2;
  a[0] = t15;
  a[1] = t18;
  a[2] = t15 * initialTx + t18 * initialTy + finalTx;
  a[3] = t26;
  a[4] = t29;
  a[5] = t26 * initialTx + t29 * initialTy + finalTy;
}

void affine_image_generator06::generate_random_affine_transformation(void)
{
  float theta, phi, lambda1, lambda2;

  transformation_range.generate_random_parameters(theta, phi, lambda1, lambda2);
  generate_affine_transformation(a, 0, 0, theta, phi, lambda1, lambda2, 0, 0);

  int Tx, Ty;
  float nu0, nv0, nu1, nv1, nu2, nv2, nu3, nv3;

  affine_transformation(0.,                           0.,                            nu0, nv0);
  affine_transformation(float(original_image->width), 0.,                            nu1, nv1);
  affine_transformation(float(original_image->width), float(original_image->height), nu2, nv2);
  affine_transformation(0.,                           float(original_image->height), nu3, nv3);

  if (rand() % 2 == 0) Tx = -(int)min(min(nu0, nu1), min(nu2, nu3));
  else                 Tx = generated_image->width - (int)max(max(nu0, nu1), max(nu2, nu3));
  
  if (rand() % 2 == 0) Ty = -(int)min(min(nv0, nv1), min(nv2, nv3));
  else                 Ty = generated_image->height - (int)max(max(nv0, nv1), max(nv2, nv3));

  generate_affine_transformation(a, 0., 0., theta, phi, lambda1, lambda2, float(Tx), float(Ty));
}

void affine_image_generator06::generate_Id_affine_transformation(void)
{
  generate_affine_transformation(a, 0, 0 , 0, 0, 1, 1, 0, 0);	//生成仿射变换矩阵
}

void affine_image_generator06::affine_transformation(float p_a[6],
                                                     float u, float v,
                                                     float & nu, float & nv)
{
  nu = u * p_a[0] + v * p_a[1] + p_a[2];
  nv = u * p_a[3] + v * p_a[4] + p_a[5];
}

void affine_image_generator06::inverse_affine_transformation(float p_a[6],
							     float u, float v,
							     float & nu, float & nv)
{
  float det = p_a[0] * p_a[4] - p_a[3] * p_a[1];

  nu = 1.f / det * ( p_a[4] * (u - p_a[2]) - p_a[1] * (v - p_a[5]));
  nv = 1.f / det * (-p_a[3] * (u - p_a[2]) + p_a[0] * (v - p_a[5]));
}

void affine_image_generator06::affine_transformation(float u, float v, float & nu, float & nv)
{
  affine_transformation(a, u, v, nu, nv);
}

void affine_image_generator06::inverse_affine_transformation(float u, float v, float & nu, float & nv)
{
  inverse_affine_transformation(a, u, v, nu, nv);
}

//加噪函数
void affine_image_generator06::add_white_noise(IplImage * image, int gray_level_to_avoid)
{
  for(int y = 0; y < image->height; y++) {
    unsigned char * line = (unsigned char *)(image->imageData + y * image->widthStep);

    int * noise = limited_white_noise + rand() % (prime - image->width);

    for(int x = 0; x < image->width; x++) {
      int p = int(*line);

      if (p != gray_level_to_avoid) {
	p += *noise;

	*line = (p > 255) ? 255 : (p < 0) ? 0 : (unsigned char)p;
      }
      line++;
      noise++;
    }
  }
}

//指定值噪声替代函数
void affine_image_generator06::replace_by_noise(IplImage * image, int value)
{
  for(int y = 0; y < image->height; y++) {
    unsigned char * row = mcvRow(image, y, unsigned char);

    for(int x = 0; x < image->width; x++)
      if (int(row[x]) == value) {
	row[x] = white_noise[index_white_noise];
	index_white_noise++;
	if (index_white_noise >= prime) index_white_noise = 1 + rand() % 6;
      }
  }
}

void affine_image_generator06::generate_affine_image(void)
{
  CvMat A = cvMat(2, 3, CV_32F, a);	//初始化仿射变换矩阵

  if (use_random_background)
    cvSet(generated_image, cvScalar(128));	//将生成图像所有像素设置成128
  else
    cvSet(generated_image, cvScalar(rand() % 256));	//将生成图像所有像素随机设置

  //对原始图片进行仿射变换，生成生成图像
  cvWarpAffine(original_image_with_128_as_background, generated_image, &A,
	       CV_INTER_NN + CV_WARP_FILL_OUTLIERS /* + CV_WARP_INVERSE_MAP*/, cvScalarAll(128));

  if (use_random_background)
    replace_by_noise(generated_image, 128);	//使用白噪声替代图像中像素值为128的点

  //使用随机参数对图像进行高斯平滑（为什么用随机参数？？？）
  if (add_gaussian_smoothing && rand() % 3 == 0) {
    int aperture = 3 + 2 * (rand() % 3);
    cvSmooth(generated_image, generated_image, CV_GAUSSIAN, aperture, aperture);
  }

  //对图像进行线性变换（？）
  if (change_intensities) cvCvtScale(generated_image, generated_image, rand(0.8f, 1.2f), rand(-10, 10));

  //   mcvSaveImage("g.bmp", generated_image);
  //   exit(0);

  //对图像加入白噪声
  if (noise_level > 0 && add_noise)
    if (use_random_background) add_white_noise(generated_image);
    else add_white_noise(generated_image, 128);

  if (save_images) {
    static int n = 0;
    mcvSaveImage(generic_name_of_saved_images, n, generated_image);	//存储图像
    n++;
  }
}
