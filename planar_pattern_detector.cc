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
#include <algorithm>
#include <fstream>
using namespace std;

#include "mcv.h"
#include "planar_pattern_detector.h"
#include "buffer_management.h"

//初始化空白检测器函数
planar_pattern_detector::planar_pattern_detector(void)
{
  model_image = 0;

  image_generator = new affine_image_generator06(); //新建仿射变换图像生成器
  point_detector = new pyr_yape06();	//新建点检测器

  //设置部分参数
  model_points = detected_points = 0;
  match_probabilities = 0;

  maximum_number_of_points_to_detect = 500;	//设置最大检测点数

  H_estimator = new homography_estimator;	//新建单应估计器
}

planar_pattern_detector::~planar_pattern_detector(void)
{
  if (model_image != 0) cvReleaseImage(&model_image);
  if (model_points != 0) delete [] model_points;
  delete_managed_buffer(detected_points);

  delete image_generator;

  //delete point_detector; <= LE DESTRUCTEUR DE PYR_YAPE06 PLANTE !!!

  if (match_probabilities != 0) {
    for(int i = 0; i < number_of_model_points; i++)
      delete [] match_probabilities[i];
    delete [] match_probabilities;
  }
}

//检测器数据文件读取函数
bool planar_pattern_detector::load(const char * filename)
{
  ifstream f(filename, ios::binary);

  if (!f.is_open()) return false;

  cout << "> [planar_pattern_detector::load] Loading detector file " << filename << " ... " << endl;

  bool result = load(f);

  f.close();

  cout << "> [planar_pattern_detector::load] Ok." << endl;

  return result;
}

bool planar_pattern_detector::save(const char * filename)
{
  ofstream f(filename, ios::binary);

  if (!f.is_open()) {
    cerr << ">! [planar_pattern_detector::save] Error saving file " << filename << "." << endl;
    return false;
  }

  cout << "> [planar_pattern_detector::save] Saving detector file " << filename << " ... " << flush;

  bool result = save(f);

  f.close();

  cout << "> [planar_pattern_detector::save] Ok." << endl;

  return result;
}

bool planar_pattern_detector::load(ifstream & f)
{
  f >> image_name;

  cout << "> [planar_pattern_detector::load] Image name: " << image_name << endl;

  for(int i = 0; i < 4; i++) f >> u_corner[i] >> v_corner[i];

  f >> patch_size >> yape_radius >> number_of_octaves;
  cout << "> [planar_pattern_detector::load] Patch size = " << patch_size
       << ", Yape radius = " << yape_radius
       << ", Number of octaves = " << number_of_octaves
       << "." << endl;

  pyramid = new fine_gaussian_pyramid(yape_radius, patch_size, number_of_octaves);

  image_generator->load_transformation_range(f);
  
  f >> mean_recognition_rate;
  cout << "> [planar_pattern_detector::load] Recognition rate: " << mean_recognition_rate << endl;

  load_managed_image_in_pakfile(f, &model_image);

  f >> number_of_model_points;
  cout << "> [planar_pattern_detector::load] " << number_of_model_points << " model points." << endl;
  model_points = new keypoint[number_of_model_points];
  for(int i = 0; i < number_of_model_points; i++) {
    f >> model_points[i].u >> model_points[i].v >> model_points[i].scale;
    model_points[i].class_index = i;
  }

  image_generator->set_original_image(model_image);
  image_generator->set_mask(u_corner[0], v_corner[0], u_corner[2], v_corner[2]);

  classifier = new fern_based_point_classifier(f);

  return true;
}

bool planar_pattern_detector::save(ofstream & f)
{
  f << image_name << endl;

  for(int i = 0; i < 4; i++) f << u_corner[i] << " " << v_corner[i] << endl;

  f << patch_size << " " << yape_radius << " " << number_of_octaves << endl;

  image_generator->save_transformation_range(f);

  f << mean_recognition_rate << endl;

  save_image_in_pakfile(f, model_image);

  f << number_of_model_points << endl;
  for(int i = 0; i < number_of_model_points; i++)
    f << model_points[i].u << " " << model_points[i].v << " " << model_points[i].scale << endl;

  classifier->save(f);

  return true;
}

//! Set the maximum number of points we want to detect
void planar_pattern_detector::set_maximum_number_of_points_to_detect(int max)
{
  maximum_number_of_points_to_detect = max;
}

//当前帧检测函数
bool planar_pattern_detector::detect(const IplImage * input_image)
{ //检测输入图像是否为灰度图像
  if (input_image->nChannels != 1 || input_image->depth != IPL_DEPTH_8U) {
    cerr << ">! [planar_pattern_detector::detect] Wrong image format" << endl;
    cerr << ">! nChannels = " << input_image->nChannels
	 << ", depth = " << input_image->depth << "." << endl;

    return false;
  } 

  pyramid->set_image(input_image); //设定高斯金字塔边界
  detect_points(pyramid); //对所有高斯金字塔生成图像进行关键点检测
  match_points(); //对特征点进行分类并计算分类分数

  pattern_is_detected = estimate_H(); //生成单应矩阵

  if (pattern_is_detected) {
    for(int i = 0; i < 4; i++)
      H.transform_point(u_corner[i], v_corner[i], detected_u_corner[i], detected_v_corner[i]); //转换检测roi的坐标系,保持原来的矩形不变

    number_of_matches = 0;
    for(int i = 0; i < number_of_model_points; i++)
      if (model_points[i].class_score > 0) {
	float Hu, Hv;
	H.transform_point(model_points[i].fr_u(), model_points[i].fr_v(), Hu, Hv);
	float dist2 =
	  (Hu - model_points[i].potential_correspondent->fr_u()) *
	  (Hu - model_points[i].potential_correspondent->fr_u()) +
	  (Hv - model_points[i].potential_correspondent->fr_v()) *
	  (Hv - model_points[i].potential_correspondent->fr_v()); //计算估计点与原始关键点的欧式距离
	if (dist2 > 10.0 * 10.0)
	  model_points[i].class_score = 0.0; //以10*10作为阈值判断是否检测检测到特征点
	else
	  number_of_matches++;
      }
  }

  return pattern_is_detected;
}
//关键点图像存储函数
void planar_pattern_detector::save_image_of_model_points(const char * filename, int patch_size)
{
  IplImage * color_model_image = mcvGrayToColor(model_image); //黑白图像转彩色图像
  //在图像上用圆圈把关键点标记出来
  for(int i = 0; i < number_of_model_points; i++)
    mcvCircle(color_model_image,
	      int( model_points[i].fr_u() + 0.5 ), int( model_points[i].fr_v() + 0.5 ), patch_size / 2 * (1 << int(model_points[i].scale)),
	      cvScalar(0, 255, 0), 2);

  mcvSaveImage(filename, color_model_image);
  cvReleaseImage(&color_model_image);
}

//图像关键点检测函数
void planar_pattern_detector::detect_points(fine_gaussian_pyramid * pyramid)
{
  manage_buffer(detected_points, maximum_number_of_points_to_detect, keypoint); //初始化关键点存储区
  //   point_detector->set_laplacian_threshold(10);
  //   point_detector->set_min_eigenvalue_threshold(10);
  number_of_detected_points = point_detector->detect(pyramid, detected_points, maximum_number_of_points_to_detect); //检测图像关键点
}

//特征点分类函数
void planar_pattern_detector::match_points(void)
{
  for(int i = 0; i < number_of_model_points; i++) {
    model_points[i].potential_correspondent = 0;
    model_points[i].class_score = 0;
  }

  for(int i = 0; i < number_of_detected_points; i++) {
    keypoint * k = detected_points + i;

    classifier->recognize(pyramid, k);

    if (k->class_index >= 0) {
      float true_score = exp(k->class_score);

      if (model_points[k->class_index].class_score < true_score) {
	model_points[k->class_index].potential_correspondent = k;
	model_points[k->class_index].class_score = true_score;
      }
    }
  }
}

bool planar_pattern_detector::estimate_H(void)
{
  H_estimator->reset_correspondences(number_of_model_points);

  for(int i = 0; i < number_of_model_points; i++)
    if (model_points[i].class_score > 0)
      H_estimator->add_correspondence(model_points[i].fr_u(), model_points[i].fr_v(),
				      model_points[i].potential_correspondent->fr_u(), model_points[i].potential_correspondent->fr_v(),
                                      model_points[i].class_score);

  return H_estimator->ransac(&H, 10., 1500, 0.99, true) > 10; //随机抽样一致性
}

//! test()
void planar_pattern_detector::test(int number_of_samples_for_test, bool verbose)
{
  float rate = classifier->test(model_points, number_of_model_points,
				number_of_octaves, yape_radius, number_of_samples_for_test,
				image_generator,
				verbose);

  cout << " [planar_pattern_detector::test] Rate: " << rate << endl;
}

// For debug purposes:

IplImage * planar_pattern_detector::create_image_of_matches(void)
{
  int width = MAX(model_image->width, pyramid->original_image->width);
  int height = model_image->height + pyramid->original_image->height;
  IplImage * result = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);

  mcvPut(result, model_image, 0, 0);
  mcvPut(result, pyramid->original_image, 0, model_image->height);

  int h = model_image->height;

  for(int i = 0; i < number_of_model_points; i++)
    mcvCross(result, model_points[i].fr_u(), model_points[i].fr_v(), 5, cvScalar(255, 0, 0), 1);
  cout << number_of_detected_points << " detected points." << endl;
  for(int i = 0; i < number_of_detected_points; i++)
    mcvCross(result, detected_points[i].fr_u(), h + detected_points[i].fr_v(), 5, cvScalar(255, 0, 0), 1);

  if (!pattern_is_detected) return result;
  
  CvScalar color = cvScalar(0,255,0);
  cvLine(result,
	 cvPoint(detected_u_corner[0], h + detected_v_corner[0]),
	 cvPoint(detected_u_corner[1], h + detected_v_corner[1]),
	 color, 3);

  cvLine(result,
	 cvPoint(detected_u_corner[1], h + detected_v_corner[1]),
	 cvPoint(detected_u_corner[2], h + detected_v_corner[2]),
	 color, 3);

  cvLine(result,
	 cvPoint(detected_u_corner[2], h + detected_v_corner[2]),
	 cvPoint(detected_u_corner[3], h + detected_v_corner[3]),
	 color, 3);

  cvLine(result,
	 cvPoint(detected_u_corner[3], h + detected_v_corner[3]),
	 cvPoint(detected_u_corner[0], h + detected_v_corner[0]),
	 color, 3);

  number_of_matches = 0;
  for(int i = 0; i < number_of_model_points; i++)
    if (model_points[i].class_score > 0) {
      number_of_matches++;
      cvLine(result,
	     cvPoint(model_points[i].fr_u(), model_points[i].fr_v()),
	     cvPoint(model_points[i].potential_correspondent->fr_u(),
		     h + model_points[i].potential_correspondent->fr_v()),
	     color, 1);
      cvCircle(result,
	       cvPoint(model_points[i].potential_correspondent->fr_u(),
		       h + model_points[i].potential_correspondent->fr_v()),
	       16 * (1 << int(model_points[i].potential_correspondent->scale)),
	       color, 1);
    }

  cout << "Number of matches: " << number_of_matches << endl;
  return result;
}

