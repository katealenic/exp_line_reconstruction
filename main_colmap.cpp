/* 
 * Line3D++ - Line-based Multi View Stereo
 * Copyright (C) 2015  Manuel Hofer

 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

// check libs
#include "configLIBS.h"

// EXTERNAL
#include <tclap/CmdLine.h>
#include <tclap/CmdLineInterface.h>
#include <boost/filesystem.hpp>
#include "eigen3/Eigen/Eigen"

// std
#include <iostream>
#include <fstream>
#include <glob.h>
#include <cmath>                                         // для функции cos

#define PI 3.14159265

// opencv
#ifdef L3DPP_OPENCV3
#include <opencv2/highgui.hpp>
#else
#include <opencv/highgui.h>
#endif //L3DPP_OPENCV3

// lib
#include "line3D.h"

// INFO:
// This executable reads colmap results (cameras.txt, images.txt, and points3D.txt) and executes the Line3D++ algorithm.
// If distortion coefficients are stored in the cameras.txt file, you need to use the _original_ (distorted) images!

int main(int argc, char *argv[])
{
    TCLAP::CmdLine cmd("LINE3D++");

    TCLAP::ValueArg<std::string> inputArg("i", "input_folder", "folder containing the images", true, "", "string");
    cmd.add(inputArg);

    TCLAP::ValueArg<std::string> inputLines("f", "input_lines", "folder containing the lines", false, "lines", "string");
    cmd.add(inputLines);

    TCLAP::ValueArg<std::string> inputMatches("b", "input_matches", "folder containing the matches", false, "matches", "string");
    cmd.add(inputMatches);

    TCLAP::ValueArg<bool> matchepip("u", "epip_match", "use epipolar matching", false, true, "bool");
    cmd.add(matchepip);

    TCLAP::ValueArg<std::string> inputType("t", "type", "dataset type", false, "outdoor", "string");
    cmd.add(inputType);

    TCLAP::ValueArg<std::string> outputArg("o", "output_folder", "folder where result and temporary files are stored (if not specified --> sfm_folder+'/Line3D++/')", false, "", "string");
    cmd.add(outputArg);

    TCLAP::ValueArg<int> scaleArg("w", "max_image_width", "scale image down to fixed max width for line segment detection", false, L3D_DEF_MAX_IMG_WIDTH, "int");
    cmd.add(scaleArg);

    TCLAP::ValueArg<int> neighborArg("n", "num_matching_neighbors", "number of neighbors for matching", false, L3D_DEF_MATCHING_NEIGHBORS, "int");
    cmd.add(neighborArg);

    TCLAP::ValueArg<float> sigma_A_Arg("a", "sigma_a", "angle regularizer", false, L3D_DEF_SCORING_ANG_REGULARIZER, "float");
    cmd.add(sigma_A_Arg);

    TCLAP::ValueArg<float> sigma_P_Arg("p", "sigma_p", "position regularizer (if negative: fixed sigma_p in world-coordinates)", false, L3D_DEF_SCORING_POS_REGULARIZER, "float");
    cmd.add(sigma_P_Arg);

    TCLAP::ValueArg<float> epipolarArg("e", "min_epipolar_overlap", "minimum epipolar overlap for matching", false, L3D_DEF_EPIPOLAR_OVERLAP, "float");
    cmd.add(epipolarArg);

    TCLAP::ValueArg<int> knnArg("k", "knn_matches", "number of matches to be kept (<= 0 --> use all that fulfill overlap)", false, L3D_DEF_KNN, "int");
    cmd.add(knnArg);

    TCLAP::ValueArg<int> segNumArg("y", "num_segments_per_image", "maximum number of 2D segments per image (longest)", false, L3D_DEF_MAX_NUM_SEGMENTS, "int");
    cmd.add(segNumArg);

    TCLAP::ValueArg<int> visibilityArg("v", "visibility_t", "minimum number of cameras to see a valid 3D line", false, L3D_DEF_MIN_VISIBILITY_T, "int");
    cmd.add(visibilityArg);

    TCLAP::ValueArg<bool> diffusionArg("d", "diffusion", "perform Replicator Dynamics Diffusion before clustering", false, L3D_DEF_PERFORM_RDD, "bool");
    cmd.add(diffusionArg);

    TCLAP::ValueArg<bool> loadArg("l", "load_and_store_flag", "load/store segments (recommended for big images)", false, L3D_DEF_LOAD_AND_STORE_SEGMENTS, "bool");
    cmd.add(loadArg);

    TCLAP::ValueArg<float> collinArg("r", "collinearity_t", "threshold for collinearity", false, L3D_DEF_COLLINEARITY_T, "float");
    cmd.add(collinArg);

    TCLAP::ValueArg<bool> cudaArg("g", "use_cuda", "use the GPU (CUDA)", false, true, "bool");
    cmd.add(cudaArg);

    TCLAP::ValueArg<bool> ceresArg("c", "use_ceres", "use CERES (for 3D line optimization)", false, L3D_DEF_USE_CERES, "bool");
    cmd.add(ceresArg);

    TCLAP::ValueArg<float> constRegDepthArg("z", "const_reg_depth", "use a constant regularization depth (only when sigma_p is metric!)", false, -1.0f, "float");
    cmd.add(constRegDepthArg);

    // read arguments
    cmd.parse(argc,argv);
    std::string inputFolder = inputArg.getValue().c_str();
    std::string inputFolderLines = inputLines.getValue().c_str();
    std::string inputFolderMatches = inputMatches.getValue().c_str();

    /*
    boost::filesystem::path linef(inputFolder+inputFolderLines);
    if(!boost::filesystem::exists(linef))
    {
        std::cerr << "lines result folder " << inputFolder+inputFolderLines << " does not exist!" << std::endl;
        return -1;
    }
    */
    boost::filesystem::path idsf(inputFolder+"arkit_data.txt");
    if(!boost::filesystem::exists(idsf))
    {
        std::cerr << "ids folder " << inputFolder+"arkit_data.txt" << " does not exist!" << std::endl;
        return -1;
    }

    boost::filesystem::path cmsf(inputFolder+"K.txt");
    if(!boost::filesystem::exists(cmsf))
    {
        std::cerr << "cameras.txt " << inputFolder+"K.txt" << " does not exist!" << std::endl;
        return -1;
    }


    std::string outputFolder = outputArg.getValue().c_str();
    //boost::filesystem::path sfm(sfmFolder);


    //boost::filesystem::path dir1(inputFolder+"/Sfm/");
    //if(!boost::filesystem::exists(dir1)) boost::filesystem::create_directory(dir1);
    //boost::filesystem::path dir2(inputFolder+"/Sfm/results/");
    //if(!boost::filesystem::exists(dir2)) boost::filesystem::create_directory(dir2);


    //if(outputFolder.length() == 0)
    //    outputFolder = inputFolder+"/Sfm/results/";
    //else outputFolder = inputFolder+"/Sfm/results/"+outputFolder;

    int maxWidth = scaleArg.getValue();
    unsigned int neighbors = std::max(neighborArg.getValue(),2);
    bool diffusion = diffusionArg.getValue();
    bool loadAndStore = loadArg.getValue();
    float collinearity = collinArg.getValue();
    bool useGPU = cudaArg.getValue();
    bool useCERES = ceresArg.getValue();
    float epipolarOverlap = fmin(fabs(epipolarArg.getValue()),0.99f);
    float sigmaA = fabs(sigma_A_Arg.getValue());
    float sigmaP = sigma_P_Arg.getValue();
    int kNN = knnArg.getValue();
    unsigned int maxNumSegments = segNumArg.getValue();
    unsigned int visibility_t = visibilityArg.getValue();
    float constRegDepth = constRegDepthArg.getValue();
    bool matchepip_ = matchepip.getValue();
    std::string inpType = inputType.getValue();


    // create output directory
    boost::filesystem::path dir(outputFolder);
    boost::filesystem::create_directory(dir);

    // create Line3D++ object
    L3DPP::Line3D* Line3D = new L3DPP::Line3D(outputFolder,inputFolder,inputFolderMatches,loadAndStore,maxWidth,
                                              maxNumSegments,true,useGPU,matchepip_);

    // check if result files exist

    std::cout << std::endl << "reading arkit data..." << std::endl;

    std::ifstream file_arkit(inputFolder+"/arkit_data.txt");
    std::string str_data;
    std::vector<std::vector<std::string>> matrix_arkit_data;
    int ii =0;
    while (std::getline(file_arkit, str_data))
        {
            std::string value;
            std::stringstream ss(str_data);
            matrix_arkit_data.push_back(std::vector<std::string>());
            while (ss >> value)
            {
                matrix_arkit_data[ii].push_back(value);
                //std::cout <<value<< std::endl;
            }
            ++ii;
        }

    std::cout << std::endl << "reading K..." << std::endl;
    std::ifstream file_k(inputFolder+"/K.txt");
    std::string str_k;
    std::vector<std::vector<float>> matrix_k;
    while (std::getline(file_k, str_k))
    {
            float value;
            std::stringstream ss(str_k);
            matrix_k.push_back(std::vector<float>());
            while (ss >> value)
            {
                matrix_k[0].push_back(value);
                //std::cout <<value<< std::endl;
            }
    }

    Eigen::Matrix3d K;
    K(0,0) = matrix_k[0][0]; K(0,1) = 0;  K(0,2) = matrix_k[0][2];
    K(1,0) = 0;  K(1,1) = matrix_k[0][1]; K(1,2) = matrix_k[0][3];
    K(2,0) = 0;  K(2,1) = 0;  K(2,2) = 1;

    //Eigen::Matrix3d tensor;
    //tensor(0,0) = 1;  tensor(0,1) = 0;  tensor(0,2) = 0;
    //tensor(1,0) = 0;  tensor(1,1) = -1;  tensor(1,2) = 0;
    ///tensor(2,0) = 0;  tensor(2,1) = 0;  tensor(2,2) = -1;

    std::cout << std::endl << "reading pair..." << std::endl;
    std::string sf6=inputFolder+"pair.txt";
        std::ifstream file6(sf6);
        std::string str6;
        std::vector<std::vector<int>> matrix_pair;
        ii =0;
        while (std::getline(file6, str6))
        {
            float value;
            std::stringstream ss(str6);
            matrix_pair.push_back(std::vector<int>());
            while (ss >> value)
            {
                matrix_pair[ii].push_back(value);
                //std::cout <<value<< std::endl;
            }
            ++ii;
        }

    std::cout << std::endl << "reading.." << std::endl;
    for(int id=0; id<matrix_arkit_data.size(); ++id)
    {
        int imgID= std::stoi(matrix_arkit_data[id][0]);
        std::string sname = matrix_arkit_data[id][1];

        double qw,qx,qy,qz,tx,ty,tz;

        qw = std::stod(matrix_arkit_data[id][2]);
        qz = (-1)*std::stod(matrix_arkit_data[id][3]);
        qy = std::stod(matrix_arkit_data[id][4]);
        qx = std::stod(matrix_arkit_data[id][5]);

        tx = std::stod(matrix_arkit_data[id][6]);
        ty = std::stod(matrix_arkit_data[id][7]);
        tz = std::stod(matrix_arkit_data[id][8]);

        Eigen::Matrix3d R = Line3D->rotationFromQ(qw,qx,qy,qz);

        Eigen::Vector3d t(tx,ty,tz);

        t = (-1*R) *t;

        Eigen::Vector3d C = (R.transpose()) * (-1.0 * t);


        std::string::size_type const p(sname.find_last_of('.'));
        std::string sname_ls = sname.substr(0, p);


        std::string sf4=inputFolder+inputFolderLines+"/lines_"+sname_ls+".txt";
        std::cout <<sf4<< std::endl;
        std::ifstream file4(sf4);
        std::string str4;
        std::vector<std::vector<float>> matrix_lines;
        ii =0;

        while (std::getline(file4, str4))
        {
            float value;
            std::stringstream ss(str4);
            matrix_lines.push_back(std::vector<float>());
            while (ss >> value)
            {
                matrix_lines[ii].push_back(value);
                //std::cout <<value<< std::endl;
            }
            ++ii;
        }

        std::vector<cv::Vec4f> all_lines(matrix_lines.size());
        //std::cout <<"!"<< std::endl;
        for(int j=0; j<matrix_lines.size(); ++j)
        {
        all_lines[j](0)=matrix_lines[j][0];
        all_lines[j](1)=matrix_lines[j][1];
        all_lines[j](2)=matrix_lines[j][2];
        all_lines[j](3)=matrix_lines[j][3];
        }

        //std::cout <<"!!"<< std::endl;

        cv::Mat image = cv::imread(inputFolder+"/images/"+sname,CV_LOAD_IMAGE_GRAYSCALE);
        std::list<unsigned int> wps_list;

        float med_depth = 1e-12;
        //std::cout <<"0"<< std::endl;


        for(int j=0; j<matrix_pair.size(); ++j)
        {
        if (matrix_pair[j][0]==imgID){
            for (int jj=1; jj<neighbors; ++jj)  wps_list.push_back(matrix_pair[j][jj]);

        }
        }
        wps_list.sort();

         if (all_lines.size()>0) Line3D->addImage(imgID,image,K,R,t,med_depth,wps_list,all_lines);
         else Line3D->addImage(imgID,image,K,R,t,med_depth,wps_list);


    }

    // match images
    Line3D->matchImages(sigmaP,sigmaA,neighbors,epipolarOverlap,
                        kNN,constRegDepth);

    // compute result
    Line3D->reconstruct3Dlines(visibility_t,diffusion,collinearity,useCERES);

    // save end result
    std::vector<L3DPP::FinalLine3D> result;
    Line3D->get3Dlines(result);

    // save as STL
    Line3D->saveResultAsSTL(outputFolder);
    // save as OBJ
    Line3D->saveResultAsOBJ(outputFolder);
    // save as TXT
    Line3D->save3DLinesAsTXT(outputFolder);
    // save as BIN
    Line3D->save3DLinesAsBIN(outputFolder);

    // cleanup
    delete Line3D;
}
