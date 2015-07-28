#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class OCR
{        
        private:
                string fPath;
                Size cSize;
                
                vector<Mat> trainData;
                vector<char> targetData;
                
                HOGDescriptor hog;
                CvSVM svm;
                CvSVMParams params;             
                
        public:
                OCR(string fp, Size cs) : fPath(fp), cSize(cs)
                {
                        //Create HOG Descriptor
                        int win = 4;
                        int block = 4;
                        int stride = 4;
                        int cell = 2;
                        int bins = 16;
                        hog = HOGDescriptor ( Size(win,win), Size(block,block), Size(stride,stride), Size(cell,cell), bins);
                        
                        //Set SVM params
                        params.svm_type = CvSVM::C_SVC;
                        params.kernel_type = CvSVM::LINEAR;
                        params.degree = 3;
                        params.gamma = 4;
                        params.C = 3;
                        
                        //Check SVM file existence and load
                        if (bool(ifstream(fPath)))
                                svm.load(fPath.c_str());
                };
                
                void clearTrainingVector();
                void addTrainingData(Mat& img, char target);
                void train();
                char recognize(Mat& img);                                
                
};
