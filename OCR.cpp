#include "OCR.hpp"

void OCR::clearTrainingVector()
{
        trainData.clear();
        targetData.clear();
}


void OCR::addTrainingData(Mat& img, char target)
{
        //Append target value
        targetData.push_back(target);
        
        //Append train data feature
        Mat c, feat;
        resize(img, c, cSize);
        
        vector<float> descriptors;
        hog.compute(c, descriptors);
        trainData.push_back(Mat(descriptors).t());
}

void OCR::train()
{
        //Organize training data into single matrix
        Mat tDataMat(Size(0, trainData.size()), CV_32F);
        for (Mat& fv : trainData)
                tDataMat.push_back(fv);
                
        //Organize target data
        Mat targetMat(targetData);
        targetMat.convertTo(targetMat, CV_32F);
        
        //Train SVM
        cout << "Started training" << endl;
        
        svm.train(tDataMat, targetMat, Mat(), Mat(), params);
        svm.save(fPath.c_str());
        
        cout << "Finished training" << endl;
}

char OCR::recognize(Mat& img)
{
        //Resize char img
        Mat chr;
        resize(img, chr, cSize);
        
        //Get HOG feature vector
        vector<float> descriptors;
        hog.compute(chr, descriptors);
        Mat desc = Mat(descriptors).t();
        
        //Get SVM response
        int rsp = cvRound(svm.predict(desc));
        return char(rsp);
}

int main()
{
        cout << "hi" << endl;
        return 0;
}
