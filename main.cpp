#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include <string>
#include <iostream>
using namespace std;
using namespace cv;
void R(Mat &src,Mat &dst) {
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;
    Mat tmp;
    tmp.create(src.size(), src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst = tmp;
}
void FoFFT(Mat &Src, Mat *Img) {
    int M = getOptimalDFTSize( Src.rows );
    int N = getOptimalDFTSize( Src.cols );
    Mat padded;
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    split(complexImg, planes);
    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));
    R(planes[0],planes[0]);
    R(planes[1],planes[1]);
    planes[0]/=float(M*N);
    planes[1]/=float(M*N);
    Img[0]=planes[0].clone();
    Img[1]=planes[1].clone();
}
void InFFT(Mat *Img,Mat &Dst) {
    R(Img[0],Img[0]);
    R(Img[1],Img[1]);
    Mat complexImg;
    merge(Img, 2, complexImg);
    idft(complexImg, complexImg);
    split(complexImg,Img);
    normalize(Img[0], Dst, 0, 1, CV_MINMAX);
}
void FoFFT_Phase(Mat &src, Mat &Mag,Mat &Phase) {
    Mat planes[2];
    FoFFT(src,planes);
    Mag.zeros(planes[0].rows,planes[0].cols,CV_32F);
    cv::cartToPolar(planes[0],planes[1],Mag,Phase);
}
void InFFT_Phase(Mat &Mag,Mat &Phase, Mat &src) {
    Mat planes[2];
    planes[0].create(Mag.rows,Mag.cols,CV_32F);
    planes[1].create(Mag.rows,Mag.cols,CV_32F);
    cv::polarToCart(Mag,Phase,planes[0],planes[1]);
    InFFT(planes,src);
}
int main(int argc, char ** argv) {
    Mat img,Mag, Phase;
    img=imread("/home/azamat/a/len.png",0);
    imshow("Original",img);
    FoFFT_Phase(img,Mag,Phase);
    int R=300;
    int r=30;
    Mat mask;
    mask.create(Mag.cols,Mag.rows,CV_32F);
    int cx = Mag.cols>>1;
    int cy = Mag.rows>>1;
    mask=1;
    cv::circle(mask,cv::Point(cx,cy),R,CV_RGB(0,0,0),-1);
    cv::circle(mask,cv::Point(cx,cy),r,CV_RGB(1,1,1),-1);
    cv::multiply(Mag,mask,Mag);
    cv::multiply(Phase,mask,Phase);
    InFFT_Phase(Mag,Phase,img);
    imshow("Phaze", Phase);
    imshow("Результат фильтрации", img);
    cvWaitKey(0);
    return 0;
}
