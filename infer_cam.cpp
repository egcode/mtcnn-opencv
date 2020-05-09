#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "mtcnn/detector.h"
#include "draw.hpp"

#include <iostream>
#include <string>

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/infer_cam ./models

using namespace cv;
using std::cout; using std::cerr; using std::endl;


int main(int argc, char **argv)
{
    /////////////////////////////////////////////start mtcnn

    if(argv[1] == NULL)
    {
        cerr << "ERROR: path argument should not be empty" << endl;
        std::cerr << "Usage " << ": "
            << "<app_binary> "
            << "<path_to_models_dir>\n";
        return 1;
    }
    std::string modelPath = argv[1];

    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = modelPath + "/det1.caffemodel";
    pConfig.protoText = modelPath + "/det1.prototxt";
    pConfig.threshold = 0.6f;

    RefineNetwork::Config rConfig;
    rConfig.caffeModel = modelPath + "/det2.caffemodel";
    rConfig.protoText = modelPath + "/det2.prototxt";
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = modelPath + "/det3.caffemodel";
    oConfig.protoText = modelPath + "/det3.prototxt";
    oConfig.threshold = 0.7f;

    MTCNNDetector detector(pConfig, rConfig, oConfig);

    /////////////////////////////////////////////end mtcnn
    Mat frame;
    cout << "Opening camera..." << endl;
    VideoCapture capture(0); // open the first camera
    if (!capture.isOpened())
    {
        cerr << "ERROR: Can't initialize camera capture" << endl;
        return 1;
    }
    capture.set(CAP_PROP_FRAME_WIDTH,640);
    capture.set(CAP_PROP_FRAME_HEIGHT,480);

    cout << "Frame width: " << capture.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "     height: " << capture.get(CAP_PROP_FRAME_HEIGHT) << endl;
    cout << "Capturing FPS: " << capture.get(CAP_PROP_FPS) << endl;

    cout << endl << "Press 'ESC' to quit, 'space' to toggle frame processing" << endl;
    cout << endl << "Start grabbing..." << endl;

    size_t nFrames = 0;
    bool enableProcessing = false;
    int64 t0 = cv::getTickCount();
    int64 processingTime = 0;
    for (;;)
    {
        capture >> frame; // read the next frame from camera
        if (frame.empty())
        {
            cerr << "ERROR: Can't grab camera frame." << endl;
            break;
        }
        nFrames++;
        if (nFrames % 10 == 0)
        {
            const int N = 10;
            int64 t1 = cv::getTickCount();
            cout << "Frames captured: " << cv::format("%5lld", (long long int)nFrames)
                 << "    Average FPS: " << cv::format("%9.1f", (double)getTickFrequency() * N / (t1 - t0))
                 << "    Average time per frame: " << cv::format("%9.2f ms", (double)(t1 - t0) * 1000.0f / (N * getTickFrequency()))
                 << "    Average processing time: " << cv::format("%9.2f ms", (double)(processingTime) * 1000.0f / (N * getTickFrequency()))
                 << std::endl;
            t0 = t1;
            processingTime = 0;
        }
        if (!enableProcessing)
        {
            /////////////////////////////////////////////start mtcnn
            std::vector<Face> faces;

            {
                faces = detector.detect(frame, 20.f, 0.709f);
            }

            std::cout << "Number of faces found in the supplied image - " << faces.size()
                        << std::endl;

            std::vector<rectPoints> data;

            // show the image with faces in it
            for (size_t i = 0; i < faces.size(); ++i) {
                std::vector<cv::Point> pts;
                for (int p = 0; p < NUM_PTS; ++p) {
                pts.push_back(
                    cv::Point(faces[i].ptsCoords[2 * p], faces[i].ptsCoords[2 * p + 1]));
                }

                auto rect = faces[i].bbox.getRect();
                auto d = std::make_pair(rect, pts);
                data.push_back(d);
            }

            auto resultImg = drawRectsAndPoints(frame, data);
            cv::imshow("test-oc", resultImg);
            /////////////////////////////////////////////end mtcnn
        }
        else
        {
            int64 tp0 = cv::getTickCount();
            Mat processed;
            cv::Canny(frame, processed, 400, 1000, 5);
            processingTime += cv::getTickCount() - tp0;
            imshow("Frame", processed);
        }
        int key = waitKey(1);
        if (key == 27/*ESC*/)
            break;
        if (key == 32/*SPACE*/)
        {
            enableProcessing = !enableProcessing;
            cout << "Enable frame processing ('space' key): " << enableProcessing << endl;
        }
    }
    std::cout << "Number of captured frames: " << nFrames << endl;
    return nFrames > 0 ? 0 : 1;
}
