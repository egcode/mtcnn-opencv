#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mtcnn/detector.h"

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/infer_photo ./data/got.jpg

using rectPoints = std::pair<cv::Rect, std::vector<cv::Point>>;

static cv::Mat drawRectsAndPoints(const cv::Mat &img,
                                  const std::vector<rectPoints> data) {
  cv::Mat outImg;
  img.convertTo(outImg, CV_8UC3);

  for (auto &d : data) {
    cv::rectangle(outImg, d.first, cv::Scalar(0, 255, 255), 2);
    auto pts = d.second;
    for (size_t i = 0; i < pts.size(); ++i) {
      cv::circle(outImg, pts[i], 3, cv::Scalar(0, 255, 255), 1);
    }
  }
  return outImg;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Usage " << argv[0] << ": "
              << " "
              << "<test-image>\n";
    return -1;
  }

  ProposalNetwork::Config pConfig;
  pConfig.caffeModel = "./data/models/det1.caffemodel";
  pConfig.protoText = "./data/models/det1.prototxt";
  pConfig.threshold = 0.6f;

  RefineNetwork::Config rConfig;
  rConfig.caffeModel = "./data/models/det2.caffemodel";
  rConfig.protoText = "./data/models/det2.prototxt";
  rConfig.threshold = 0.7f;

  OutputNetwork::Config oConfig;
  oConfig.caffeModel = "./data/models/det3.caffemodel";
  oConfig.protoText = "./data/models/det3.prototxt";
  oConfig.threshold = 0.7f;

  MTCNNDetector detector(pConfig, rConfig, oConfig);
  cv::Mat img = cv::imread(argv[1]);

  std::vector<Face> faces;

  {
    faces = detector.detect(img, 20.f, 0.709f);
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

  auto resultImg = drawRectsAndPoints(img, data);
  cv::imshow("test-oc", resultImg);
  cv::waitKey(0);

  return 0;
}
