#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "mtcnn/detector.h"
#include "draw.hpp"

//// rm -rf build; mkdir build;cd build;cmake ..;make;cd ..
//// ./build/infer_photo ./models ./data/got.jpg

int main(int argc, char **argv) {

  if (argc < 3) {
        std::cerr << "Usage " << ": "
            << "<app_binary> "
            << "<path_to_models_dir>"
            << "<path_to_test_image>\n";
        return 1;
    return -1;
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
  cv::Mat img = cv::imread(argv[2]);

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
