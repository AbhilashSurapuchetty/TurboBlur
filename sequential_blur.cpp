// sequential_blur.cpp
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

vector<vector<float>> generateGaussianKernel(int size, float sigma) {
    vector<vector<float>> kernel(size, vector<float>(size));
    float sum = 0.0;
    int half = size / 2;

    for (int i = -half; i <= half; ++i) {
        for (int j = -half; j <= half; ++j) {
            float value = exp(-(i * i + j * j) / (2 * sigma * sigma));
            value /= (2 * CV_PI * sigma * sigma);
            kernel[i + half][j + half] = value;
            sum += value;
        }
    }

    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            kernel[i][j] /= sum;

    return kernel;
}

Mat applyGaussianBlur(const Mat& input) {
    int kernelSize = 5;
    float sigma = 1.0;
    vector<vector<float>> kernel = generateGaussianKernel(kernelSize, sigma);
    int half = kernelSize / 2;

    Mat padded;
    copyMakeBorder(input, padded, half, half, half, half, BORDER_REFLECT);
    Mat output = Mat::zeros(input.size(), input.type());

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            Vec3f sum = {0, 0, 0};
            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    Vec3b pixel = padded.at<Vec3b>(y + ky + half, x + kx + half);
                    float weight = kernel[ky + half][kx + half];
                    sum[0] += weight * pixel[0];
                    sum[1] += weight * pixel[1];
                    sum[2] += weight * pixel[2];
                }
            }
            output.at<Vec3b>(y, x) = Vec3b((uchar)sum[0], (uchar)sum[1], (uchar)sum[2]);
        }
    }
    return output;
}

int main() {
    string inputFolder = "Images/";
    string outputFolder = "blurred_images_sequential/";
    vector<int> datasetSizes = {125, 250, 500, 1000, 2000, 4000, 5000};
    ofstream logFile("sequential_times.csv");
    logFile << "NumImages,TotalTime\n";

    if (!fs::exists(outputFolder)) fs::create_directory(outputFolder);

    for (int numImages : datasetSizes) {
        double totalTime = 0.0;
        cout << "\nRunning for " << numImages << " images...\n";

        auto start = chrono::high_resolution_clock::now();

        for (int i = 0; i < numImages; ++i) {
            char filename[20];
            sprintf(filename, "%06d.jpg", i);
            string filePath = inputFolder + filename;

            Mat image = imread(filePath);
            if (image.empty()) {
                cerr << "Failed to load: " << filePath << endl;
                continue;
            }

            Mat blurred = applyGaussianBlur(image);

            if (numImages == 1000) {
                string outPath = outputFolder + "Blur_Image" + to_string(i + 1) + ".jpg";
                imwrite(outPath, blurred);
            }
        }

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(end - start).count();
        logFile << numImages << "," << duration << "\n";

        cout << "Total Time: " << duration << " seconds\n";
        cout << "Throughput: " << (numImages / duration) << " images/second\n";
    }

    logFile.close();
    cout << "\nTiming data saved to sequential_times.csv\n";
    return 0;
}
