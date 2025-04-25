#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <filesystem>
#include <omp.h>
#include <immintrin.h>
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Gaussian kernel generation
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

// Apply Gaussian blur using AVX + OpenMP
Mat applyGaussianBlurAVX(const Mat& input) {
    int kernelSize = 5;
    float sigma = 1.0;
    vector<vector<float>> kernel = generateGaussianKernel(kernelSize, sigma);
    int half = kernelSize / 2;

    Mat padded;
    copyMakeBorder(input, padded, half, half, half, half, BORDER_REFLECT);

    Mat output = Mat::zeros(input.size(), input.type());

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            __m256 sum_b = _mm256_setzero_ps();
            __m256 sum_g = _mm256_setzero_ps();
            __m256 sum_r = _mm256_setzero_ps();

            for (int ky = -half; ky <= half; ++ky) {
                for (int kx = -half; kx <= half; ++kx) {
                    Vec3b pixel = padded.at<Vec3b>(y + ky + half, x + kx + half);
                    float weight = kernel[ky + half][kx + half];
                    __m256 w = _mm256_set1_ps(weight);

                    __m256 b = _mm256_set1_ps(pixel[0]);
                    __m256 g = _mm256_set1_ps(pixel[1]);
                    __m256 r = _mm256_set1_ps(pixel[2]);

                    sum_b = _mm256_add_ps(sum_b, _mm256_mul_ps(w, b));
                    sum_g = _mm256_add_ps(sum_g, _mm256_mul_ps(w, g));
                    sum_r = _mm256_add_ps(sum_r, _mm256_mul_ps(w, r));
                }
            }

            // Extract values from AVX register (only first lane needed)
            float blur_b = ((float*)&sum_b)[0];
            float blur_g = ((float*)&sum_g)[0];
            float blur_r = ((float*)&sum_r)[0];

            output.at<Vec3b>(y, x) = Vec3b((uchar)blur_b, (uchar)blur_g, (uchar)blur_r);
        }
    }

    return output;
}

int main() {
    string inputFolder = "Images/";  // Path to input images
    string outputFolder = "blurred_images_avx_openmp/"; // Path to save processed images
    vector<int> datasetSizes = {125, 250, 500, 1000, 2000, 4000, 5000};  // Dataset sizes to process
    ofstream logFile("parallel_times.csv");  // Log file for execution times
    logFile << "NumImages,TotalTime\n";

    if (!fs::exists(outputFolder)) fs::create_directory(outputFolder);

    for (int numImages : datasetSizes) {
        double totalTime = 0.0;
        cout << "\nRunning for " << numImages << " images...\n";

        auto start = chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+:totalTime)
        for (int i = 0; i < numImages; ++i) {
            char filename[20];
            sprintf(filename, "%06d.jpg", i);
            string filePath = inputFolder + filename;

            auto imgStart = chrono::high_resolution_clock::now();

            Mat image = imread(filePath);
            if (image.empty()) {
                cerr << "Failed to load: " << filePath << endl;
                continue;
            }

            Mat blurred = applyGaussianBlurAVX(image);

            if (numImages == 1000) {  // Optional: Save only for a specific dataset size (1000 images)
                string outPath = outputFolder + "Blur_Image" + to_string(i + 1) + ".jpg";
                imwrite(outPath, blurred);
            }

            auto imgEnd = chrono::high_resolution_clock::now();
            double imgTime = chrono::duration<double>(imgEnd - imgStart).count();

            totalTime += imgTime;
        }

        auto end = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(end - start).count();

        logFile << numImages << "," << duration << "\n";

        cout << "Total Time: " << duration << " seconds\n";
        cout << "Throughput: " << (numImages / duration) << " images/second\n";
    }

    logFile.close();
    cout << "\nTiming data saved to parallel_times.csv\n";
    return 0;
}
