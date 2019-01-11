#include "ETF.h"
# define M_PI 3.14159265358979323846

using namespace cv;



ETF::ETF() {
	Size s(300, 300);

	Init(s);
}

ETF::ETF(Size s) {
	Init(s);
}

void ETF::Init(Size s) {
	flowField = Mat::zeros(s, CV_32FC3);
	refinedETF = Mat::zeros(s, CV_32FC3);
	gradientMag = Mat::zeros(s, CV_32FC3);
}

/**
 * Generate initial ETF 
 * by taking perpendicular vectors(counter-clockwise) from gradient map
 */
void ETF::initial_ETF(string file, Size s) {
	resizeMat(s);

	Mat src = imread(file, 1);
	Mat src_n;
	Mat grad;
	normalize(src, src_n, 0.0, 1.0, NORM_MINMAX, CV_32FC1);
	//GaussianBlur(src_n, src_n, Size(51, 51), 0, 0);

	// Generate grad_x and grad_y
	Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
	Sobel(src_n, grad_x, CV_32FC1, 1, 0, 5);
	Sobel(src_n, grad_y, CV_32FC1, 0, 1, 5);

	//Compute gradient
	magnitude(grad_x, grad_y, gradientMag);
	normalize(gradientMag, gradientMag, 0.0, 1.0, NORM_MINMAX);

	flowField = Mat::zeros(src.size(), CV_32FC3);

#pragma omp parallel for
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f u = grad_x.at<Vec3f>(i, j);
			Vec3f v = grad_y.at<Vec3f>(i, j);

			flowField.at<Vec3f>(i, j) = normalize(Vec3f(v.val[0], u.val[0], 0));
		}
	}

	rotateFlow(flowField, flowField, 90);
}

__device__ double norm_2(double a, double b) {
    return sqrt(a*a + b*b);
}

__device__ double norm_(float3 a) {
    return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__device__ float dot_product_(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float3 normalize(float3 v) {
    // v = make_float3(log(v.x), log(v.y), log(v.z));
    float norm = rsqrt(dot_product_(v, v));
    if (norm == 0) return make_float3(0, 0, 0);
    else return make_float3(v.x*norm, v.y*norm, v.z*norm);
}


__global__  void Exe_refine_ETF(float3 *outflowField, float3 *flowField, float3 *gradientMag, int numRows, int numCols, int kernel) {
	int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if ( col >= numCols || row >= numRows ){
        return;
    }

	int idx_x = row * numCols + col;

    float3 t_cur_c = flowField[idx_x];
	float3 t_new = make_float3(0, 0, 0);

	for (int r = row - kernel; r <= row + kernel; r++) {
		for (int c = col - kernel; c <= col + kernel; c++) {
			if (r < 0 || r >= numRows || c < 0 || c >= numCols) continue;
			int idx_y = r * numCols + c;
			float3 t_cur_r = flowField[idx_y];
            float phi = dot_product_(t_cur_c, t_cur_r) > 0 ? 1 : -1; // computePhi(t_cur_c, t_cur_r);
			float w_s = norm_2(col-c, row-r) < kernel ? 1 : 0; // computeWs(Point2f(col, row), Point2f(c, r), kernel);
			// computeWm(norm(gradientMag.at<Vec3f>(r, c)), norm(gradientMag.at<float>(r, c)));
            // float* fp = (float *)&gradientMag;
            // g[idx_x] = ((float *)gradientMag)[idx_y];// - norm_(gradientMag[idx_x]);
            // g3[idx_x] = gradientMag[idx_y];// - norm_(gradientMag[idx_x]);
            float w_m = (1 + tanh(norm_(gradientMag[idx_y]) - norm_(gradientMag[idx_x]))) / 2; 
			float w_d = abs(dot_product_(t_cur_c, t_cur_r)); // computeWd(t_cur_c, t_cur_r);
            float f_p = phi * w_s * w_m * w_d;
            // float f_p = phi * w_s * w_d;
			t_new = make_float3(t_new.x + t_cur_r.x * f_p, t_new.y + t_cur_r.y * f_p, t_new.z + t_cur_r.z * f_p);
		}
	}

    /*******************************************************************************
    *        here to do normalize but fail with norm_term which is too smail       *
    *******************************************************************************/
    // outflowField[idx_x] = normalize(t_new);
    float norm_term = norm_(t_new);
    // if (norm_term  == 0)  outflowField[idx_x] = make_float3(0, 0, 0);// t_new;
    if (norm_term < 0.0001 && norm_term > -0.0001)  outflowField[idx_x] = make_float3(0, 0, 0);
    else outflowField[idx_x] =  make_float3(t_new.x/norm_term, t_new.y/norm_term, t_new.z/norm_term);
    
}


void ETF::refine_ETF(int kernel) {
// #pragma omp parallel for
	// for (int r = 0; r < flowField.rows; r++) {
	// 	for (int c = 0; c < flowField.cols; c++) {
	// 		// computeNewVector(c, r, kernel);
			
	// 	}
    // }
    int rowNumber = flowField.rows;
    int colNumber = flowField.cols;
    dim3 threadPerBlock(16, 16);
    dim3 numBlock((colNumber+threadPerBlock.x-1)/threadPerBlock.x, (rowNumber+threadPerBlock.y-1)/threadPerBlock.y);
    // dim3 block(8, 8);
    // dim3 grid( (colNumber+block.x-1)/block.x, (rowNumber+block.y-1)/block.y);
    float3* d_flowField;
    float3* d_gradientMag;
    float3* d_out_flowField;
    cudaMalloc(&d_flowField, sizeof(float3) * rowNumber * colNumber);
    cudaMalloc(&d_gradientMag, sizeof(float3) * rowNumber * colNumber);
    cudaMalloc(&d_out_flowField, sizeof(float3) * rowNumber * colNumber);
    
	// float *d_phi;
	// float *d_w_s;
	// float *d_w_m;
	// float *d_w_d; 
	// float3 *d_t_cur_c; 
	// float3 *d_t_cur_r; 
	// float3 *d_t_new; 
	// float3 *d_g3; 
	// float *d_g; 

	// float phi[rowNumber * colNumber];
	// float w_s[rowNumber * colNumber];
	// float w_m[rowNumber * colNumber];
	// float w_d[rowNumber * colNumber];
	// float3 t_cur_c[rowNumber * colNumber];
	// float3 t_cur_r[rowNumber * colNumber];
	// float3 t_new[rowNumber * colNumber];
	// float3 g3[rowNumber * colNumber];
	// float g[rowNumber * colNumber];

	// cudaMalloc(&d_phi, sizeof(float)* rowNumber * colNumber);
	// cudaMalloc(&d_w_s, sizeof(float)* rowNumber * colNumber);
	// cudaMalloc(&d_w_m, sizeof(float)* rowNumber * colNumber);
	// cudaMalloc(&d_w_d, sizeof(float)* rowNumber * colNumber);
	// cudaMalloc(&d_t_cur_c, sizeof(float3)* rowNumber * colNumber);
	// cudaMalloc(&d_t_cur_r, sizeof(float3)* rowNumber * colNumber);
	// cudaMalloc(&d_t_new, sizeof(float3)* rowNumber * colNumber);
	// cudaMalloc(&d_g3, sizeof(float3)* rowNumber * colNumber);
	// cudaMalloc(&d_g, sizeof(float)* rowNumber * colNumber);
        

	cudaMemcpy(d_flowField, flowField.data, sizeof(float3) * rowNumber * colNumber, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradientMag, gradientMag.data, sizeof(float3) * rowNumber * colNumber, cudaMemcpyHostToDevice);
	Exe_refine_ETF<<<numBlock, threadPerBlock>>>(d_out_flowField, d_flowField, d_gradientMag, rowNumber, colNumber, kernel);

    cudaMemcpy(refinedETF.data, d_out_flowField, sizeof(float3) * rowNumber * colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(phi, d_phi, sizeof(float)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(w_s, d_w_s, sizeof(float)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(w_m, d_w_m, sizeof(float)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(w_d, d_w_d, sizeof(float)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(t_cur_c, d_t_cur_c, sizeof(float3)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(t_cur_r, d_t_cur_r, sizeof(float3)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(t_new, d_t_new, sizeof(float3)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(g3, d_g3, sizeof(float3)* rowNumber *colNumber, cudaMemcpyDeviceToHost);
	// cudaMemcpy(g, d_g, sizeof(float)* rowNumber *colNumber, cudaMemcpyDeviceToHost);

	cudaFree(d_flowField);
	cudaFree(d_gradientMag);	
    cudaFree(d_out_flowField);  
	// cudaFree(d_phi);
	// cudaFree(d_w_s);
	// cudaFree(d_w_m);
	// cudaFree(d_w_d);
	// cudaFree(d_t_cur_c);
	// cudaFree(d_t_cur_r);
	// cudaFree(d_t_new);
	// cudaFree(d_g3);
	// cudaFree(d_g);
    /*
	for(int i = 0; i < rowNumber *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
    */
	flowField = refinedETF.clone();

    //cout << colNumber << ", " << rowNumber << endl;
}

/*
 * Paper's Eq(1)
er *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
	flowField = refinedETF.clone();
er *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
	flowField = refinedETF.clone();y, const int kernel) {
er *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
	flowField = refinedETF.clone();<Vec3f>(y, x);
er *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
	flowField = refinedETF.clone();
er *colNumber; i++){
		// cout << "["<< t_new[i].x  << ", "<< t_new[i].y  << ", "<< t_new[i].z  << "], ";

		cout << refinedETF.at<Vec3f>(i/colNumber, i%colNumber) << ", ";
        // printf("\t % .3f", w_m[i]);
		if(i % 10 == 9) cout << "\n";
	}  
	flowField = refinedETF.clone();
	for (int r = y - kernel; r <= y + kernel; r++) {
		for (int c = x - kernel; c <= x + kernel; c++) {
			if (r < 0 || r >= refinedETF.rows || c < 0 || c >= refinedETF.cols) continue;

			const Vec3f t_cur_y = flowField.at<Vec3f>(r, c);
			float phi = t_cur_x.dot(t_cur_y) > 0 ? 1 : -1; // computePhi(t_cur_x, t_cur_y);
			float w_s = norm(t_cur_x - t_cur_y) < kernel ? 1 : 0; // computeWs(Point2f(x, y), Point2f(c, r), kernel);
			// computeWm(norm(gradientMag.at<Vec3f>(y, x)), norm(gradientMag.at<float>(r, c)));
			float w_m = (1 + tanh(norm(gradientMag.at<float>(r, c)) - norm(gradientMag.at<Vec3f>(y, x)))) / 2; 
			float w_d = abs(t_cur_x.dot(t_cur_y)); // computeWd(t_cur_x, t_cur_y);
			t_new += phi * t_cur_y * w_s * w_m * w_d;
		}
	}
	refinedETF.at<Vec3f>(y, x) = normalize(t_new);
}

/*
 * Paper's Eq(5)
 */
float ETF::computePhi(cv::Vec3f x, cv::Vec3f y) {
	return x.dot(y) > 0 ? 1 : -1;
}

/*
 * Paper's Eq(2)
 */
float ETF::computeWs(cv::Point2f x, cv::Point2f y, int r) {
	return norm(x - y) < r ? 1 : 0;
}

/*
 * Paper's Eq(3)
 */
float ETF::computeWm(float gradmag_x, float gradmag_y) {
	float wm = (1 + tanh(gradmag_y - gradmag_x)) / 2;
	return wm;
}

/*
 * Paper's Eq(4)
 */
float ETF::computeWd(cv::Vec3f x, cv::Vec3f y) {
	return abs(x.dot(y));
}

void ETF::rotateFlow(Mat& src, Mat& dst, float theta) {
	theta = theta / 180.0 * M_PI;

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3f v = src.at<cv::Vec3f>(i, j);
			float rx = v[0] * cos(theta) - v[1] * sin(theta);
			float ry = v[1] * cos(theta) + v[0] * sin(theta);
			dst.at<cv::Vec3f>(i, j) = Vec3f(rx, ry, 0.0);
		}
	}

}

void ETF::resizeMat(Size s) {
	resize(flowField, flowField, s, 0, 0, CV_INTER_LINEAR);
	resize(refinedETF, refinedETF, s, 0, 0, CV_INTER_LINEAR);
	resize(gradientMag, gradientMag, s, 0, 0, CV_INTER_LINEAR);
}


