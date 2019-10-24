//
//  MIT License
//  
//  Copyright (c) 2019 Miguel Angel Moreno
//  Based on original code by Tom Joy
//  
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//  

#include <iostream>
#include <vector>
#include <tuple>
#include <numeric>
#include <cmath>
#include <limits>


using namespace std;


class LinearRegression {

public:

	LinearRegression() {}

	~LinearRegression() {}

	LinearRegression(vector<double>& m_x_vals_, vector<double> m_y_vals_)
		    : m_x_vals(m_x_vals_)
		    , m_y_vals(m_y_vals_)
		    , m_num_elems(m_y_vals_.size())
		    , m_old_err(std::numeric_limits<double>::max()) {}


	void trianAlgorithm(int num_iters, double a_init, double b_init) {

		int iter = 0;
		m_a = a_init;
		m_b = b_init;

		while (!isConverged(m_a, m_b) && iter < num_iters) {
			double step = 0.02;
			cout << "step " << step << "\r\n";
			double a_grad = 0;
			double b_grad = 0;

			for (int i = 0; i < m_x_vals.size(); i++) {
				a_grad += m_x_vals[i] * (((m_a * m_x_vals[i] + m_b)) - m_y_vals[i]);
			}
			a_grad = (2 * a_grad) / m_num_elems;

			for (int i = 0; i < m_x_vals.size(); i++) {
				b_grad += (((m_a * m_x_vals[i] + m_b)) - m_y_vals[i]);
			}
			b_grad = (2 * b_grad) / m_num_elems;

			//take steps
			m_a = m_a - (a_grad * step);
			m_b = m_b - (b_grad * step);
			std::cout << "a:\t" << m_a << ", b:\t" << m_b << "\r\n";
			std::cout << "a_g:\t" << a_grad << ", b_g:\t" << b_grad << "\r\n";

			iter++;
		}

	}


	double regress(double x_) const {
		double res = m_a * x_ + m_b;
		return res;
	}


private:

	bool isConverged(double a, double b) {
		double error = 0;
		double thresh = 0.001;
		for (int i = 0; i < m_x_vals.size(); i++) {
			error += ((a * m_x_vals[i]) + b - m_y_vals[i]) * ((a * m_x_vals[i] + b) - m_y_vals[i]);
		}
		error /= m_num_elems;
		std::cout << "Error = " << error << "\r\n";
		bool res = (abs(error) > m_old_err - thresh && abs(error) < m_old_err + thresh) ? true : false;
		cout << "Is converged:" << res << "\r\n";
		m_old_err = abs(error);
		return res;
	}


	vector<double> m_x_vals;
	vector<double> m_y_vals;
	int m_num_elems;
	double m_a;
	double m_b;
	double m_old_err;

};


int main(int argc, char** argv) {

	vector<double> y({ 2.8, 2.9, 7.6, 9.0, 8.6 });
	vector<double> x({   1,   2,   3,   4,   5 });


	LinearRegression lr(x, y);

	lr.trianAlgorithm(10000, 3, -10);


	cout << lr.regress(3) << endl;


	return 0;
}

