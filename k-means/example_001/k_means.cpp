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
#include <valarray>
#include <vector>
#include <utility>
#include <cmath>


using namespace std;


class kMeans {

public:

	kMeans() {}

	~kMeans() {}

	kMeans(int k, vector<pair<double,double>> & data_)
			: m_k(k)
			, m_means(k)
			, m_data(k) {
		m_data[0] = data_; //this just assigns the first label to all data
	}


	void clusterData(valarray<pair<double,double>> & init_means_, int num_iters_ = 10) {
		//inilise data
		cout << "Initilising labels\r\n"; 
		m_means = init_means_;
		this->assignLabels();

		int i = 0;
		while(i < num_iters_ && !this->computeMeans()) {
			cout << "Running iteration << " << i << "\r\n";		
			this->assignLabels();
			i++;
		}

	}

	void printClusters() const {
		for (int k = 0; k < m_k; k++) {
			cout << "Cluster: " << k << "\r\n";
			for (auto const & feature : m_data[k]) {
				cout <<  " [" << get<0>(feature) << "," <<  get<1>(feature) << "] ";
			}
			cout << "\r\n";
		}

	}


private:

	bool computeMeans() {
		//return true if means are the same
		bool res = true;
		cout << "Mean: ";
		for (int k = 0; k < m_k; k++) {
			pair<double,double> mean(0,0);
			int num_features_for_k = m_data[k].size();
			for (auto const & it : m_data[k]) {
				get<0>(mean) += get<0>(it);
				get<1>(mean) += get<1>(it);
			}
			get<0>(mean) /= num_features_for_k;
			get<1>(mean) /= num_features_for_k;
			res = (m_means[k] == mean && res == true) ? true : false;
			cout << "Converged? " << res << "\r\n";
			m_means[k] = mean;
			cout << "cluster " << get<0>(mean) << " , " << get<1>(mean) << "\t";
		}
		cout << "\r\n";
		return res;
		

	}

	void assignLabels() {
		valarray<vector<pair<double,double>>> new_data(m_k);
		for (auto const & clust : m_data) {
			for (auto const & feature : clust) {
				 int closest_mean = this->computeClosestCentroid(feature);
				 new_data[closest_mean].push_back(feature);
			}
		}
		m_data = new_data;

	}

	int computeClosestCentroid(const pair<double,double> & point_) const {
		valarray<double> distances(m_k);
		for (int k = 0; k < m_k; k++) {
			double del_x = get<0>(point_) - get<0>(m_means[k]);
			double del_y = get<1>(point_) - get<1>(m_means[k]);
			double dist = sqrt((del_x * del_x) + (del_y * del_y)); 
			distances[k] = dist;
		}
		auto closest = distance(begin(distances),min_element(begin(distances),end(distances)));

		return closest;

	}


	int m_k;
	int m_features;
	valarray<pair<double,double>> m_means;		// is of length equal to k, the mean is a 2d vector
	valarray<vector<pair<double,double>>> m_data;	// array is of length and holds the vectors of the 
							// data points classified as that label

};



using namespace std;


int main (int argc, char ** argv) {

	vector<pair<double, double>> data = 
		{ { 1.1, 1 }
		, { 1.4, 2 }
		, { 3.8, 7 }
		, { 5.0, 8 }
		, { 4.3, 6 }
		, { 8, 5.0 }
		, { 6, 8.5 }
		, { 3, 2.0 }
		, { 9,   6 }
		, { 9.1, 4 }
		};
	

	kMeans km(3, data);

	valarray<pair<double, double>> init_means = 
                { { 1, 1 }
                , { 3, 4 }
                , { 8, 8 }
                };

	km.clusterData(init_means);

	km.printClusters();


	return 0;
}

