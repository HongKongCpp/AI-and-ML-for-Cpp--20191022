//
//  MIT License
//  
//  Copyright (c) 2019 Miguel Angel Moreno
//  Based on original code by Gerard Taylor
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

#include <cmath>
#include <limits>
#include <map>
#include "stdint.h"
#include "data.hpp"
#include "data_handler.hpp"


// O(k*n) where k is the number of neighbors and N is the size of training data
// O(n) + O(k*n) + k


using namespace std;


class kNearestNeighbors {

public:
    kNearestNeighbors(int val) {
        k = val;
    }

    kNearestNeighbors() {
    
    }

    ~kNearestNeighbors() {
    
    }


    void find_knearest(data *query_point) {
    
        neighbors = new vector<data *>;
        double min = numeric_limits<double>::max();
        double previous_min = min;
        int index;
        for(int i = 0; i < k; i++) {
            if(i == 0) {
                for(int j = 0; j < training_data->size(); j++) {
                    double dist = calculate_distance(query_point, training_data->at(j));
                    training_data->at(j)->set_distance(dist);
                    if(dist < min) {
                        min = dist;
                        index = j;
                    }
                }
                neighbors->push_back(training_data->at(index));
                previous_min = min;
                min = numeric_limits<double>::max();
            } else {
                for(int j = 0; j < training_data->size(); j++) {
                    double dist = training_data->at(j)->get_distance();
                    if(dist > previous_min && dist < min) {
                        min = dist;
                        index = j;
                    }
                }
                neighbors->push_back(training_data->at(index));
                previous_min = min;
                min = numeric_limits<double>::max();
            }
        }
    }

    void set_k(int val) {
        k = val;
    }

    int find_most_frequent_class() {
    
        map<uint8_t, int> freq_map;
        for(int i = 0; i < neighbors->size(); i++) {
            if(freq_map.find(neighbors->at(i)->get_label()) == freq_map.end()) {
                freq_map[neighbors->at(i)->get_label()] = 1;
            } else {
                freq_map[neighbors->at(i)->get_label()]++;
            }
        }

        int best = 0;
        int max = 0;

        for(auto kv : freq_map) {
            if(kv.second > max) {
                max = kv.second;
                best = kv.first;
            }
        }
        delete neighbors;
        return best;

    }

    double calculate_distance(data* query_point, data* input) {
    
        double value = 0;
        if(query_point->get_normalized_feature_vector()->size() != input->get_normalized_feature_vector()->size()) {
            printf("Vector size mismatch.\n");
            exit(1);
        }
        for(unsigned i = 0; i < query_point->get_normalized_feature_vector()->size(); i++) {
            value += pow(query_point->get_normalized_feature_vector()->at(i) - input->get_normalized_feature_vector()->at(i),2);
        }
        return sqrt(value);
    }

    double validate_perforamnce() {
    
        double current_performance = 0;
        int count = 0;
        int data_index = 0;
        for(data *query_point : *validation_data) {
            find_knearest(query_point);
            int prediction = find_most_frequent_class();
            data_index++;
            if(prediction == query_point->get_label())
                count++;
            printf("Current Performance: %.3f %%\n", ((double)count)*100.0 / ((double)data_index));
        }
        current_performance = ((double)count)*100.0/((double)validation_data->size());
        printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
        return current_performance;
    }

    double test_performance() {
        double current_performance = 0;
        int count = 0;
        for(data *query_point : *test_data) {
            find_knearest(query_point);
            int prediction = find_most_frequent_class();
            if(prediction == query_point->get_label())
                count++;
        }
        current_performance = ((double)count)*100.0/((double)test_data->size());
        printf("Validation Performance for K = %d: %.3f\n", k, current_performance);
        return current_performance;
    }

    void set_training_data(vector<data *> *vect) {
        training_data = vect;
    }

    void set_test_data(vector<data *> *vect) {
        test_data = vect;
    }

    void set_validation_data(vector<data *> *vect) {
        validation_data = vect;
    }


protected:

    vector<data *> *training_data;
    vector<data *> *test_data;
    vector<data *> *validation_data;
    vector<data *> *neighbors;
    int k;

};


int main(int argc, char** argv) {

    DataHandler dh;
    dh.read_csv("iris.data",",");
    dh.count_classes();
    dh.split_data();

    kNearestNeighbors knn( 3 );
    knn.set_training_data(dh.get_training_data());
    knn.set_test_data(dh.get_test_data());
    knn.set_validation_data(dh.get_validation_data());

    double performance = 0;
    double best_performance = 0;
    int best_k = 1;
    
    for(int k = 1; k <= 3; k++) {
        if(k == 1) {
            performance = knn.validate_perforamnce();
            best_performance = performance;
        } else {
            knn.set_k(k);
            performance = knn.validate_perforamnce();
            if(performance > best_performance) {
                best_performance = performance;
                best_k = k;
            }
        }
    }
    knn.set_k(best_k);
    knn.test_performance();

}

