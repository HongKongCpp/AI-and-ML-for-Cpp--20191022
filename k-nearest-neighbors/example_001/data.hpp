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

#ifndef __DATA_H
#define __DATA_H

#include <vector>
#include "stdint.h" // uint8_t 
#include "stdio.h"


using namespace std;


class data
{

public:

    void set_distance(double dist)
    {
        m_distance = dist;
    }

    void set_feature_vector(vector<uint8_t>* vect)
    {
        m_feature_vector = vect;
    }

    void set_normalized_feature_vector(vector<double>* vect)
    {
        m_normalized_feature_vector = vect;
    }

    void append_to_feature_vector(uint8_t val)
    {
        m_feature_vector->push_back(val);
    }

    void append_to_feature_vector(double val)
    {
        m_normalized_feature_vector->push_back(val);
    }

    void set_label(uint8_t val)
    {
        m_label = val;
    }

    void set_enumerated_label(uint8_t val)
    {
        m_enumerated_label = val;
    }

    void setClassVector(int classCounts)
    {
        class_vector = new vector<int>();
        for(int i = 0; i < classCounts; i++)
        {
            if(i == m_label)
                class_vector->push_back(1);
            else
                class_vector->push_back(0);
        }
    }

    void print_vector()
    {
        printf("[ ");
        for(uint8_t val : *m_feature_vector)
        {
            printf("%u ", val);
        }
        printf("]\n");
    }

    void print_normalized_vector()
    {
        printf("[ ");
        for(auto val : *m_normalized_feature_vector)
        {
            printf("%.2f ", val);
        }
        printf("]\n");
        
    }

    double get_distance()
    {
        return m_distance;
    }

    int get_feature_vector_size()
    {
        return m_feature_vector->size();
    }

    uint8_t get_label()
    {
        return m_label;
    }

    uint8_t get_enumerated_label()
    {
        return m_enumerated_label;
    }

    vector<uint8_t> * get_feature_vector()
    {
        return m_feature_vector;
    }

    vector<double> * get_normalized_feature_vector()
    {
        return m_normalized_feature_vector;
    }

    vector<int>    getClassVector()
    {
        return *class_vector;
    }


private:

    vector<uint8_t> *m_feature_vector;
    vector<double> *m_normalized_feature_vector;
    vector<int> *class_vector;
    uint8_t m_label; 
    uint8_t m_enumerated_label; // A -> 1
    double m_distance;

};

#endif

