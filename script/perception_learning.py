#!/usr/bin/python

import math
import sys

#For digit classification
TRAINING_IMAGES = "../data/trainingimages"
TRAINING_LABELS = "../data/traininglabels"

TESTING_IMAGES = "../data/testimages"
TESTING_LABELS = "../data/testlabels"

IMAGE_PIXEL_HEIGHT = 28
IMAGE_PIXEL_WIDTH = 28

VECTOR_LENGTH = IMAGE_PIXEL_HEIGHT * IMAGE_PIXEL_WIDTH

bias = 1

num_epochs = 10

def decay_func(epoch):
    return 1000.0 / (1000 + epoch)

def initiate_bias():
    return [bias for x in range(10)]

def initiate_weights():
    weights = [[0 for x in range(VECTOR_LENGTH)] for x in range(10)]
    return weights

def build_image_vector(image_fil):
    pixel_vector = []
    for i in range(IMAGE_PIXEL_HEIGHT):
        row = image_fil.readline().strip("\n")
        for j in range(IMAGE_PIXEL_WIDTH):
            pixel = row[j]
            if pixel == ' ':
                pixel_vector.append(0)
            else:
                pixel_vector.append(1)

    return pixel_vector

def sgn(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    else:
        return 0

def add_vector(v1, v2):
    _v = [(v1[x] + v2[x]) for x in range(len(v1))]
    return _v
    
def multiply_vector(v1, v2):
    _v = 0
    for x in range(len(v1)):
        _v += v1[x] * v2[x]

    return _v

def multiply_vector_by_constant(v, b):
    for x in range(len(v)):
        v[x] *= b
    return v

def predict_class(weights, pixel_vector, bias):
    predicted_class = 0
    max_val = -sys.maxsize
    for x in range(10):
        if bias != None:
            _val = multiply_vector(weights[x], pixel_vector), bias[x]
        else:
            _val = multiply_vector(weights[x], pixel_vector)
            
        if _val > max_val:
            max_val = _val
            predicted_class = x

    return predicted_class

def train():
    weights = initiate_weights()
    bias = initiate_bias()
    for time in range(num_epochs):
        with open(TRAINING_IMAGES) as image_fil:
            with open(TRAINING_LABELS) as label_fil:
                while True:
                    label = label_fil.readline().strip("\n")
                    if len(label) == 0:
                        break

                    expected_class = int(label)
                    pixel_vector = build_image_vector(image_fil)
                    predicted_class = predict_class(weights, pixel_vector, bias)
                    
                    if expected_class != predicted_class:
                        weights[expected_class] = add_vector(weights[expected_class], multiply_vector_by_constant(pixel_vector, decay_func(time)))
                        weights[predicted_class] = add_vector(weights[expected_class], multiply_vector_by_constant(pixel_vector, -decay_func(time)))

    return weights

def test(weights):
    confusion_matrix = [[0 for x in range(10)] for x in range(10)]
    correct = 0
    wrong = 0

    counter = [0 for x in range(10)]
    
    with open(TESTING_IMAGES) as image_fil:
        with open(TESTING_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                        break

                expected_class = int(label)
                pixel_vector = build_image_vector(image_fil)
                predicted_class = predict_class(weights, pixel_vector, None)
                counter[expected_class] += 1
                
                if expected_class == predicted_class:
                    correct += 1
                else:
                    wrong += 1
                    confusion_matrix[expected_class][predicted_class] += 1

    for row in range(10):
        for col in range(10):
            confusion_matrix[row][col] /= 1.0 * counter[row]

    accuracy = correct * 1.0 / (correct + wrong)
            
    return confusion_matrix, accuracy 
                
                
def print_matrix(matrix):

    for i in range(len(matrix)):
        row = str(i) + " "
        for j in range(len(matrix[0])):
            row += ' & ' + str(round(matrix[i][j] * 100, 2))
            row += "\\"
        print(row)
        print("\hline")

def parse_matrix_to_string(matrix):
    _str = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            _str += str(matrix[i][j])

    return _str

if __name__ == "__main__":
    weights = train()
    confusion_matrix, accuracy = test(weights)
    print_matrix(confusion_matrix)
    print(accuracy)
