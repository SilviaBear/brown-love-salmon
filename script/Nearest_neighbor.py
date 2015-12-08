import os
import sys
import heapq


TRAINING_IMAGES = "../data/trainingimages"
TRAINING_LABELS = "../data/traininglabels"

TESTING_IMAGES = "../data/testimages"
TESTING_LABELS = "../data/testlabels"

IMAGE_PIXEL_HEIGHT = 28
IMAGE_PIXEL_WIDTH = 28
VECTOR_LENGTH = IMAGE_PIXEL_HEIGHT * IMAGE_PIXEL_WIDTH

def get_distance(image1, image2):
    dist = 0
    for i in range(VECTOR_LENGTH):
        if image1[i] == image2[i]:
            dist += 1
    
    return dist

def build_samples():
    samples = []
    with open(TRAINING_IMAGES) as image_fil:
        with open(TRAINING_LABELS) as label_fil:
            while True:
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                    break
                pixel_vector = build_image_vector(image_fil)
                samples.append((int(label), pixel_vector))

    return samples

def test(samples):

    correct = [0 for x in range(10)]
    wrong = [0 for x in range(10)]

    time = 0

    confusion_matrix = [[[0 for x in range(10)] for x in range(10)] for x in range(10)]

    num_test = sys.maxsize

    counter = [[0 for x in range(10)] for x in range(10)]
    
    with open(TESTING_IMAGES) as image_fil:
        with open(TESTING_LABELS) as label_fil:
            while True:
                if time > num_test:
                    break
                time += 1
                label = label_fil.readline().strip("\n")
                if len(label) == 0:
                        break
                
                expected = int(label)
                pixel_vector = build_image_vector(image_fil)
                pq = []
                for i in range(len(samples)):
                     heapq.heappush(pq, (get_distance(pixel_vector, samples[i][1]), i))

                for K in range(1, 11):     
                    voters = heapq.nlargest(K, pq)
                    votes = {}
                    for voter in voters:
                        vote = samples[voter[1]][0]
                        if vote not in votes:
                            votes[vote] = 0
                            votes[vote] = votes[vote] + 1

                    predicted = -1
                    max_vote = 0
                    for digit in votes:
                        if votes[digit] > max_vote:
                            predicted = digit
                            max_vote = votes[digit]

                    counter[K - 1][expected] += 1
                    if predicted == expected:
                        correct[K - 1] += 1
                    else:
                        wrong[K - 1] += 1
                        confusion_matrix[K - 1][expected][predicted] += 1

    accuracy = [0 for x in range(10)]
    print(counter)
    print(correct)
    print(wrong)
    for x in range(10):
        accuracy[x] = correct[x] * 1.0 / (correct[x] + wrong[x])
        for row in range(10):
            for col in range(10):
                confusion_matrix[x][row][col] /= 1.0 * counter[x][row]

        print_matrix(confusion_matrix[x])
        print("NUM " + str(x))
    print(accuracy)

def print_matrix(matrix):

    for i in range(len(matrix)):
        row = str(i) + " "
        for j in range(len(matrix[0])):
            row += ' & ' + str(round(matrix[i][j] * 100, 2))
            row += "\\"
        print(row)
        print("\hline")
    
                        
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

if __name__ == "__main__":
    samples = build_samples()
    test(samples)
