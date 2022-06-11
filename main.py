# Shubham Sharma
# CS205 Project 2 (Feature Selection with Nearest Neighbor)

import timeit
import math

def parse_file(file):
    # print("Parse function")
    data = []
    with open(file,'r') as file:
        lines = [line.rstrip().split() for line in file]
    for line in lines:
        data.append([float(i) for i in line])
    return data


def Normalize(instance_count, feature_count, instances):
    # print("Normalise data")
    #Normalizing the data so that each attribute has a mean of 0
    # and standard deviation of 1
    normalized_instances = list(instances)
    mean = []
    for i in range(1, feature_count + 1):
        mean.append((sum(row[i] for row in instances)) / instance_count)

    std = []
    for i in range(1, feature_count + 1):
        std.append(math.sqrt((sum(pow((row[i] - mean[i-1]), 2) for row in instances)) / instance_count))
        
    for i in range(0, instance_count):
        for j in range(1, feature_count + 1):
            normalized_instances[i][j] = ((instances[i][j] - mean[j-1]) / std[j-1])

    return normalized_instances

def NearestNeighborClassification(instances, num_instances, one_out, features):
    nearest_neighbor = -1
    nearest_neighbor_distance = float('inf')
    num_features = len(features)
    for i in range(0, num_instances):
        if (i == one_out):
            pass
        else:
            sum = 0
            for j in range(0, num_features):
                sum = sum + pow((instances[i][features[j]] - instances[one_out][features[j]]), 2)
            distance = math.sqrt(sum)
            if distance < nearest_neighbor_distance:
                nearest_neighbor_distance = distance
                nearest_neighbor = i
    #final check for the class
    if(instances[nearest_neighbor][0] != instances[one_out][0]):
        return False
    return True

def Leave1outCV(data, instance_count, current_features, my_feature, minimax):
    count = 0

    if my_feature > 0:
        list_features = list(current_features)
        list_features.append(my_feature)
    elif my_feature == 0:
        list_features = list(current_features)
    elif my_feature < 0:
        my_feature = my_feature * -1
        current_features.remove(my_feature)
        list_features = list(current_features)
        current_features.add(my_feature)

    correct_count = 0
    if minimax == -1:
        for i in range(0,instance_count):
            one_out = i
            correct_classification = NearestNeighborClassification(data,instance_count,one_out,list_features)
            if (correct_classification):
                num_correct += 1
    else:
        for i in range(0, instance_count):
            #calculating confidence for optimization
            confidence = (correct_count + (instance_count - count)) / instance_count    
            count += 1
            if confidence >= minimax:         # Condition which checks the pruning
                one_out = i
                correct_classification = NearestNeighborClassification(data, instance_count, one_out, list_features)
                if (correct_classification):
                    correct_count += 1
            else:
                return -1
    accuracy = correct_count / instance_count
    print("Testing features: ", list_features, " with accuracy %f" % accuracy)
    return accuracy


def ForwardSelection(instance_count, feature_count, data):
    print("Forward Selection")
    print("*"*50)
    feature_set = set()
    curr_best_accuracy = 0
    for i in range(feature_count):
        print("On level %d of the search tree" % (i+ 1),"with our set as", feature_set)
        feature_to_add = -1
        for j in range(1, feature_count + 1):
            if (j not in feature_set):
                accuracy = Leave1outCV(data, instance_count,feature_set, j,curr_best_accuracy)
                if accuracy == -1:
                    print("Search tree pruned")
                else:
                    if accuracy > curr_best_accuracy:
                        curr_best_accuracy = accuracy
                        feature_to_add = j
        if (feature_to_add > 0):
            feature_set.add(feature_to_add)
            print("On level %d of the search tree," % ((i+1)),"adding feature %d gives accuracy: %f" % (feature_to_add, curr_best_accuracy))
            print("-" * 50)
        else:
            print("Accuracy decreasing, optima reached")
            break
    print("-" * 25,"Important Features Found","-" * 25)
    print("Best subset of features to use: ", feature_set,"with accuracy", curr_best_accuracy)

def BackwardElimination(instance_count, feature_count, data):
    print("Backward Elimation")
    print("*"*50)
    feature_set = set(i+1 for i in range(0, feature_count))
    curr_best_accuracy = 0
    for i in range(feature_count):
        print("On level %d of the search tree" % (i+1), "with our set as", feature_set)
        feature_to_remove = -1
        for j in range(1, feature_count + 1):
            if (j in feature_set):
                accuracy = Leave1outCV(data, instance_count,feature_set, (-1 *j),curr_best_accuracy)
                if accuracy == -1:
                    print("search tree pruned") 
                else:
                    if accuracy > curr_best_accuracy:
                        curr_best_accuracy = accuracy
                        feature_to_remove = j
        if (feature_to_remove > 0):
            feature_set.remove(feature_to_remove)
            print("On level %d of the search tree," % (i+1), "removing feature %d gives accuracy: %f" % (feature_to_remove, curr_best_accuracy))
            print("-" * 50)
        else:
            print("Accuracy decreasing, optima reached")
            break
    print("*" * 25,"Important Features Found","*" * 25)
    print("Best subset of features to use:", feature_set, "with accuracy", curr_best_accuracy)


def main():
    file = input("Enter the name of the file:")
    instances = parse_file(file)
    instance_count = len(instances)
    feature_count = len(instances[0]) - 1
    # Subtract 1 as instances[i][0] is the class label. 
    # Rest n-1 are features

    method = int(input("""Please select the algorithm you wish to run:
                           1 - Forward Selection
                           2 - Backward Elimination
                        \r"""))

    start = timeit.default_timer()
    print("\t*** Normalizing ***")
    normalized_instances = Normalize(instance_count, feature_count, instances)
    print("Total number of features: ", feature_count)
    print("Total number of Instances: ", instance_count)

    if (method == 1):
        ForwardSelection(instance_count, feature_count, normalized_instances)
    elif (method == 2):
        BackwardElimination(instance_count, feature_count, normalized_instances)
    else:
        print("Invalid Input")

    stop = timeit.default_timer()
    print("Time:", stop-start)

main()