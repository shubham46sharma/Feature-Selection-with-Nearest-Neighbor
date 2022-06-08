# Shubham Sharma
# CS205 Project 2 (Feature Selection with Nearest Neighbor)

import timeit

def parse_file():
    print("Parse function")
    

def Normalize():
    print("Normalise data")

def ForwardSelection():
    print("Forward Selection")

def BackwardElimination():
    print("Backward Elimation")

def main():
    file = input("Enter the name of the file:")
    instances = parse_file(file)
    instance_count = len(instances)
    feature_count = len(instances[0]) - 1
    # Subtract 1 as instances[i][0] is the class label. 
    # Rest n-1 are features

    method = input("""Please select the algorithm you wish to run:
                           1 - Forward Selection
                           2 - Backward Elimination
                        \r""")

    start = timeit.default_timer()
    print("\t***Normalizing...***")
    normalized_instances = Normalize(instance_count, feature_count, instances)
    print("Total number of features: {feature_count}")
    print("Total number of Instances: {instance_count}")

    if (method == 1):
        ForwardSelection(instance_count, feature_count, normalized_instances)
    elif (method == 2):
        BackwardElimination(instance_count, feature_count, normalized_instances)
    else:
        print("Invalid Input")

    stop = timeit.default_timer()
    print("Time:", stop-start)

main()

