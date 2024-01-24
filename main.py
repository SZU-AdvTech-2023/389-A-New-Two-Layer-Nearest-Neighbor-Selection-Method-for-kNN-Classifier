import math
import random
from tqdm import tqdm
random.seed(42)


def shuffle_matrix(matrix):   # disrupting dataset contents by rows
    num_rows = len(matrix)
    for i in range(num_rows):
        random_index = random.randint(0, num_rows - 1)
        matrix[i], matrix[random_index] = matrix[random_index], matrix[i]


def remove_duplicates(input_list):   # removes duplicates from the input list and returns a new list with unique elements.
    unique_list = list(set(input_list))
    return unique_list


def check_element(data, target_element):   # check if the list contains a specific element
    return target_element in data    # The function returns a boolean value, True if target_element exists in data, False otherwise.


def read_data(filepath, label_loc):    # read data from txt file and convert to matrix
    with open(filepath, 'r') as file:
        lines = file.readlines()

    shuffle_matrix(lines)      # smash data by rows

    label_loc = label_loc - 1
    label = []
    for i, line in enumerate(lines):     # extract label columns by the input of label_loc, username will be added, such as: [1, M], M is the class of 1
        values = line.strip().split('\t')
        label.append([i + 1, determine_type(values[label_loc])])

    data = []
    for i, line in enumerate(lines):    # extract other columns as data of each instance, username will be added in first column
        values = line.strip().split('\t')
        data.append([i + 1] + [determine_type(value) for j, value in enumerate(values) if j != label_loc])

    return data, label


def determine_type(value):
    # Attempts to convert the input value to a floating point number.
    # If the conversion succeeds, it further checks if the float is an integer, and returns the integer type if it is an integer,
    # otherwise it returns the float type
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        else:
            return float_value
    except ValueError:
        return value


def split_dataset(data, label, which_one):  # divide the data into ten parts, realization of ten cross validations
    subset_size = len(data) // 10
    results = []
    for i in range(10):
        if i < 9:
            test_indices = list(range(i * subset_size, (i + 1) * subset_size))
        else:   # The last part would be to make up the part that could not be divided into ten parts.
            length = len(data)
            test_indices = list(range(i * subset_size, length))
        train_indices = [j for j in range(len(data)) if j not in test_indices]
        test_set_data = [data[idx] for idx in test_indices]
        test_set_label = [label[idx] for idx in test_indices]
        train_set_data = [data[idx] for idx in train_indices]
        train_set_label = [label[idx] for idx in train_indices]
        results.append({
            'train_data': train_set_data,
            'train_label': train_set_label,
            'test_data': test_set_data,
            'test_label': test_set_label
        })
    selected_set = results[which_one]
    return selected_set['train_data'], selected_set['train_label'], selected_set['test_data'], selected_set['test_label']
    # The return values are four matrices, including training set data and labels, test set data and labels


def dist(c1, c2):  # Calculate the distance between two points
    dim = len(c1)
    dis = 0
    for i in range(dim):  # The two instances should be in the same dimension
        dis = (c1[i] - c2[i]) ** 2 + dis
    dis = math.sqrt(dis)
    return dis


def cent(data):    # find the centroid of some point
    num_rows = len(data)
    num_columns = len(data[0])
    centroid = [sum(data[i][j] for i in range(num_rows)) / num_rows for j in range(num_columns)]
    return centroid


def bottom_k_indices(data, k):    # sort by bubble method
    n = len(data)
    indices = list(range(n))
    for i in range(n):
        for j in range(0, n-i-1):
            if data[indices[j]] > data[indices[j + 1]]:
                indices[j], indices[j+1] = indices[j+1], indices[j]
    return indices[:k], data[indices[k-1]]   # return row indices and k-th small data


def find_row_by_id(data, name):     # find the row by user_id, from 0 to ...., so, it is indices
    for i, row in enumerate(data):
        if row[0] == name:
            return i
            break
    else:
        return False


def k_nearest_neigh(data, query, k):      # find the k_nearst_neighborhood of query and the radius
    dist_query_sample = [0 for _ in range(len(data))]
    for i in range(len(data)):
        if data[i][0] != query[0]:    # if name not same
            c1 = query[1:]    # withdraw the data part
            c2 = data[i][1:]
            distance_c1_c2 = dist(c1, c2)
        else:
            distance_c1_c2 = 2e20
        dist_query_sample[i] = [data[i][0], distance_c1_c2]    # dist_query_sample format:[name, distance]
    dist_without_name = [row[1:] for row in dist_query_sample]    # dist_without_name format:[distance]
    [first_nn, radius] = bottom_k_indices(dist_without_name, k)    # first_nn is the indices
    list_name = []
    for i in range(len(first_nn)):
        list_name.append(data[first_nn[i]][0])
    return list_name, radius[0]        # the name list of query's knn, and the max distance


def count_unique_elements(label):       # check the label, and find all the classes' name
    unique_elements = []

    for element in label:
        if element not in unique_elements:
            unique_elements.append(element)

    return unique_elements


def max_one_idx(martix):        # find the max one, return its indices
    max_value = float('-inf')
    max_index = None
    for i in range(len(martix)):
        if martix[i] > max_value:
            max_value = martix[i]
            max_index = i
    return max_index


def majority_vote(data):   # Realization of the equation for the majority voting method
    class_kind = count_unique_elements(data)
    num = len(class_kind)
    cout_each_class = [0 for _ in range(num)]
    for i in range(len(data)):
        for j in range(num):
            if data[i] == class_kind[j]:
                cout_each_class[j] = cout_each_class[j] + 1
    kind = max_one_idx(cout_each_class)
    return class_kind[kind]


def acc(data, label):   # calculate the error rate
    score = 0
    for row in range(len(data)):
        ind = find_row_by_id(label, data[row][0])
        fail_num = 0
        for col in range(1, len(data[row])):
            if data[row][col] != label[ind][1]:
                fail_num = fail_num + 1
        score = fail_num / (len(data[row])-1) + score
    score = score / len(data)
    return score


def kTLNN(file_path, column, k, kb):
    dataset, label = read_data(file_path, column)
    result = [[] for _ in range(len(dataset))]
    if len(dataset) == len(label):
        for i in range(len(label)):
            result[i] = [label[i][0]]
    with tqdm(total=10, desc='Processing') as pbar:
        for loop_i in range(10):    # ten cross validations
            sample_set, sample_label, test_set, test_label = split_dataset(dataset, label, loop_i)
            for loop_j in range(len(test_set)):    # iterate through each instance in the test set
                knn_query, radius = k_nearest_neigh(sample_set, test_set[loop_j], k)  # knn_query: query's k-nearest neighbor
                nn_ext_query = []
                for loop_k in range(len(knn_query)):    # Operate on each knn_query instance
                    nn_eff_knn_query = []
                    knn_name = knn_query[loop_k]
                    nn_eff_knn_query.append(knn_name)
                    indices_knn_name = find_row_by_id(dataset, knn_name)
                    knn_data = dataset[indices_knn_name]
                    knn_query_knn, noname = k_nearest_neigh(sample_set, knn_data, k)   # knn_query's k-nearest neighbor, save the name in knn_query_knn
                    # if dist(knn's knn, query) <= 2*radius, it is NN_eff point, save in nn_eff_knn_query
                    # traversal all the knn_query's k-nearest neighbor
                    for loop_l in range(len(knn_query_knn)):
                        knn_query_knn_name = knn_query_knn[loop_l]
                        indices_knn_knn = find_row_by_id(dataset, knn_query_knn_name)
                        knn_query_knn_data = dataset[indices_knn_knn]
                        distance = dist(test_set[loop_j][1:], knn_query_knn_data[1:])
                        if distance <= radius * 2:
                            nn_eff_knn_query.append(knn_query_knn_name)
                    data = []
                    for loop_m in range(len(nn_eff_knn_query)):
                        name = nn_eff_knn_query[loop_m]
                        indice_1 = find_row_by_id(dataset, name)
                        data.append(dataset[indice_1][1:])
                    cent_1 = cent(data)
                    query = test_set[loop_j][1:]
                    dist1 = dist(cent_1, query)
                    dist2 = dist(knn_data[1:], query)
                    if dist1 < dist2:    # determine if the distance from the center of mass to the query is lower than the distance from knn_query to the query
                        for loop_n in range(len(nn_eff_knn_query)):
                            nn_ext_query.append(nn_eff_knn_query[loop_n])
                    else:
                        nn_ext_query.append(knn_name)
                nn_two_query_cand = remove_duplicates(nn_ext_query)    # remove duplicate instances in NN_ext(x)
                # Determine whether the kb nearest neighbor of a point in NN_ext(x) contains the query x
                nn_two_query = []   # NN_two(x)
                sample_set_query = []
                sample_set_query = sample_set
                sample_set_query.append(test_set[loop_j])
                for loop_o in range(len(nn_two_query_cand)):
                    nn_two_name = nn_two_query_cand[loop_o]
                    indice_2 = find_row_by_id(dataset, nn_two_name)
                    nn_two_data = dataset[indice_2]
                    nn_kb, noname = k_nearest_neigh(sample_set_query, nn_two_data, kb)
                    result_query = check_element(nn_kb, test_set[loop_j][0])
                    if result_query:   # If it contains, then it is added to the two-layer-nearest neighbor set NN_two(x) of the query x
                        nn_two_query.append(nn_two_name)
                if not nn_two_query:   # If NN_two(x) is the empty set then use the kb nearest neighbor of query x instead
                    nn_two_query, noname = k_nearest_neigh(sample_set, test_set[loop_j], kb)
                nn_two_query_label = []
                # calculate the error rate
                for loop_p in range(len(nn_two_query)):
                    name1 = nn_two_query[loop_p]
                    indice_3 = find_row_by_id(label, name1)
                    nn_two_query_label.append(label[indice_3][1])
                query_result = majority_vote(nn_two_query_label)
                indice = find_row_by_id(label, test_set[loop_j][0])
                result[indice].append(query_result)
            pbar.update(1)
    error_rate = acc(result, label)
    return error_rate


file_path = './dataset/glass.txt'   # the path of dataset
multirun = 10    # number of repeated runs
column = 10    # number of columns of labeled columns(starting from 1)
save_path = 'result_Ionosphere.txt'    # where to save your calculations
with open(save_path, "a") as file:
    file.write(f"dataset: {file_path}\n")
    for k in range(1, 21):    # k=1,2,...,20
        rate = 1
        kb = 0
        while rate <= 2:   # k_b={1.0,1.2,1.4,1.6,1.8,2.0}*k
            kb = int(k * rate)   # rounding operation
            result = 0
            max = 1   # minimum error rate after multiple repetitions, initialize it
            print(f"k:{k}, kb: {rate} * k:")
            for loop_times in range(multirun):
                print(f"{loop_times+1}-th:\n")
                result_i = kTLNN(file_path, column, k, kb)
                result = result + result_i
                if result_i < max:
                    max = result_i
                print(max)
            print(result_i)
            result = result / multirun    # average error rate
            file.write(f"k: {k}, kb: {rate} * k, average result: {result}, max: {max}\n")   # printing the result in a file, which is in the save_path
            rate = rate + 0.2   # The representation of 1.6 to 2.0 will be small due to the problem of decimal representation, e.g., 1.6 is represented as 1.599999999999
