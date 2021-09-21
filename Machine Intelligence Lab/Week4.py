import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """
    
    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p
    
    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """
        
        # print(data)

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def p_root(value, root):
     
        root_value = 1 / float(root)
        return round (Decimal(value) **
                 Decimal(root_value), 3)
 
    def minkowski_distance(x, y, p_value):
     
    # pass the p_root function to calculate
    # all the value of vector parallelly
        return (p_root(sum(pow(abs(a-b), p_value) for a, b in zip(x, y)), p_value))


    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        
        dist_213 = [[0 for i in range(len(self.target))] for j in range(len(x))]
        
        for i in range(len(x)):
            for j in range(len(self.data)):
                dist_213[i][j] = minkowski_distance(x[i],data[j])
            
        
        # pass
        # DONE

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input
            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        
        dist_arr_213 = []
        
        len_213 = self.data.shape[0]
        for i in range(len_213):
            j = {'dist':self.find_distance(self.data[i], x), 'id':i}
            dist_arr_213.append(j)
        
        sort_213 = sorted(dist_arr_213, key = lambda i : i['dist'])
        final_213 = sort_213[0:self.k_neigh]
        
        return final_213
        
        # pass
        # DONE

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        
        pred_213 = np.array([])
        
        for i in x:
            neigh_213 = self.k_neighbours(i)
            
            dict_213 = {}
            for n_213 in neigh_213:
                class_213 = int(self.data[n_213['y']])
                if str(class_213) not in dict_213:
                    dict_213[str(class_213)] = 1
                else:
                    dict_213[str(class_213)] += 1
            val_213 = int(max(dict_213.items()))
            
            pred_213 = np.append(pred_213, val_213)
            
        return pred_213
        
        # pass
        # DONE

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        
        length_213 = x.shape[0]
        pred_213 = self.predict(x)
        
        count_213 = 0
        
        for i in range(len(pred_213)):
            if pred_213[i] == y[i]:
                count_213 += 1
        
        accuracy_213 = float(count_213/length_213)
        return accuracy_213
        
        # pass
        # DONE
        


def test_case1():
    data = np.array([[2.7810836, 2.550537003, 0],
                 [1.465489372, 2.362125076, 0],
                 [3.396561688, 4.400293529, 0],
                 [1.38807019, 1.850220317, 0],
                 [3.06407232, 3.005305973, 0],
                 [7.627531214, 2.759262235, 1],
                 [5.332441248, 2.088626775, 1],
                 [6.922596716, 1.77106367, 1],
                 [8.675418651, -0.242068655, 1],
                 [7.673756466, 3.508563011, 1]])
    X = data[:, 0:2]
    y = data[:, 2]

    dist = np.array([[0.0, 1.3290173915275787, 1.9494646655653247, 1.5591439385540549, 0.5356280721938492,
                    4.850940186986411, 2.592833759950511, 4.214227042632867, 6.522409988228337, 4.985585382449795],
                    [1.3290173915275787, 0.0, 2.80769851166859, 0.5177260009197887, 1.7231219074407058, 6.174826117844725,
                    3.876611681862114, 5.4890230596711325, 7.66582710454398, 6.313232155500879]])

    model = KNN(k_neigh=2, p=2)
    model.fit(X, y)

    kneigh_dist = np.array([[0., 0.53562807],
                            [0., 0.517726]])

    kneigh_idx = np.array([[0, 4],
                        [1, 3]], dtype=np.int64)

    sample = np.array([[2.6, 3.4], [5.2, 4.33]])

    pred = np.array([0, 0])
    try:
        np.testing.assert_array_almost_equal(
            model.find_distance(X[0:2, :]), dist, decimal=2)
        print("Test Case 1 for the function find_distance PASSED")
    except:
        print("Test Case 1 for the function find_distance FAILED")

    try:
        np.testing.assert_array_almost_equal(
            model.k_neighbours(X[0:2, :])[0], kneigh_dist, decimal=2)
        print("Test Case 2 for the function k_neighbours (distance) PASSED")
    except:
        print("Test Case 2 for the function k_neighbours (distance) FAILED")

    try:
        np.testing.assert_array_equal(
            model.k_neighbours(X[0:2, :])[1], kneigh_idx)
        print("Test Case 3 for the function k_neighbours (idx) PASSED")
    except:
        print("Test Case 3 for the function k_neighbours (idx) FAILED")

    try:
        np.testing.assert_array_equal(
            model.predict(sample), pred)
        print("Test Case 4 for the function predict PASSED")
    except:
        print("Test Case 4 for the function predict FAILED")

    try:
        assert model.evaluate(sample, np.array([0, 1])) == 50
        print("Test Case 5 for the function evaluate PASSED")
    except:
        print("Test Case 5 for the function evaluate FAILED")


test_case1()  
  
        
'''        
def test_case2():
    data = np.array([[0.68043616, 0.39113473, 0.1165562 , 0.70722573, 0],
       [0.67329238, 0.69782966, 0.73278321, 0.78787406, 0],
       [0.56134898, 0.25358895, 0.10497708, 0.05846073, 1],
       [0.6515744 , 0.85627836, 0.44305142, 0.53280211, 0],
       [0.47014548, 0.18108572, 0.3235044 , 0.45490616, 0],
       [0.33544621, 0.51322212, 0.98769111, 0.53091437, 0],
       [0.4577167 , 0.80579291, 0.19350921, 0.46502849, 0],
       [0.25709202, 0.06937377, 0.92718944, 0.54662592, 1],
       [0.07637632, 0.3176806 , 0.74102328, 0.32849423, 1],
       [0.2334587 , 0.67725537, 0.4323325 , 0.38766629, 0]])
    X_train = data[:, :4]
    y_train = data[:, 4]
    samples = np.array([[0.41361609, 0.45603303, 0.33195254, 0.09371524, 1],
       [0.19091752, 0.07588166, 0.03198771, 0.15245555, 1],
       [0.29624916, 0.80906772, 0.35025253, 0.78940926, 0],
       [0.96729604, 0.89730852, 0.39105022, 0.37876973, 0],
       [0.52963052, 0.29303055, 0.27697515, 0.67815307, 1]])
    X_test = samples[:, :4]
    y_test = samples[:, 4]
    kneigh_dist = np.array([[0, 0.87960746, 0.91697707], [0, 0.72497042, 1.01071404]])
    kneigh_idx = np.array([[0, 4, 2], [1, 3, 0]])
    pred = np.array([0, 1, 0, 0, 0])
    model = KNN(k_neigh = 3, p = 1, weighted=True)
    model.fit(X_train, y_train)
    try:
        np.testing.assert_array_almost_equal(
            model.k_neighbours(X_train[0:2, :])[0], kneigh_dist, decimal=2)
        print("Test Case 1 for the function k_neighbours (distance) PASSED")
    except:
        print("Test Case 1 for the function k_neighbours (distance) FAILED")
    try:
        np.testing.assert_array_equal(
            model.k_neighbours(X_train[0:2, :])[1], kneigh_idx)
        print("Test Case 2 for the function k_neighbours (idx) PASSED")
    except:
        print("Test Case 2 for the function k_neighbours (idx) FAILED")
    try:
        np.testing.assert_array_equal(
            model.predict(X_test), pred)
        print("Test Case 3 for the function predict PASSED")
    except:
        print("Test Case 3 for the function predict FAILED")
    try:
        assert model.evaluate(X_test, y_test) == 60
        print("Test Case 4 for the function evaluate PASSED")
    except:
        print("Test Case 4 for the function evaluate FAILED")
        
        
test_case2()
'''
