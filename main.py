import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from random import randint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        """An implementation of a node in a decision tree classifier"""
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        # Note: Children may not exist, thus, the default value is None
        # Left child of the current node
        self.left = left
        # Right child of the current node
        self.right = right
        # the Information Gain for the current node
        # Computed by the formula: Entropy(parent) - sum(weight(node_i) * Entropy(node_i))
        self.info_gain = info_gain

        self.value = value

class DecisionTreeRegressor():
    def __init__(self, min_sample_split=2, max_depth=2):
        """An implementation of a Classifier Decision Tree, using entropy"""
        # Each split can have 2 or more splits
        # The tree can be smaller than max_depth

        # Define the root of the tree
        self.root = None

        # Base cases
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth

    def fit(self, x, y):
        """Function to train the tree"""
        dataset = np.concatenate((x,y), axis=1)
        self.root = self.build_tree(dataset)
    

    def build_tree(self, dataset, cur_depth=0):
        """Recursively build a binary decision tree"""

        x, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(x)

        best_split = {} #Initializes the  to avoid an UnboundLocalError

        if num_samples >= self.min_sample_split and cur_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)

            if best_split and "info_gain" in best_split and best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], cur_depth + 1)
                right_subtree = self.build_tree(best_split["dataset_right"], cur_depth + 1)
                return Node(
                    feature_index=best_split["feature_index"],
                threshold=best_split["threshold"],
                left=left_subtree,
                right=right_subtree,
                info_gain=best_split["info_gain"]
            )

    # compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        return Node(value=leaf_value)


    
    def get_best_split(self, dataset, num_samples, num_features):
        "Finds the best split for a dataset and return a dictionary"
        # dictionary to store the best split
        best_split = {}
        # Set to negative infinite to any value greater than it will be set as info gain
        max_info_gain = -float("inf")


        # Loop over all features, traverse through all threshold values
        # Go through every possible value from dataset
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            # Only the unique values
            possible_threshold_values = np.unique(feature_values)
            # Loop over all present feature values
            for threshold in possible_threshold_values:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # Makes sure children exist
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    cur_info_gain = self.information_gain(y, left_y, right_y)
                    # update the best split, if needed
                    if cur_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = cur_info_gain
                        max_info_gain = cur_info_gain

        return best_split
        

    def split(self, dataset, feature_index, threshold):
        """function to split the data"""
        # Note:
        # Left is inclusive, as these include all values that meet our node conditions
        # Right is exclusive, as all values that don't meet node condition
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, left_child, right_child, mode="variance"):
        """Calculates the reduction in variance (***not actually information gain***)"""
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        parent_var = np.var(parent)
        left_var = np.var(left_child)
        right_var = np.var(right_child)
        return parent_var - (weight_left * left_var + weight_right * right_var)


    def entropy(self, feature_slice):
        """Calculates entropy based on the probability of the classifier"""
        # My implementation
        # slice_probability = (feature_slice / self.num_features)
        # entropy_slice = -slice_probability * np.log2(slice_probability)
        # return entropy_slice

        feature_labels = np.unique(feature_slice)
        entropy = 0
        for feature in feature_labels:
            probability_feature = len(feature_slice[feature_slice == feature]) / len(feature_slice)
            entropy += -probability_feature * np.log2(probability_feature)
        return entropy
    
    def gini(self, feature_slice):
        """Calculate Gini index based on the probability of the classifier"""
        # My implementation
        # slice_probability = feature_slice / self.num_features
        # gini_slice = 1 - (slice_probability^2)
        # return gini_slice

        feature_labels = np.unique(feature_slice)
        gini = 0
        for feature in feature_labels:
            probability_feature = len(feature_slice[feature_slice == feature]) / len(feature_slice)
            gini += (probability_feature**2)
        return 1-gini
    
    def calculate_leaf_value(self, y):
        """Compute the leaf value using mean"""
        return np.mean(y)

    
    def print_tree(self, tree=None, indent = " "):
        """Prints a visual representation of the Decision Tree"""
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold, "?", tree.info_gain)
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + indent)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + indent)
    def predict(self, x):
        """Function to predict new dataset"""

        prediction = [self.make_prediction(x, self.root) for x in x]
        return prediction
    
    def make_prediction(self, x, tree):
        """Predicts a single data point"""

        # If the node is a leaf node
        if tree.value is not None:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)

def call_api():
    url = "https://api.sportradar.com/mlb/trial/v8/en/league/injuries.json"
    API_KEY = 'api_key'
    params = {'api_key':API_KEY}
    response = requests.get(url, params=params)
    response = response.json()
    
    return response
    
def flip_name(name):
    last_name = str()
    first_name =str()
    j = 0
    while name[j] != ',':
        last_name += str(name[j])
        j+=1
        
    for j in range(j +2, len(name)):
        first_name +=str(name[j])
        
    full_name = f"{first_name} {last_name}"
    return full_name
    
def flip_names(df):
    df['player_name'] = df['player_name'].apply(flip_name)
    return df

def find_injuries(response):
    teams = response["teams"]
    injuries = []
    
    for team in teams:
        team_players = team["players"]
        
        for player in team_players:
            
            if player["position"] == "P":
                pitcher_name = player["full_name"]
                
                injury_info = player["injuries"]
                pitcher_injury = injury_info
                injury_type = pitcher_injury[0]
                injury_type = injury_type['desc']
                injuries.append({'pitcher_name' : pitcher_name, 'injury_type' : injury_type})
    
    
    return injuries

def get_name(pitcher):
    return(pitcher['pitcher_name'])

def create_total_injuries(df, injury_list):
    injury_column = []

    injured_players = [inj['pitcher_name'] for inj in injury_list]
    players_list = df['player_name'].tolist()
    
    i = 0
    j = 0
    
    while i < len(players_list):
        player = players_list[i]
        
        if j < len(injured_players) and player == injured_players[j]:
            injury_column.append(1)
            i += 1
            j += 1
        elif j < len(injured_players) and injured_players[j] not in players_list:
            j += 1
        else:
            injury_column.append(0)
            i += 1
    
    return injury_column

class RandomTree():
    def __init__(self, smallest_index, number_features, keep_columns, starting_df, min_sample_split, max_depth):
        """An implementation random decision tree"""
        # Has to be near the top, otherwise function call happens first, leading to a no attribute error
        self.smallest_index = smallest_index
        self.number_features = number_features
        self.pruned_features = sqrt(number_features)
        self.random_tree_df = pd.DataFrame(columns=keep_columns)

        self.createTreeDataSet(starting_df)
        self.tree = DecisionTreeRegressor(min_sample_split, max_depth)

    def chooseRows(self, df):
        """Pick random rows with replacement"""
        output = []
        for _ in range(len(df)):
            row_index = randint(1, len(df)-1)
            # Add row to dataframe
            output.append(row_index)
        return output

    def chooseFeatures(self, all_columns):
        """Pick random columns with replacement"""
        i = self.number_features
        # Drops number_features amount of columns
        while i > self.pruned_features:
            columns = self.random_tree_df.columns.to_list()
            # Find a random columns index
            feature_index = randint(self.smallest_index, self.number_features - 1)
            # Unless its already dropped, drop it
            if all_columns[feature_index] in columns:
                self.random_tree_df.drop(columns=all_columns[feature_index])
                i -= 1

    def createTreeDataSet(self, starting_df):
        """Create random decision trees for random forest classification"""

        selected_rows = self.chooseRows(starting_df)

        for row_index in selected_rows:
            self.random_tree_df.loc[len(self.random_tree_df)] = starting_df.iloc[row_index]

        all_columns = starting_df.columns.to_list()
        self.chooseFeatures(all_columns)

class RandomForestRegressor():
    def __init__(self, shaping_data, columns, n_trees=10, min_sample_split=2, max_depth=3, number_features=None, starting_feature_index=0):
        """Random Forest built using DecisionTreeRegressor"""
        self.shaping_data = shaping_data
        self.n_trees = n_trees
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.number_features = number_features
        self.starting_feature_index = starting_feature_index
        self.trees = []
        self.columns = columns

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if self.number_features is None:
            self.number_features = int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            feature_indices = np.random.choice(n_features, self.number_features, replace=False)

            X_subset = X_sample[:, feature_indices]
            tree = RandomTree(self.starting_feature_index, self.number_features, columns, self.shaping_data, self.min_sample_split, self.max_depth)
            tree.tree.fit(X_subset, y_sample)
            # tree = DecisionTreeRegressor(min_sample_split=self.min_sample_split, max_depth=self.max_depth)
            # tree.fit(X_subset, y_sample)
# 
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_preds = np.zeros((len(self.trees), X.shape[0]))
        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset = X[:, feature_indices]
            preds = tree.tree.predict(X_subset)
            tree_preds[i] = preds

        return np.mean(tree_preds, axis=0)

# Use this for iris testing:
    # columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    # num_columns = len(columns)
    # df = pd.read_csv("iris.csv", skiprows=1, header=None, names=columns)

columns = ["pitches","player_id","player_name","total_pitches",
           "spin_rate","velocity","effective_speed","release_extension"
           ,"k_percent","bb","bb_percent","release_pos_z","release_pos_x","arm_angle"]

test_dataframe = pd.read_csv('savant_data.csv', usecols=columns)
test_dataframe = flip_names(test_dataframe)


test_sorted = test_dataframe.sort_values(by = "player_name")
test_sorted.dropna()
test_sorted.to_csv("output.csv")
tree_data = test_sorted

columns = test_sorted.columns.to_list()
num_columns = len(columns)
df = test_sorted
number_features = 9
starting_index = 4
tree = RandomTree(starting_index, number_features, columns, df, 3, 3)
# tree.createTreeDataSet(test_sorted)
x = df.drop(columns=['k_percent']).values
y = df['k_percent'].values.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=41)
forest = RandomForestRegressor(test_sorted, columns, n_trees=100, min_sample_split=3, max_depth=5, number_features=number_features, starting_feature_index=starting_index)
forest.fit(x_train, y_train)
y_pred = forest.predict(x_test)
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
