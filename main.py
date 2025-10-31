import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from random import randint

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

class DecisionTreeClassifier():
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

        x, y = dataset[:,:-1], dataset[:,-1]
        # Extract the number of samples and the number of features
        num_samples, num_features = np.shape(x)

        if num_samples >= self.min_sample_split and cur_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # Creates all of the left subtrees, reaches a leaf node, then creates all of the right subtrees
            if best_split["info_gain"] > 0:
                left_subtree = self.build_tree(best_split["dataset_left"], cur_depth=1)
                right_subtree = self.build_tree(best_split["dataset_right"], cur_depth = 1)
                # Return the decision node for the current classifier, decision node
                return Node(feature_index=best_split["feature_index"], threshold=best_split["threshold"], left=left_subtree, right=right_subtree, info_gain=best_split["info_gain"])
        # compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
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
                    cur_info_gain = self.information_gain(y, left_y, right_y, "gini")
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

    def information_gain(self, parent, left_child, right_child, mode="gini"):
        """Calculates the information gain of a split, defaults to entropy"""
        information_gain = 0
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        if mode == "entropy":
            information_gain = self.entropy(parent) - (self.entropy(left_child) * weight_left + self.entropy(right_child) * weight_right)
        elif mode == "gini":
            information_gain = self.gini(parent) - (self.gini(left_child) * weight_left + self.gini(right_child) * weight_right)
        return information_gain

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
        """Compute the leaf value using a majority voting system"""
        y = list(y)
        return max(y, key=y.count)
    
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
    API_KEY = 'tTKcLzpDz1zoW2wuXRrgBnRxzmeMSOuUPWaAgVf0'
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
    injury_types = [inj['injury_type'] for inj in injury_list]
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

            
         

    
# df = pd.read_csv("./savant_data.csv")
# df = flip_names(df)
# df_sorted = df.sort_values(by = "player_name")
# response = call_api()
# injuries = find_injuries(response)
# sorted_injuries = sorted(injuries, key =get_name)

# injury_column = create_total_injuries(df_sorted, sorted_injuries)
# injury_df = pd.DataFrame(injury_column)
# df_sorted['injuries'] = injury_column


columns = ["pitches","player_id","player_name","total_pitches",
           "spin_rate","velocity","effective_speed","release_extension"
           ,"k_percent","bb","bb_percent","release_pos_z","release_pos_x","arm_angle"] 

test_dataframe = pd.read_csv('savant_data.csv', usecols=columns)
test_dataframe = flip_names(test_dataframe)


test_sorted = test_dataframe.sort_values(by = "player_name")
response = call_api()
injuries = find_injuries(response)
sorted_injuries = sorted(injuries, key=get_name)
injury_column = create_total_injuries(test_sorted, sorted_injuries)
test_sorted['injuries'] = injury_column

test_sorted.to_csv("output.csv")


#print(df_sorted.head())
# print(injury_column)


# col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
# df = pd.read_csv("output.csv", skiprows=1, header=None, names=columns)
# x = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values.reshape(-1, 1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=41)
# classifier = DecisionTreeClassifier(min_sample_split=3, max_depth=3)
# classifier.fit(x_train, y_train)
# classifier.print_tree()

# y_pred = classifier.predict(x_test)
# print(accuracy_score(y_test, y_pred))

# df_sorted.to_csv('df_sorted')


# Random columns, from 4 to 15
# around 100 datasets
# Potentially drop columns with NaN values
# test_sorted.dropna()
random_tree = pd.DataFrame(columns=columns + ["injuries"])

# Choose row 2298 times
for _ in range(len(test_sorted)):
    row_index = randint(1, len(test_sorted)-1)
    # Add row to dataframe
    print(test_sorted.iloc[row_index])
    random_tree.loc[len(random_tree)] = test_sorted.iloc[row_index]

random_tree.to_csv("random_mid.csv")
# Randomly select features, cols 4 to 15

number_features = 9
total_features = []
all_columns = test_sorted.columns.to_list()
random_tree.drop(columns="pitches")
i = 0
while i < number_features:
    columns = random_tree.columns.to_list()
    feature_index = randint(4, len(columns)-1)
    if all_columns[feature_index] in columns:
        random_tree.drop(columns=all_columns[feature_index])
        i += 1
random_tree.to_csv("random.csv")


class RandomTree():
    def __init__(self, keep_columns, starting_df):
        """An implementation random decision tree"""
        self.random_tree_df = pd.DataFrame(columns=keep_columns)
        self.createRandomTree(starting_df)

    def chooseRows(self, df):
        """Pick random rows with replacement"""
        output = []
        for _ in range(len(df)):
            row_index = randint(1, len(df)-1)
            # Add row to dataframe
            output.append(row_index)
            # random_tree.loc[len(random_tree)] = df.iloc[row_index]
        return output

# Note: Make this with replacement
    def chooseFeatures(self, all_columns):
        """Pick random columns with replacement"""
        i = 0
        while i < number_features:
            columns = self.random_tree_df.columns.to_list()
            feature_index = randint(4, len(columns)-1)
            if all_columns[feature_index] in columns:
                self.random_tree_df.drop(columns=all_columns[feature_index])
                i += 1

    def createRandomTree(self, starting_df):
        """Create random decision trees for random forest classification"""

        selected_rows = self.chooseRows(starting_df)

        for row_index in selected_rows:
            self.random_tree_df.loc[len(self.random_tree_df)] = starting_df.iloc[row_index]

        all_columns = starting_df.columns.to_list()
        self.chooseFeatures(all_columns)
        self.random_tree_df.to_csv("potent.csv")
        


tree = RandomTree(columns+["injuries"], test_sorted)
# tree.createRandomTree(test_sorted)


# injury_column = []
    
    # for injured in injury_list:
    #     injured_players = injured['pitcher_name']
    #     injury_types = [inj['injury_type'] for inj in injury_list]
    #     players_list = df['player_name'].tolist()
    
    # i = 0
    # j = 0
    # injured_pitchers = [injury["pitcher_name"] for injury in injury_list]
    # # print(injury_list)
    # # print(len(players_list))
    # # print(len(injured_pitchers))
    # # print(set(injury_types))
    # # print(len(players_list))
    # while i < len(players_list):
    #     player = players_list[i]

    #     if player in injured_pitchers:
    #         injury_column.append(injury_types[j])
    #         i += 1
    #         j += 1
        
    #     # if j < len(injured_players) and player == injured_players[j]:
    #     #     print(injury_types[j])
    #     #     injury_column.append(injury_types[j])
    #     #     i += 1
    #     #     j += 1
            
    #     elif j < len(injured_players) and injured_players[j] not in players_list:
    #         j += 1
    #     else:
    #         injury_column.append('0')
    #         i += 1
    # print(len(injury_column))   
    # 