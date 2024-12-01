import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold   
from statistics import mean
import joblib
import os 

raw_data = pd.read_csv(r'D:\pokemon_combined.csv')

# 3.1 Quick view of the data
print('\n____________ Dataset info ____________')
print(raw_data.info())              
print('\n____________ Some first data examples ____________')
print(raw_data.head(10)) 

# Focus on combat stats to analyze the Catch rate
combat_stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

print('\n____________ Counts on Catch rate ____________')
print(raw_data['Catch rate'].value_counts()) 

print('\n____________ Statistics of combat stats ____________')
print(raw_data[combat_stats].describe())    

print('\n____________ Get specific rows and cols (Catch rate and Combat Stats) ____________')     
print(raw_data.iloc[[0, 5, 48], raw_data.columns.get_indexer(['Catch rate'] + combat_stats)])

# 3.2 Scatter plot between Catch rate and Combat Stats

# Create the directory in D: if it doesn't exist
if not os.path.exists(r'D:\pokemon_figures'):
    os.makedirs(r'D:\pokemon_figures')

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="HP", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs HP')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_hp.png', format='png', dpi=300)
    plt.show()

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="Attack", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs Attack')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_attack.png', format='png', dpi=300)
    plt.show()

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="Defense", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs Defense')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_defense.png', format='png', dpi=300)
    plt.show()

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="Sp. Atk", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs Sp. Atk')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_sp_atk.png', format='png', dpi=300)
    plt.show()

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="Sp. Def", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs Sp. Def')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_sp_def.png', format='png', dpi=300)
    plt.show()

if 1:  # Change to 1 to activate the plot
    raw_data.plot(kind="scatter", y="Catch rate", x="Speed", alpha=0.2)
    plt.title('Scatter Plot: Catch Rate vs Speed')
    plt.savefig(r'D:\pokemon_figures\scatter_catchrate_speed.png', format='png', dpi=300)
    plt.show()

# 3.3 Scatter plot between every pair of features (Catch rate and Combat Stats)
if 1:  # Change to 1 to enable the plot
    from pandas.plotting import scatter_matrix
    features_to_plot = ["Catch rate", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8))  # Histograms on the main diagonal
    plt.savefig(r'D:\pokemon_figures\scatter_mat_combat_stats.png', format='png', dpi=300)
    plt.show()
    
    

# 3.4 Scatter matrix for 1 feature (histogram will be shown on the diagonal)
if 1:  # Change to 1 to activate the plot
    from pandas.plotting import scatter_matrix
    features_to_plot = ["Catch rate"]  # Single feature

    scatter_matrix(raw_data[features_to_plot], figsize=(8, 6), diagonal='hist')
    plt.savefig(r'D:\pokemon_figures\scatter_matrix_catchrate_hist.png', format='png', dpi=300)
    plt.show()


# 3.5 Plot histograms of numeric features
if 1:  # Change to 1 to activate the plot
    # Plot histograms for all numeric features
    raw_data.hist(figsize=(12, 8), bins=10)  # bins: number of intervals (10 by default)
    
    # Adjust label sizes
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure to D drive
    plt.savefig(r'D:\pokemon_figures\hist_numeric_features.png', format='png', dpi=300)  # Save before showing
    plt.show()


print("\n")

# 3.6 Compute correlations between features
corr_matrix = raw_data.corr(numeric_only=True)

# Print correlation between Catch rate and other features (combat stats)
print("Correlation of 'Catch rate' with other features:")
print(corr_matrix["Catch rate"].sort_values(ascending=False))  # Correlation of Catch rate with other features

print("\n")

# Plot heatmap for the full correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Combat Stats and Catch Rate')
plt.show()

print("\n")

# 3.7 Try combining features
raw_data["Attack + Defense"] = raw_data["Attack"] + raw_data["Defense"]  # Combining Attack and Defense
raw_data["Sp. Atk + Sp. Def"] = raw_data["Sp. Atk"] + raw_data["Sp. Def"]  # Combining Special Attack and Special Defense


# Compute the correlation matrix
corr_matrix = raw_data.corr(numeric_only=True)

# Print correlation between Catch rate and other features
print(corr_matrix["Catch rate"].sort_values(ascending=False))  # Correlation with Catch rate

# Remove the experimental columns
raw_data.drop(columns=["Attack + Defense", "Sp. Atk + Sp. Def"], inplace=True)  # Remove combined features

# 3.8: Remove rare Catch rate values (Preprocessing step) !!!!!!!!!!!!!!!!!!!!!!
catch_rate_counts = raw_data["Catch rate"].value_counts()
rare_catch_rates = catch_rate_counts[catch_rate_counts == 1].index

# Remove rows with rare Catch rate values
raw_data = raw_data[~raw_data["Catch rate"].isin(rare_catch_rates)].copy()  # Using .copy() to avoid SettingWithCopyWarning

# 4.1: Remove unused features
raw_data.drop(columns=["Name", "Species", "Base Friendship", "Base Exp.", "Growth Rate", "Gender", "Height", "Weight",'Abilities','Type'], inplace=True)

# Calculate Z-scores for each combat stat and catch rate (for outlier removal) !!!!!!!!!!!!!!!!!!!!
from scipy import stats
z_scores = stats.zscore(raw_data[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Catch rate']])

# Keep rows where the Z-score is less than 3 (i.e., remove outliers)
raw_data = raw_data[(abs(z_scores) < 3).all(axis=1)].copy()  # Using .copy() to avoid SettingWithCopyWarning

# Check the info of the cleaned data
print("\n")
raw_data.info()


# 4.2: Split training-test set and NEVER touch test set until test phase
method = 2  # You can change to method = 2 if stratified sampling is desired

if method == 1:  # Method 1: Randomly select 20% of data for test set
    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(raw_data, test_size=0.2, random_state=42) 
    # random_state ensures that the split is reproducible
    
elif method == 2:  # Method 2: Stratified sampling with Catch rate bins
    # Create new feature "CATCH RANGE" to bin catch rates into ranges
    raw_data["CATCH RANGE"] = pd.cut(raw_data["Catch rate"],
                                     bins=[0, 50, 100, 150, np.inf],
                                     labels=["low", "medium", "high", "very high"],  # Ordered from low to very high
                                     ordered=True)  # Ensure the order is preserved
    
    # Create training and test set using stratified sampling by "CATCH RANGE"
    from sklearn.model_selection import StratifiedShuffleSplit

    # Define the splitter for one train-test split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Apply stratified sampling based on "CATCH RANGE"
    for train_index, test_index in splitter.split(raw_data, raw_data["CATCH RANGE"]): 
        train_set = raw_data.iloc[train_index]
        test_set = raw_data.iloc[test_index]
    
    # See if the stratification worked as expected
    if 1:
        # Plot histograms for "CATCH RANGE"
        plt.figure(figsize=(10, 5))

        # Plot histogram for raw data
        plt.subplot(1, 2, 1)
        raw_data["CATCH RANGE"].value_counts(sort=False).plot(kind='bar')
        plt.title('Raw Data Catch Rate Distribution')
        plt.xlabel('Catch Rate Range')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)  # Keep labels horizontal
        plt.grid(True)

        # Plot histogram for the training set
        plt.subplot(1, 2, 2)
        train_set["CATCH RANGE"].value_counts(sort=False).plot(kind='bar')
        plt.title('Training Set Catch Rate Distribution')
        plt.xlabel('Catch Rate Range')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)  # Keep labels horizontal
        plt.grid(True)

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    # Remove the new feature "CATCH RANGE" after the split
    for _set_ in (train_set, test_set):
        _set_ = _set_.copy()  # Create a copy to avoid SettingWithCopyWarning
        _set_.drop(columns="CATCH RANGE", inplace=True)  # Drop "CATCH RANGE" from train and test sets
        
# Print split information
print('\n____________ Split training and test set ____________')     
print(len(train_set), "training +", len(test_set), "test examples")
print(train_set.head(10))

# 4.3 Separate labels from data, since we do not process label values
train_set_labels = train_set["Catch rate"].copy()  # Copy the labels from the training set
train_set = train_set.drop(columns = "Catch rate")  # Drop the label from the training set

test_set_labels = test_set["Catch rate"].copy()  # Copy the labels from the test set
test_set = test_set.drop(columns = "Catch rate")  # Drop the label from the test set


# 4.4 Define pipelines for processing data. 
# 4.4.1 Define ColumnSelector: a transformer for choosing columns
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, dataframe, labels=None):
        return self
    def transform(self, dataframe):
        return dataframe[self.feature_names].values  # Extract selected columns as NumPy array

# Define numerical feature names (combat stats)
num_feat_names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

# Define categorical feature names (if any categorical features exist in your dataset)
# 4.4.2 Define categorical feature names
cat_feat_names = []  # No categorical features to be processed

# 4.4.2 Pipeline for categorical features (This won't process anything if cat_feat_names is empty)
cat_pipeline = Pipeline([
    ('selector', ColumnSelector(cat_feat_names)),  # No categorical features selected
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="constant", fill_value="NO INFO", copy=True)),  # Handle missing values (not necessary if no categorical data)
    ('cat_encoder', OneHotEncoder())  # OneHotEncoder is not necessary if cat_feat_names is empty
])

# 4.4.3 Define MyFeatureAdder: a transformer for adding features (e.g., "TOTAL COMBAT POWER")
class MyFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_total_combat_power=True):  # Custom argument to control feature addition
        self.add_total_combat_power = add_total_combat_power

    def fit(self, feature_values, labels=None):
        return self  # Nothing to fit here

    def transform(self, feature_values, labels=None):
        if self.add_total_combat_power:
            # Define column indices for the combat stats in num_feat_names
            HP_id, Attack_id, Defense_id, Sp_Atk_id, Sp_Def_id, Speed_id = 0, 1, 2, 3, 4, 5  # Assuming correct indices

            # Compute total combat power as the sum of all combat stats
            total_combat_power = (feature_values[:, HP_id] + feature_values[:, Attack_id] +
                                  feature_values[:, Defense_id] + feature_values[:, Sp_Atk_id] +
                                  feature_values[:, Sp_Def_id] + feature_values[:, Speed_id])

            # Concatenate the new feature (total_combat_power) to the original feature set
            feature_values = np.c_[feature_values, total_combat_power]  # Concatenate arrays
        return feature_values


# 4.4.4 Pipeline for numerical features
num_pipeline = Pipeline([
    ('selector', ColumnSelector(num_feat_names)),  # Select the numerical combat stats
    ('imputer', SimpleImputer(missing_values=np.nan, strategy="median", copy=True)),  # Handle missing values
    ('attribs_adder', MyFeatureAdder(add_total_combat_power=True)),  # Add total combat power feature
    ('std_scaler', StandardScaler(with_mean=True, with_std=True, copy=True))  # Scale features to zero mean and unit variance
])

# Since you don't have categorical features, there's no need for a categorical pipeline
# If needed, you could leave an empty list for future categorical features
cat_feat_names = []  # No categorical features to be processed

# 4.4.5 Combine features transformed by two above pipelines

full_pipeline = Pipeline([
    ("num_pipeline", num_pipeline)  # Only use the numerical pipeline since there are no categorical features
])

# Define the directory where you want to save the pipeline on D drive
save_directory = r'D:\models'

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# 4.5 Run the pipeline to process training data           
processed_train_set_val = full_pipeline.fit_transform(train_set)

print('\n____________ Processed feature values ____________')
# Check if the result is sparse, convert to dense if needed
if hasattr(processed_train_set_val, 'toarray'):
    print(processed_train_set_val[[0, 1, 2]].toarray())
else:
    print(processed_train_set_val[[0, 1, 2]])

# Output the shape of the processed training set
print(processed_train_set_val.shape)

# Print information about the number of features
print('We have %d numeric features (combat stats) + 1 added feature (total combat power).' 
      % (len(num_feat_names)))

# Save the full pipeline for later use
joblib.dump(full_pipeline, os.path.join(save_directory, 'full_pipeline.pkl'))

# (optional) Add header to create dataframe. Just to see. We don't need header to run algorithms
if 10: 
    # Check if "Total Combat Power" is added by comparing the shape
        columns_header = num_feat_names + ["Total Combat Power"]

    # Convert processed data into a DataFrame with the headers
processed_train_set = pd.DataFrame(processed_train_set_val, columns=columns_header)

    # Print the processed DataFrame to check the results
print('\n____________ Processed dataframe ____________')
print(processed_train_set.info())
print('\n____________ Processed data ____________')
print(processed_train_set.head(10))

