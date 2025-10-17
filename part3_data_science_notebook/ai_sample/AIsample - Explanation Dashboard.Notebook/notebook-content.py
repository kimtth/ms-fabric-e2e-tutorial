# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   }
# META }

# MARKDOWN ********************

# # A guide to Explainable AI with SHapley Additive exPlanations 

# MARKDOWN ********************

# ## Introduction
# 
# This tutorial shows how to leverage SHapley Additive exPlanations (SHAP) to explain the output of machine learning models in Microsoft Fabric.
# 
# SHAP is a method used for interpreting machine learning models by attributing the contribution of each feature to the model's output for a specific data point. In this tutorial, you use Kernel SHAP to explain a tabular classification model built from the Adults Census dataset and then visualize the explanation in the ExplanationDashboard from [Responsible AI Widgets](https://github.com/microsoft/responsible-ai-widgets) in Microsoft Fabric.
# 
# This tutorial covers these topics:
# 
# 1. Install `raiwidgets` library
# 2. Load and process the data and train a binary classification model
# 3. Create a TabularSHAP explainer and extract SHAP values
# 4. Show how to visualize the explanation using the RAI ExplanationDashboard


# MARKDOWN ********************

# ## Step 1: Install custom library

# MARKDOWN ********************

# Prior to process the data and train a model, you need to install a custom library for which you will use the in-line installation capabilities (e.g., `pip`, `conda`, etc.) to quickly get started. Please note that this process will solely install the custom libraries within your notebook environment, and not in the workspace.
# 
# Additionally, please be aware that the PySpark kernel will automatically restart after executing the `%pip install` command. Therefore, it is crucial to install the desired library prior to running any other cells within your notebook.
# 
# You'll use `%pip install` to install the `raiwidgets` library. You can follow instructions available at [Package management - Azure Synapse Analytics | Microsoft Docs](https://docs.microsoft.com/en-us/azure/synapse-analytics/spark/apache-spark-azure-portal-add-libraries) for further information about how to install ["raiwidgets"](https://pypi.org/project/raiwidgets/) and ["interpret-community"](https://pypi.org/project/interpret-community/) packages.

# CELL ********************

%pip install raiwidgets itsdangerous==2.0.1 interpret-community

# MARKDOWN ********************

# You also need to import the required libraries from [PySpark](https://spark.apache.org/docs/latest/api/python/index.html) and [SynapseML](https://microsoft.github.io/SynapseML/) and define some User Defined Functions (UDFs) that you will need later.

# CELL ********************

from IPython.terminal.interactiveshell import TerminalInteractiveShell
from synapse.ml.explainers import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import *
import pandas as pd

vec_access = udf(lambda v, i: float(v[i]), FloatType())
vec2array = udf(lambda vec: vec.toArray().tolist(), ArrayType(FloatType()))

# MARKDOWN ********************

# To disable Microsoft Fabric autologging in a notebook session, call `mlflow.autolog()` and set `disable=True`.

# CELL ********************

# Set up MLflow for experiment tracking
import mlflow

mlflow.autolog(disable=True)  # Disable MLflow autologging

# MARKDOWN ********************

# ## Step 2: Load the data and train the model

# MARKDOWN ********************

# For this tutorial, you will use the [Adult Census Income dataset](https://archive.ics.uci.edu/ml/datasets/Adult). The dataset contains 32,561 rows and 14 columns/features.
# 
# Download a publicly available version of the dataset from the blog storage and load the data as a spark DataFrame.

# CELL ********************

df = spark.read.parquet(
    "wasbs://publicwasb@mmlspark.blob.core.windows.net/AdultCensusIncome.parquet"
).cache()

labelIndexer = StringIndexer(
    inputCol="income", outputCol="label", stringOrderType="alphabetAsc"
).fit(df)
print("Label index assigment: " + str(set(zip(labelIndexer.labels, [0, 1]))))

# MARKDOWN ********************

# Next step is to pre-process the data (indexing categorical features and one-hot encoding them) and train a Logistic Regression model to predict the `income` label (1 or 0) based on the input features.

# CELL ********************

training = labelIndexer.transform(df)
display(training)
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
categorical_features_idx = [col + "_idx" for col in categorical_features]
categorical_features_enc = [col + "_enc" for col in categorical_features]
numeric_features = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
# Convert the categorical features into numerical indices
strIndexer = StringIndexer(
    inputCols=categorical_features, outputCols=categorical_features_idx
)
# Perform one-hot encoding
onehotEnc = OneHotEncoder(
    inputCols=categorical_features_idx, outputCols=categorical_features_enc
)
# Create a VectorAssembler to assemble all the one-hot encoded categorical features and numerical features into a single feature vector
vectAssem = VectorAssembler(
    inputCols=categorical_features_enc + numeric_features, outputCol="features"
)
# Train a Logistic Regression model
lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="fnlwgt")
pipeline = Pipeline(stages=[strIndexer, onehotEnc, vectAssem, lr])
model = pipeline.fit(training)

# MARKDOWN ********************

# After the model is trained, you randomly select some observations to be explained.

# CELL ********************

explain_instances = (
    model.transform(training).orderBy(rand()).limit(5).repartition(200).cache()
)
display(explain_instances)

# MARKDOWN ********************

# ## Step 3: Create a TabularSHAP Explainer and extract SHAP Values

# MARKDOWN ********************

# You should create a TabularSHAP explainer by configuring it with the following parameters: set the input columns to include all the features that the model uses, specify the model itself, and indicate the target output column you intend to explain.
# 
# In this particular scenario, your goal is to elucidate the `probability` output, which is represented as a vector with a length of 2. Your specific focus, however, is on class 1 probability. To simultaneously explain both class 0 and class 1 probabilities, you must define the `targetClasses` parameter as `[0, 1]`.
# 
# To serve as background data for the Kernel SHAP explanation method, it's recommended to randomly sample 100 rows from the training dataset. This sampled data will be used to integrate out the effects of individual features when calculating the SHAP values.

# CELL ********************

# Compute SHAP values for the trained model
shap = TabularSHAP(
    inputCols=categorical_features + numeric_features,
    outputCol="shapValues",
    numSamples=5000,
    model=model,
    targetCol="probability",
    targetClasses=[1],
    backgroundData=broadcast(training.orderBy(rand()).limit(100).cache()),
)

shap_df = shap.transform(explain_instances)

# MARKDOWN ********************

# Note that `inputCols` specifies the list of input features that you want to explain which in this case combines both the categorical and the numeric features. The `outputCol` specifies the name of the output column where SHAP values will be stored in the resulting DataFrame.
# 
# `targetCol` is used to specify the name of the target column where the model's output (probability scores) is stored and `targetClasses` indicates the class's output (e.g., 1 in this case) that is being explained (meaning you are explaining predictions for class 1).
# 
# Once you have the resulting DataFrame that contain the SHAP values, you can extract the class 1 probability of the model output, the SHAP values for the target class, the original features, and the true label. Then you convert it to a pandas DataFrame for visualization.
# 
# For each observation, the first element in the SHAP values vector is the base value (the mean output of the background dataset), and each of the following element is the SHAP values for each feature.

# CELL ********************

# Choose following columns from the DataFrame
# "shapValues": The modified array of SHAP values
# "probability": The extracted class 1 probability
# "label": A column assumed to contain labels or target values
shaps = (
    shap_df.withColumn("probability", vec_access(col("probability"), lit(1)))
    .withColumn("shapValues", vec2array(col("shapValues").getItem(0)))
    .select(
        ["shapValues", "probability", "label"] + categorical_features + numeric_features
    )
)

shaps_local = shaps.toPandas()
shaps_local.sort_values("probability", ascending=False, inplace=True, ignore_index=True) # Arrange with the highest probabilities at the top
pd.set_option("display.max_colwidth", None)
shaps_local

# MARKDOWN ********************

# ## Step 4: Visualize the explanation using the RAI ExplanationDashboard


# MARKDOWN ********************

# You can visualize the explanation in [interpret-community format](https://github.com/interpretml/interpret-community) in the [ExplanationDashboard](https://github.com/microsoft/responsible-ai-widgets/).

# CELL ********************

import numpy as np

features = categorical_features + numeric_features
features_with_base = ["Base"] + features

rows = shaps_local.shape[0]

local_importance_values = shaps_local[["shapValues"]] # Extract the "shapValues" column from the "shaps_local" DataFrame
eval_data = shaps_local[features]
true_y = np.array(shaps_local[["label"]])

# MARKDOWN ********************

# Process the SHAP values stored to separate the bias values (likely representing the base prediction) and the actual importance values for each data point and class. 

# CELL ********************

list_local_importance_values = local_importance_values.values.tolist()
converted_importance_values = []
bias = []
for classarray in list_local_importance_values:
    for rowarray in classarray:
        converted_list = rowarray.tolist()
        # The bias values are stored in the bias list
        bias.append(converted_list[0])
        # Remove the bias from local importance values
        del converted_list[0]
        # Importance values are stored in the converted_importance_values list
        converted_importance_values.append(converted_list)

# MARKDOWN ********************

# Create a global explanation that is based on feature importance values (SHAP values), evaluation data, and expected values (bias terms).

# CELL ********************

from interpret_community.adapter import ExplanationAdapter

adapter = ExplanationAdapter(features, classification=True) # List of features used in the explanation
# eval_data is the dataset used to train or test the machine learning model
global_explanation = adapter.create_global(
    converted_importance_values, eval_data, expected_values=bias
)

# MARKDOWN ********************

# View the global importance values.


# CELL ********************

global_explanation.global_importance_values

# MARKDOWN ********************

# View the local importance values.

# CELL ********************

global_explanation.local_importance_values

# CELL ********************

class wrapper(object):
    def __init__(self, model):
        self.model = model

    def predict(self, data):
        sparkdata = spark.createDataFrame(data)
        return (
            model.transform(sparkdata)
            .select("prediction")
            .toPandas()
            .values.flatten()
            .tolist()
        )

    def predict_proba(self, data):
        sparkdata = spark.createDataFrame(data)
        prediction = (
            model.transform(sparkdata)
            .select("probability")
            .toPandas()
            .values.flatten()
            .tolist()
        )
        proba_list = [vector.values.tolist() for vector in prediction]
        return proba_list

# MARKDOWN ********************

# The following shows how the final results using the kernel SHAP will look like. You can select the feature of your interest, choose the chart type, etc. to gain valuable insights about the impact of different features.


# CELL ********************

# View the explanation in the ExplanationDashboard
from raiwidgets import ExplanationDashboard

ExplanationDashboard(
    global_explanation, wrapper(model), dataset=eval_data, true_y=true_y
)

# MARKDOWN ********************

# ## Summary of the learnings
# 
# In summary, in this tutorial you have learned how to leverage kernel SHAP to provide a holistic and actionable understanding of ML models by quantifying feature importance, promoting model transparency, and facilitating model improvement and debugging. 
# 
# Kernel SHAP is a technique that helps explain the predictions of complex models by attributing the contribution of each feature to the model's output. It uses a kernel-based approach to estimate feature importance, providing insights into how different input variables influence the model's decisions. This interpretability tool aids in understanding and debugging machine learning models, making them more transparent and trustworthy.
# 
# Through the practical illustrations presented above, you've acquired the skills to effectively utilize Kernel SHAP, ensuring the reliability and alignment of machine learning models with their intended goals.
