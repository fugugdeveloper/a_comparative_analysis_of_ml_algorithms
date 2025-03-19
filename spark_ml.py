from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, ChiSqSelector
from pyspark.ml.classification import RandomForestClassifier, LinearSVC, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F

# Initialize Spark session
spark = SparkSession.builder.appName("ML_Pipeline").getOrCreate()

# Load the dataset using PySpark
file_path = 'cloned_repos/ghtorrent-2019-01-07.csv/ghtorrent-2019-01-07.csv'
df = spark.read.csv(file_path, header=True, inferSchema=True)

# Feature and target selection
label_col = 'Label'  # Assuming 'Label' is the target column
features_cols = [col for col in df.columns if col != label_col]

# Handling missing values
df = df.na.fill({c: 0 for c in df.columns})  # Replace missing values with 0 (modify based on requirements)

# Convert string columns to numerical (if any)
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in df.columns if df.schema[column].dataType == 'StringType']
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# Assembling features into a single vector column
assembler = VectorAssembler(inputCols=[c+"_index" if c in indexers else c for c in features_cols], outputCol="features")
df = assembler.transform(df)

# Feature selection (optional, equivalent to SelectKBest)
selector = ChiSqSelector(numTopFeatures=20, featuresCol="features", outputCol="selectedFeatures", labelCol=label_col)
df = selector.fit(df).transform(df)

# Data splitting
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Addressing data imbalance with SMOTE (PySpark doesn't have native SMOTE, consider stratified sampling or using an external library if needed)
# Skipping SMOTE implementation here as PySpark doesn't natively support it.

# Standardization
scaler = StandardScaler(inputCol="selectedFeatures", outputCol="scaledFeatures", withStd=True, withMean=False)
train_df = scaler.fit(train_df).transform(train_df)
test_df = scaler.fit(test_df).transform(test_df)

# Model training and hyperparameter tuning
models = {
    'RandomForest': RandomForestClassifier(labelCol=label_col, featuresCol='scaledFeatures', seed=42),
    'SVM': LinearSVC(labelCol=label_col, featuresCol='scaledFeatures', maxIter=10, regParam=0.1),
    'NeuralNetwork': MultilayerPerceptronClassifier(labelCol=label_col, featuresCol='scaledFeatures', maxIter=300, layers=[20, 10, 5, 2])
}

# Example for RandomForest hyperparameter tuning
paramGrid_rf = (ParamGridBuilder()
                .addGrid(models['RandomForest'].numTrees, [100, 200, 300])
                .addGrid(models['RandomForest'].maxDepth, [10, 20, 30])
                .build())

crossval_rf = CrossValidator(estimator=models['RandomForest'],
                             estimatorParamMaps=paramGrid_rf,
                             evaluator=MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy"),
                             numFolds=5)

cv_model_rf = crossval_rf.fit(train_df)
best_rf = cv_model_rf.bestModel

# Evaluate model
evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
accuracy_rf = evaluator.evaluate(best_rf.transform(test_df))
print(f"Random Forest Accuracy: {accuracy_rf}")

# You would repeat a similar process for SVM and Neural Network models
# Example SVM tuning
paramGrid_svm = ParamGridBuilder().addGrid(models['SVM'].regParam, [0.01, 0.1, 1.0]).build()
crossval_svm = CrossValidator(estimator=models['SVM'],
                              estimatorParamMaps=paramGrid_svm,
                              evaluator=MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy"),
                              numFolds=5)
cv_model_svm = crossval_svm.fit(train_df)
best_svm = cv_model_svm.bestModel
accuracy_svm = evaluator.evaluate(best_svm.transform(test_df))
print(f"SVM Accuracy: {accuracy_svm}")

# Example Neural Network tuning
paramGrid_nn = ParamGridBuilder().addGrid(models['NeuralNetwork'].layers, [[20, 10, 5, 2], [20, 15, 10, 2]]).build()
crossval_nn = CrossValidator(estimator=models['NeuralNetwork'],
                             estimatorParamMaps=paramGrid_nn,
                             evaluator=MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy"),
                             numFolds=5)
cv_model_nn = crossval_nn.fit(train_df)
best_nn = cv_model_nn.bestModel
accuracy_nn = evaluator.evaluate(best_nn.transform(test_df))
print(f"Neural Network Accuracy: {accuracy_nn}")

# Model evaluation function (custom implementation as needed for each model)

def evaluate_model(model, test_df):
    predictions = model.transform(test_df)
    accuracy = evaluator.evaluate(predictions)
    print(f"Accuracy: {accuracy}")
    predictions.groupBy('prediction').count().show()  # Example of showing class distribution in predictions

# Evaluate all models
print("Random Forest Evaluation:")
evaluate_model(best_rf, test_df)

print("SVM Evaluation:")
evaluate_model(best_svm, test_df)

print("Neural Network Evaluation:")
evaluate_model(best_nn, test_df)

# Stop the Spark session
spark.stop()
