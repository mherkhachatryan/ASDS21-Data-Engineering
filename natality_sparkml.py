from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler

from pyspark.sql.types import StructType, StructField, FloatType

from pyspark.context import SparkContext

seed = 42
# %%
sc = SparkContext()
spark = SparkSession(sc)
schema = StructType([
    StructField("weight_pounds", FloatType(), True),
    StructField("mother_age", FloatType(), True),
    StructField("father_age", FloatType(), True),
    StructField("gestation_weeks", FloatType(), True),
    StructField("weight_gain_pounds", FloatType(), True),
    StructField("apgar_5min", FloatType(), True),
])
label = "weight_pounds"
# Read the data from BigQuery as a Spark Dataframe.
# %%

natality_data = spark.read.format("bigquery").option(
    "table", "natality_regression.regression_input").load()
# %%
# Create a view so that Spark SQL queries can be run against the data.
natality_data.createOrReplaceTempView("natality")

# As a precaution, run a query in Spark SQL to ensure no NULL values exist.
sql_query = """
SELECT *
from natality
where weight_pounds is not null
and mother_age is not null
and father_age is not null
and gestation_weeks is not null
"""
clean_data = spark.sql(sql_query)
# %%
assambler = VectorAssembler(inputCols=["mother_age", "father_age", "gestation_weeks",
                                       "weight_gain_pounds", "apgar_5min"],
                            outputCol='features')

clean_data_assamlbed = assambler.transform(clean_data)

clean_data_assamlbed = clean_data_assamlbed.select("features", label)
# %%
training_data, test_data = clean_data_assamlbed.randomSplit([0.8, 0.2], seed=seed)
# %%
training_data.cache()
# %%
# Construct a new LinearRegression object and fit the training data.
simple_lr_model = LinearRegression(featuresCol="features", labelCol=label, maxIter=5, regParam=0.2, solver="normal")
simple_lr_model = simple_lr_model.fit(training_data)
# %%
simple_tree = DecisionTreeRegressor(featuresCol="features", labelCol=label).fit(training_data)
# %%
prediction_lr = simple_lr_model.transform(test_data)
simple_lr_metrics = RegressionMetrics(prediction_lr.select(label, "prediction").rdd)
# %%
print(f"*Test* Simple LR R2: {simple_lr_metrics.r2}")
print(f"*Test* Simple LR RMSE: {simple_lr_metrics.rootMeanSquaredError}")
# %%
print("Lin reg params")
print("Coefficients:" + str(simple_lr_model.coefficients))
print("Intercept:" + str(simple_lr_model.intercept))
print("*Train* R2:" + str(simple_lr_model.summary.r2))
print("*Train* RMSE:" + str(simple_lr_model.summary.rootMeanSquaredError))
# %%
prediction_tree = simple_tree.transform(test_data)
simple_tree_metrics = RegressionMetrics(prediction_tree.select(label, "prediction").rdd)
# %%
print(f"*Test* Simple tree R2: {simple_tree_metrics.r2}")
print(f"*Test* Simple tree RMSE: {simple_tree_metrics.rootMeanSquaredError}")
# %%
print("Feature importance:\n", simple_tree.featureImportances)
# %%
# Feature engineering
new_data = clean_data
new_data = new_data.withColumn("parent_age_harmonic_mean",
                               2 * new_data.mother_age * new_data.father_age / (
                                       new_data.mother_age + new_data.father_age))
new_data = new_data.withColumn("gestation_days", new_data.gestation_weeks * 7)
new_data = new_data.withColumn("weight_gain_kgs", new_data.weight_gain_pounds / 2.205)
# %%
new_assambler = VectorAssembler(inputCols=["mother_age", "father_age", "gestation_weeks",
                                           "weight_gain_pounds", "apgar_5min", "parent_age_harmonic_mean",
                                           "gestation_days", "weight_gain_kgs"],
                                outputCol='features')
new_data_assamlbed = assambler.transform(clean_data)

new_data_assamlbed = new_data_assamlbed.select("features", label)
# %%
training_data_new, test_data_new = new_data_assamlbed.randomSplit([0.8, 0.2], seed=seed)
# %%
training_data_new.cache()
# %%
simple_lr_model_new = LinearRegression(featuresCol="features", labelCol=label, maxIter=5, regParam=0.2,
                                       solver="normal")
simple_lr_model_new = simple_lr_model_new.fit(training_data_new)
# %%
prediction_lr_new = simple_lr_model_new.transform(test_data_new)
simple_lr_metrics = RegressionMetrics(prediction_lr_new.select(label, "prediction").rdd)
# %%
print(f"*Test* Simple LR New R2: {simple_lr_metrics.r2}")
print(f"*Test* Simple LR New RMSE: {simple_lr_metrics.rootMeanSquaredError}")
# %%
print("New Lin reg params")
print("Coefficients:" + str(simple_lr_model_new.coefficients))
print("Intercept:" + str(simple_lr_model_new.intercept))
print("*Train* R2:" + str(simple_lr_model_new.summary.r2))
print("*Train* RMSE:" + str(simple_lr_model_new.summary.rootMeanSquaredError))
