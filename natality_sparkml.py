from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler

# def vector_from_inputs(r):
#     return (r["weight_pounds"], Vectors.dense(float(r["mother_age"]),
#                                               float(r["father_age"]),
#                                               float(r["gestation_weeks"]),
#                                               float(r["weight_gain_pounds"]),
#                                               float(r["apgar_5min"])))
#sc = SparkContext()
from pyspark.sql.types import StructType, StructField, FloatType

spark = SparkSession.builder.master("local[*]").appName("test").getOrCreate()
schema = StructType([
    StructField("mother_age", FloatType(), True),
    StructField("father_age", FloatType(), True),
    StructField("gestation_weeks", FloatType(), True),
StructField("weight_gain_pounds", FloatType(), True),
StructField("apgar_5min", FloatType(), True),
StructField("weight_pounds", FloatType(), True)])

assambler = VectorAssembler(inputCols=["mother_age", "father_age", "gestation_weeks",
                                       "weight_gain_pounds", "apgar_5min"],
                            outputCol='features')

# Read the data from BigQuery as a Spark Dataframe.
natality_data = spark.read.format("csv").option(
    "header", "true").schema(schema).load('natality_sample.csv')

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




clean_data.printSchema()
clean_data_assamlbed = assambler.transform(clean_data)

clean_data_assamlbed.show()
#training_data = clean_data.rdd.map(vector_from_inputs).toDF(["label",
                           #                                  "features"])
#training_data.head()
#training_data.cache()

# Construct a new LinearRegression object and fit the training data.
# lr = LinearRegression(maxIter=5, regParam=0.2, solver="normal")
# model = lr.fit(training_data)
#
#
# print("Coefficients:" + str(model.coefficients))
# print("Intercept:" + str(model.intercept))
# print("R^2:" + str(model.summary.r2))
# model.summary.residuals.show()

clean_data.show()
