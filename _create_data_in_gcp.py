from google.cloud import bigquery

# Create a new Google BigQuery client using Google Cloud Platform project
# defaults.
client = bigquery.Client()
#%%
# Prepare a reference to a new dataset for storing the query results.
dataset_id = "natality_regression"
dataset_id_full = f"{client.project}.{dataset_id}"

dataset = bigquery.Dataset(dataset_id_full)

# Create the new BigQuery dataset.
dataset = client.create_dataset(dataset)

# Configure the query job.
job_config = bigquery.QueryJobConfig()

# Set the destination table to where you want to store query results.

job_config.destination = f"{dataset_id_full}.regression_input"
#%%

query = """
    SELECT
        weight_pounds, mother_age, father_age, gestation_weeks,
        weight_gain_pounds, apgar_5min
    FROM
        `bigquery-public-data.samples.natality`
    WHERE
        weight_pounds IS NOT NULL
        AND mother_age IS NOT NULL
        AND father_age IS NOT NULL
        AND gestation_weeks IS NOT NULL
        AND weight_gain_pounds IS NOT NULL
        AND apgar_5min IS NOT NULL
"""
#%%
# Run the query.
query_job = client.query(query, job_config=job_config)
query_job.result()