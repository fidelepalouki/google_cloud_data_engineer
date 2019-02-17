# Cloud

## Def: Compute & Storage (Compute Engine & Cloud Storage)

## Options
- Compute Engine
- Cloud SQL, Cloud Spanner, Cloud Datastore, Cloud BigTable, Cloud Storage, Cloud Memorystore : Databases & Stores
- Cloud Dataproc: Hadoop, Spark, Hive, MapReduce (Apache products)
- BigQuery: Fast queries of peta bytes of data
- Cloud Pub/Sub: messaging architechture
- Cloud Dataflow: a way to excecute code that processes streamed & batched data

| Foundation     | Database        | Analytics an ML          | Data-handling frameworks |
| -------------- | --------------- | ------------------------ | ------------------------ |
| Compute Engine | Cloud Datastore | BigQuery                 | Cloud Pub/Sub            |
| Cloud Storage  | Cloud SQL       | Cloud Datalab            | Cloud Dataflow           |
|                | Cloud BigTable  | ML APIs(NLP, Vision ...) | Cloud Dataproc           |
|                | Cloud Spanner   |                          |                          |

## Advantages
- Spend less on ops and administration
- Incorporate real-time data into apps and architectures
- Apply Machine Learning broadly and easily
- Become a truly data-driven company

No ops: minimize the system administration overhead
Preemptible machines => 80% discount

Custom/changeable machine types, preemptible machines, and automatic discount

## Compute engine SSH:
- Create into the console || gcloud instances create
- cat /proc/cpuinfo
- sudo apt-get update
- sudo apt-get -y -qq install git
- git --version

## Storage
cp, mv, ls, rm, 
gsutil cp sales*.csv gs://acme-sales/data/
mb, rb, rsync => make, remove a bucket
ACL: Access Control List


## Compute
- App Engine
- App Engine Flex
- Container Engine
- Compute Engine

## Cloud SQL
### GCloud Shell
- gcloud auth list
- gcloud config list project
- Table creation


USE recommendation_spark;

DROP TABLE IF EXISTS Recommendation;
DROP TABLE IF EXISTS Rating;
DROP TABLE IF EXISTS Accommodation;

CREATE TABLE IF NOT EXISTS Accommodation
(
  id varchar(255),
  title varchar(255),
  location varchar(255),
  price int,
  rooms int,
  rating float,
  type varchar(255),
  PRIMARY KEY (ID)
);
);

CREATE TABLE  IF NOT EXISTS Rating
(
  userId varchar(255),
  accoId varchar(255),
  rating int,
  PRIMARY KEY(accoId, userId),
  FOREIGN KEY (accoId)
    REFERENCES Accommodation(id)
);

CREATE TABLE  IF NOT EXISTS Recommendation
(
  userId varchar(255),
  accoId varchar(255),
  prediction float,
  PRIMARY KEY(userId, accoId),
  FOREIGN KEY (accoId)
    REFERENCES Accommodation(id)
);

### In the sql database connection
select * from Accommodation where type = 'castle' and price < 1500;

## Managed Hadoop in the cloud
- Hadoop
- Pig
- Hive
- Spark

## Kubernetes best practices
- gcloud container clusters create myCluster
- gcloud builds submit --tag gcr.io/[PROJECT_ID]/quickstart-image .
- or
- gcloud builds submit --config cloudbuild.yaml .

## Datalab
- gcloud compute zones list
- datalab create my-datalab-vm --machine-type n1-highmen-8 --zone us-central1-a

import google.datalab.bigquery as bq
import pandas as pd
import numpy as np
import shutil

%bq tables describe --name bigquery-public-data.new_york.tlc_yellow_trips_2015

## 

%bq query
SELECT 
  EXTRACT (DAYOFYEAR from pickup_datetime) AS daynumber
FROM `bigquery-public-data.new_york.tlc_yellow_trips_2015` 
LIMIT 5

%bq query -n taxiquery
WITH trips AS (
  SELECT EXTRACT (DAYOFYEAR from pickup_datetime) AS daynumber 
  FROM `bigquery-public-data.new_york.tlc_yellow_trips_*`
  where _TABLE_SUFFIX = @YEAR
)
SELECT daynumber, COUNT(1) AS numtrips FROM trips
GROUP BY daynumber ORDER BY daynumber

query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
trips[:5]

## Weather

%bq query
SELECT * FROM `bigquery-public-data.noaa_gsod.stations`
WHERE state = 'NY' AND wban != '99999' AND name LIKE '%LA GUARDIA%'

%bq query -n wxquery
SELECT EXTRACT (DAYOFYEAR FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP)) AS daynumber,
       MIN(EXTRACT (DAYOFWEEK FROM CAST(CONCAT(@YEAR,'-',mo,'-',da) AS TIMESTAMP))) dayofweek,
       MIN(min) mintemp, MAX(max) maxtemp, MAX(IF(prcp=99.99,0,prcp)) rain
FROM `bigquery-public-data.noaa_gsod.gsod*`
WHERE stn='725030' AND _TABLE_SUFFIX = @YEAR
GROUP BY 1 ORDER BY daynumber DESC

query_parameters = [
  {
    'name': 'YEAR',
    'parameterType': {'type': 'STRING'},
    'parameterValue': {'value': 2015}
  }
]
weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
weather[:5]

## Merge datasets with Pandas

data = pd.merge(weather, trips, on='daynumber')
data[:5]

## Plots

j = data.plot(kind='scatter', x='maxtemp', y='numtrips')

j = data.plot(kind='scatter', x='dayofweek', y='numtrips')

j = data[data['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')

## Adding 2014 and 2016 data

data2 = data # 2015 data
for year in [2014, 2016]:
    query_parameters = [
      {
        'name': 'YEAR',
        'parameterType': {'type': 'STRING'},
        'parameterValue': {'value': year}
      }
    ]
    weather = wxquery.execute(query_params=query_parameters).result().to_dataframe()
    trips = taxiquery.execute(query_params=query_parameters).result().to_dataframe()
    data_for_year = pd.merge(weather, trips, on='daynumber')
    data2 = pd.concat([data2, data_for_year])
data2.describe()

j = data2[data2['dayofweek'] == 7].plot(kind='scatter', x='maxtemp', y='numtrips')

## Machine Learning with Tensorflow

### Linear regression

SCALE_NUM_TRIPS = 600000.0
trainsize = int(len(shuffled['numtrips']) * 0.8)
testsize = len(shuffled['numtrips']) - trainsize
npredictors = len(predictors.columns)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model_linear', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                             feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print("starting to train ... this will take a while ... use verbosity=INFO to get more verbose output")
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean(np.power((targets[trainsize:].values - pred), 2)))
print('LinearRegression has RMSE of {0}'.format(rmse))

### Neural network

SCALE_NUM_TRIPS = 600000.0
trainsize = int(len(shuffled['numtrips']) * 0.8)
testsize = len(shuffled['numtrips']) - trainsize
npredictors = len(predictors.columns)
noutputs = 1
tf.logging.set_verbosity(tf.logging.WARN) # change to INFO to get output every 100 steps ...
shutil.rmtree('./trained_model', ignore_errors=True) # so that we don't load weights from previous runs
estimator = tf.contrib.learn.DNNRegressor(model_dir='./trained_model',
                                          hidden_units=[5, 5],                             
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(predictors.values))

print("starting to train ... this will take a while ... use verbosity=INFO to get more verbose output")
def input_fn(features, targets):
  return tf.constant(features.values), tf.constant(targets.values.reshape(len(targets), noutputs)/SCALE_NUM_TRIPS)
estimator.fit(input_fn=lambda: input_fn(predictors[:trainsize], targets[:trainsize]), steps=10000)

pred = np.multiply(list(estimator.predict(predictors[trainsize:].values)), SCALE_NUM_TRIPS )
rmse = np.sqrt(np.mean((targets[trainsize:].values - pred)**2))
print('Neural Network Regression has RMSE of {0}'.format(rmse))

### Prediction

input = pd.DataFrame.from_dict(data = 
                               {'dayofweek' : [4, 5, 6],
                                'mintemp' : [60, 40, 50],
                                'maxtemp' : [70, 90, 60],
                                'rain' : [0, 0.5, 0]})
##### read trained model from ./trained_model
estimator = tf.contrib.learn.LinearRegressor(model_dir='./trained_model_linear',
                                          feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(input.values))

pred = np.multiply(list(estimator.predict(input.values)), SCALE_NUM_TRIPS )
print(pred)


## Machine Learning APIs

- Vison API
- Video Intelligence API
- Natural Language Processing API
- Speech to Text API
- Text to Speech API
- Translation API

### Translate API

# running Translate API
from googleapiclient.discovery import build
service = build('translate', 'v2', developerKey=APIKEY)

##### use the service
inputs = ['is it really this easy?', 'amazing technology', 'wow']
outputs = service.translations().list(source='en', target='fr', q=inputs).execute()
##### print outputs
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))

### Vision API

# Running Vision API
import base64
IMAGE="gs://cloud-training-demos/vision/sign2.jpg"
vservice = build('vision', 'v1', developerKey=APIKEY)
request = vservice.images().annotate(body={
        'requests': [{
                'image': {
                    'source': {
                        'gcs_image_uri': IMAGE
                    }
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 3,
                }]
            }],
        })
responses = request.execute(num_retries=3)
print(responses)

### Translate sign

inputs=[foreigntext]
outputs = service.translations().list(source=foreignlang, target='en', q=inputs).execute()
##### print(outputs)
for input, output in zip(inputs, outputs['translations']):
  print("{0} -> {1}".format(input, output['translatedText']))

### Sentiment analysis with Natural Language Processing API
polarity & magnitude


lservice = build('language', 'v1beta1', developerKey=APIKEY)
quotes = [
  'To succeed, you must have tremendous perseverance, tremendous will.',
  'It’s not that I’m so smart, it’s just that I stay with problems longer.',
  'Love is quivering happiness.',
  'Love is of all passions the strongest, for it attacks simultaneously the head, the heart, and the senses.',
  'What difference does it make to the dead, the orphans and the homeless, whether the mad destruction is wrought under the name of totalitarianism or in the holy name of liberty or democracy?',
  'When someone you love dies, and you’re not expecting it, you don’t lose her all at once; you lose her in pieces over a long time — the way the mail stops coming, and her scent fades from the pillows and even from the clothes in her closet and drawers. '
]
for quote in quotes:
  response = lservice.documents().analyzeSentiment(
    body={
      'document': {
         'type': 'PLAIN_TEXT',
         'content': quote
      }
    }).execute()
  polarity = response['documentSentiment']['polarity']
  magnitude = response['documentSentiment']['magnitude']
  print('POLARITY=%s MAGNITUDE=%s for %s' % (polarity, magnitude, quote))

### Speech API


## Pub/Sub

- Clients and services publish to a Pub/Sub Topic
- Services subscribe to this topic (pull or push)

## Serverless data pipelines

- p = beam.Pipeline(options)
- lines = p | beam.io.ReadFromText('gs://...')
- traffic = lines | beam.Map(parse_data).with_output_types(unicode)
                  | beam.Map(get_speedbysensor) # (sensor, speed)
                  | beam.GroupByKey() # (sensor, [speed])
                  | beam.Map(avg_speed) # (sensor, avgspeed)
                  | beam.Map(lambda tup: '%s: %d' % tup))
- output = traffic | beam.io.WriteToText('gs://...')
- p.run()

### Same code does real-time and batch, injesting data from Pub/Sub

- options = PipelineOptions(pipeline_args)
- options.view_as(StandardOptions).streaming = True
- p = beam.Pipeline(options = options)
- lines = p | beam.io.ReadStringsFromPubSub(input_topic)
- traffic = lines | beam.Map(parse_data).with_output_types(unicode)
                  | beam.Map(get_speedbysensor) # (sensor, speed)
                  | beam.WindowInto(window.FixedWindow(15, 0))
                  | beam.GroupByKey() # (sensor, [speed])
                  | beam.Map(avg_speed) # (sensor, avgspeed)
                  | beam.Map(lambda tup: '%s: %d' % tup))
- output = traffic | beam.io.WriteStringsToPubSub(output_topic)
- p.run()