## Unstructured data

- Data that may even have a schema but not adapted to the given purpose
- Derive value from it
- 1 PetaByte == 12 Empire state buildings == Every tweet ever tweeted \* 50 == 27 years to download over 4G
  == 2 micrograms of DNA == 1day's worth of video uploaded to youtube == 200 servers logging at 50 entries per second for 3 years
- Horizontal scaling (scale out) vs Vertical scaling (scale up)
  50000 images hosted in GCS => (5s o processing for each image) ==> 17 days ==> 4 core CPU => 4.5days ==> 100 computers in parallel ==> 4hours ==> 1000 computers ==> 25min and so on !! Yeah horizontal scaling

## Spark

It is able to mix different kinds of applications and to adjust how it uses the avaible resources

Hadoop only => Lot of overhead, for reconfiguring, scaling can take days or weeks, moving data around which has no value to the business
Dataproc => Managed Hadoop and Spark Clusters => Focus only on insight and analytics for driving better decisions for the company

gcloud dataproc clusters create my-cluster --zone us-central1-a --master-machine-type n1-standard-1 --num-workers 2 --master-boot-disk-size 50 --num-worker 2 --worker-machine-type n1-standard-1 --worker-boot-disk-size 50

gcloud dataproc --help
gcloud dataproc clusters create --help

9870: Hadoop Admin Interface
8088: Hadoop Jobs or Applications Interface

## Review

- Hadoop alternatives come with a lot of overhead
- Dataproc is designed to deal with those overhead
- Create a cluster specifically for one job
- Use Cloud Storage instead of HDFS (You lose your persitent disk if the node is shut down)
- Shutdown the cluster when it is not actually processing data
- Use custom machines to closely match the CPU and memory requirements for the job
- On non-critical jobs requiring huge clusters, use preemptible VMs to hasten results and cut costs at the same time

## Running jobs

Stage data in HDFS

hadoop fs -mkdir /pet-details
hadoop fs -put pet-details.txt /pet-details

### Hive

hive
CREATE DATABASE pets;
use pets;

CREATE TABLE details (Type String, Name String, Breed String, Color String, Weight Int) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',';
SHOW TABLES;
DESCRIBE pets.details;

load data INPATH '/pet-details/pet-details.txt' OVERWRITE INTO TABLE details;

SELECT \* FROM pets.details;

quit;

Hive ==>

- designed for batch jobs not for transactions
- ingests data into a warehouse requiring a schema
- doesn't support real-time queries, row-level updates, or unstructured data
- some queries may run much slower than others due to the underlying transformations Hive has to implement to simulate SQL

### Pig

cat pet-details.pig

rmf /GroupedByType
x1 = load '/pet-details' using PigStorage(',') as (Type:chararray,Name:chararray,Breed:chararray,Color:chararray,Weight:int);
x2 = filter x1 by Type != 'Type';
x3 = foreach x2 generate Type, Name, Breed, Color, Weight, Weight / 2.24 as Kilos:float;
x4 = filter x3 by LOWER(Color) == 'black' or LOWER(Color) == 'white';
x5 = group x4 by Type;
store x5 into '/GroupedByType';

pig < pet-details.pig

hadoop fs -get /GroupByType/part\* .

cat part-r-00000

Pig ==

- provides SQL primitives similar to Hive, but in a more flexible scripting language format
- can deal with semi-structured data
- it generates Java MapReduce jobs
- is not designed to deal with unstructured data

## HDFS (from GFS)

- Replicating blocks of storage accross multiple nodes
- monitoring the health of the nodes
- recovering the data when a node is lost
  ==> this allows the data to be pre-sharded on the cluster, instead of in mass storage in the data center

Serveless platform for all stages of the analytics data lifecycle

- Injest: Pub/sub
- Processing: Dataflow
- Analysis: BigQuery

Why high bisection bandwith is a game-changer

- First wave: Copy the data to the processor for use
- Second wave: (Hadoop) Distribute the data and process it in place
- Third wave: (Google Cloud) Use the data where it is without copying it

HDFS => GS make the cluster stateless, HDFS is now used for temporary storage only

RDD: Resilient Distributed Dataset
DAG: Directed Acyclic Graph

Transformations(map, flatMap) => stores them in the DAG versus Actions (collect, count, take) executes the transformations in the DAG and the avaiable resources and creates pipelines to efficiently perform the work

### Spark

- hadoop fs -ls /

pyspark
lines = sc.textFile("/sampledata/sherlock-holmes.txt")
type(lines) # <class 'pyspark.rdd.RDD'>
lines.count()lines.take(15)

words = lines.flatMap(lambda x: x.split(' '))
type(words) # <class 'pyspark.rdd.PipelinedRDD'>
words.count()
words.take(15)

pairs = words.map(lambda x: (x,len(x)))
type(pairs) # <class 'pyspark.rdd.PipelinedRDD'>
pairs.count()
pairs.take(5)

pairs = words.map(lambda x: (len(x),1))
pairs.take(5)

from operator import add
wordsize = pairs.reduceByKey(add)
type(wordsize)
wordsize.count()
wordsize.take(5)

output = wordsize.collect()
type(output) # <type 'list'>
for (size,count) in output: print(size, count)

output = wordsize.sortByKey().collect()
for (size,count) in output: print(size, count)

## Recap program

from operator import add
lines = sc.textFile("/sampledata/sherlock-holmes.txt")

words = lines.flatMap(lambda x: x.split(' '))
pairs = words.map(lambda x: (len(x),1))
wordsize = pairs.reduceByKey(add)
output = wordsize.sortByKey().collect()

## With spark optimizations

output2 = lines
.flatMap(lambda x: x.split(' '))
.map(lambda x: (len(x),1))
.reduceByKey(add)
.sortByKey()
.collect()

for (size, count) in output2: print(size, count)

exit()

## Spark script in nano wordcount.py

from pyspark.sql import SparkSession
from operator import add
import re

print("Okay Google.")

spark = SparkSession\
 .builder\
 .appName("CountUniqueWords")\
 .getOrCreate()

lines = spark.read.text("/sampledata/road-not-taken.txt").rdd.map(lambda x: x[0])
counts = lines.flatMap(lambda x: x.split(' ')) \
 .filter(lambda x: re.sub('[^a-za-z]+', '', x)) \
 .filter(lambda x: len(x)>1 ) \
 .map(lambda x: x.upper()) \
 .map(lambda x: (x, 1)) \
 .reduceByKey(add) \
 .sortByKey()
output = counts.collect()
for (word, count) in output:
print("%s = %i" % (word, count))

spark.stop()

## Submit the job

spark-submit wordcount.py

## Replace HDFS with GCS

gsutil cp /training/road-not-taken.txt gs://\$BUCKET

from pyspark.sql import SparkSession
from operator import add
import re

print("Okay Google.")

spark = SparkSession\
 .builder\
 .appName("CountUniqueWords")\
 .getOrCreate()

lines = spark.read.text("/sampledata/road-not-taken.txt").rdd.map(lambda x: x[0])
counts = lines.flatMap(lambda x: x.split(' ')) \
 .filter(lambda x: re.sub('[^a-za-z]+', '', x)) \
 .filter(lambda x: len(x)>1 ) \
 .map(lambda x: x.upper()) \
 .map(lambda x: (x, 1)) \
 .reduceByKey(add) \
 .sortByKey()
output = counts.collect()
for (word, count) in output:
print("%s = %i" % (word, count))

spark.stop()

lines = spark.read.text("gs://<YOUR-BUCKET>/road-not-taken.txt").rdd.map(lambda x: x[0])

spark-submit wordcount.py

## Installation script

#### #!/bin/bash

#### install Google Python client on all nodes

apt-get update
apt-get install -y python-pip
pip install --upgrade google-api-python-client
ROLE=$(/usr/share/google/get_metadata_value attributes/dataproc-role)
if [[ "${ROLE}" == 'Master' ]]; then
echo "Only on master node ..."
fi

## Create the custom cluster

gcloud dataproc clusters create cluster-custom \
--bucket $BUCKET \
--subnet default \
--zone $MYZONE \
--region $MYREGION \
--master-machine-type n1-standard-2 \
--master-boot-disk-size 100 \
--num-workers 2 \
--worker-machine-type n1-standard-1 \
--worker-boot-disk-size 50 \
--num-preemptible-workers 2 \
--image-version 1.2 \
--scopes 'https://www.googleapis.com/auth/cloud-platform' \
--tags customaccess \
--project $PROJECT_ID \
--initialization-actions 'gs://'\$BUCKET'/init-script.sh','gs://dataproc-initialization-actions/datalab/datalab.sh'

## Create a firewall rule

gcloud compute \
--project=$PROJECT_ID \
firewall-rules create allow-custom \
--direction=INGRESS \
--priority=1000 \
--network=default \
--action=ALLOW \
--rules=tcp:9870,tcp:8088,tcp:8080 \
--source-ranges=$BROWSER_IP/32 \
--target-tags=customaccess

## Machine Learning

### stagelabs.sh

if [ -d "backup" ]; then
cp backup/_dataproc_ .
else
mkdir backup
cp _dataproc_ backup
fi

# Verify that the environment variables exist

#

OKFLAG=1
if [[ -v $BUCKET ]]; then
echo "BUCKET environment variable not found"
OKFLAG=0
fi
if [[ -v $DEVSHELL_PROJECT_ID ]]; then
echo "DEVSHELL_PROJECT_ID environment variable not found"
OKFLAG=0
fi
if [[ -v $APIKEY ]]; then
echo "APIKEY environment variable not found"
OKFLAG=0
fi

if [ OKFLAG==1 ]; then

#### Edit the script files

sed -i "s/your-api-key/$APIKEY/" *dataprocML.py
  sed -i "s/your-project-id/$DEVSHELL_PROJECT_ID/" *dataprocML.py
sed -i "s/your-bucket/\$BUCKET/" *dataprocML.py

#### Copy python scripts to the bucket

gsutil cp \*dataprocML.py gs://\$BUCKET/

#### Copy data to the bucket

gsutil cp gs:\/\/cloud-training\/gcpdei\/road* gs:\/\/\$BUCKET\/sampledata\/
gsutil cp gs:\/\/cloud-training\/gcpdei\/time* gs:\/\/\$BUCKET\/sampledata\/

fi

### 03-dataprocML.py

'''
This program reads a text file and passes to a Natural Language Processing
service, sentiment analysis, and processes the results in Spark.

'''
import logging
import argparse
import json
import os
from googleapiclient.discovery import build
from pyspark import SparkContext
sc = SparkContext("local", "Simple App")
'''
You must set these values for the job to run.
'''
APIKEY="AIzaSyBzuBeKgy5Ru4cbxeMf_vfUHBjJCWVYXKU" # CHANGE
print(APIKEY)
PROJECT_ID="qwiklabs-gcp-1dea41827e59f364" # CHANGE
print(PROJECT_ID)
print(PROJECT_ID)
BUCKET="qwiklabs-gcp-1dea41827e59f364" # CHANGE

## Wrappers around the NLP REST interface

def SentimentAnalysis(text):
from googleapiclient.discovery import build
lservice = build('language', 'v1beta1', developerKey=APIKEY)
response = lservice.documents().analyzeSentiment(
body={
'document': {
'type': 'PLAIN_TEXT',
'content': text
}
}).execute()

return response

##### main

#### We could use sc.textFiles(...)

####

#### However, that will read each line of text as a separate object.

#### And using the REST API to NLP for each line will rapidly exhaust the rate-limit quota

#### producing HTTP 429 errors

####

#### Instead, it is more efficient to pass an entire document to NLP in a single call.

####

#### So we are using sc.wholeTextFiles(...)

####

#### This provides a file as a tuple.

#### The first element is the file pathname, and second element is the content of the file.

####

sample = sc.wholeTextFiles("gs://{0}/sampledata/time-machine.txt".format(BUCKET))

#### Calling the Natural Language Processing REST interface

####

#### results = SentimentAnalysis(sampleline)

rdd1 = sample.map(lambda x: SentimentAnalysis(x[1]))

#### The RDD contains a dictionary, using the key 'sentences' picks up each individual sentence

#### The value that is returned is a list. And inside the list is another dictionary

#### The key 'sentiment' produces a value of another list.

#### And the keys magnitude and score produce values of floating numbers.

####

rdd2 = rdd1.flatMap(lambda x: x['sentences'] )\
 .flatMap(lambda x: [(x['sentiment']['magnitude'], x['sentiment']['score'], [x['text']['content']])] )
