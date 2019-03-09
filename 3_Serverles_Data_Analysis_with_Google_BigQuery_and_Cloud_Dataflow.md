# BigQuery

- No-ops data warehousing and analytics
- MapReduce => shard (split) data into nodes, parallelize what is called the map operations, then combine the results in a different set of machines compute nodes that are called the reduce operations
- Doesn't scale very well because we mixed Compute and Storage

* BigQuery allows to query very large datasets in seconds
* Is a columnar database
* To reduce costs => limit the number of columns on which we do our queries

### Example

```SQL
SELECT
  airline,
  SUM(IF(arrival_delay > 0, 1, 0)) AS num_delayed,
  COUNT(arrival_delay) AS total_flights
FROM
  `bigquery-samples.airline_ontime_data_flights`
WHERE
  arrival_airport='OKC'
  AND departure_airport='DFW'
GROUP BY
  airline, departure_airport
```

### Can query from multiple tables

```SQL
SELECT
  FORMAT_UTC_USEC(event.timestamp_in_sec) AS time,
  request_url
FROM
  [myproject-1234:applogs.events_20120501],
  [myproject-1234:applogs.events_20120502],
  [myproject-1234:applogs.events_20120503],
WHERE
  event.username = 'root' AND
  NOT event.soource_ip.is_internal;
```

or simpler

```SQL
TABLE_DATE_RANGE(myproject-1234:applogs.events_,
  TIMESTAMP('20120501'),
  TIMESTAMP('20120501'))
```

### Join on fiels accross tables

```SQL
SELECT
  f.airline,
  SUM(IF(f.arrival_delay > 0, 1, 0)) AS num_delayed,
  COUNT(f.arrival_delay) AS total_flights
FROM
  `bigquery-samples.airline_ontime_data_flights` AS f
JOIN(
  SELECT
    CONCAT(CAST(year AS STRING), '-', LDAP(CAST(month AS STRING), 2, '0'), '-', LDAP(CAST(day AS STRING), 2, '0')) AS rainyday
  FROM
    `bigquery-samples.wheather_geo.gsod`
  WHERE
    station_number = 725030
    AND total_precipitation > 0
) AS w
ON
  w.rainyday = f.date
WHERE f.arrival_airport = 'LGA'
GROUP BY f.airline
```

Elle est dans une dynamique de victoire, elle n'a pas envie de s'associer à une spirale de défaite, haha Cyril

## Exploring BigQuery

```SQL
SELECT
  airline,
  date,
  departure_delay
FROM
  `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_delay > 0
  AND departure_airport = 'LGA'
LIMIT
  100
```

### Aggregate and Boolean functions

```SQL
SELECT
  airline,
  COUNT(departure_delay)
FROM
   `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_airport = 'LGA'
  AND date = '2008-05-13'
GROUP BY
  airline
ORDER BY airline
```

```SQL
SELECT
  airline,
  COUNT(departure_delay)
FROM
   `bigquery-samples.airline_ontime_data.flights`
WHERE
  departure_delay > 0 AND
  departure_airport = 'LGA'
  AND date = '2008-05-13'
GROUP BY
  airline
ORDER BY airline
```

### Get both the number of flights delayed and the total number of flights in a single quer

```SQL
SELECT
  f.airline,
  COUNT(f.departure_delay) AS total_flights,
  SUM(IF(f.departure_delay > 0, 1, 0)) AS num_delayed
FROM
   `bigquery-samples.airline_ontime_data.flights` AS f
WHERE
  f.departure_airport = 'LGA' AND f.date = '2008-05-13'
GROUP BY
  f.airline
```

### String operations

```SQL
SELECT
  CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING),2,'0'), '-', LPAD(CAST(day AS STRING),2,'0')) AS rainyday
FROM
  `bigquery-samples.weather_geo.gsod`
WHERE
  station_number = 725030
  AND total_precipitation > 0
```

### Join on Date

```SQL
SELECT
  f.airline,
  SUM(IF(f.arrival_delay > 0, 1, 0)) AS num_delayed,
  COUNT(f.arrival_delay) AS total_flights
FROM
  `bigquery-samples.airline_ontime_data.flights` AS f
JOIN (
  SELECT
    CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING),2,'0'), '-', LPAD(CAST(day AS STRING),2,'0')) AS rainyday
  FROM
    `bigquery-samples.weather_geo.gsod`
  WHERE
    station_number = 725030
    AND total_precipitation > 0) AS w
ON
  w.rainyday = f.date
WHERE f.arrival_airport = 'LGA'
GROUP BY f.airline
```

### Subquery

```SQL
SELECT
  airline,
  num_delayed,
  total_flights,
  num_delayed / total_flights AS frac_delayed
FROM (
SELECT
  f.airline AS airline,
  SUM(IF(f.arrival_delay > 0, 1, 0)) AS num_delayed,
  COUNT(f.arrival_delay) AS total_flights
FROM
  `bigquery-samples.airline_ontime_data.flights` AS f
JOIN (
  SELECT
    CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING),2,'0'), '-', LPAD(CAST(day AS STRING),2,'0')) AS rainyday
  FROM
    `bigquery-samples.weather_geo.gsod`
  WHERE
    station_number = 725030
    AND total_precipitation > 0) AS w
ON
  w.rainyday = f.date
WHERE f.arrival_airport = 'LGA'
GROUP BY f.airline
  )
ORDER BY
  frac_delayed ASC
```

## Loading data into BigQuery

### Cloud shell

```bash
curl https://storage.googleapis.com/cloud-training/CPB200/BQ/lab4/schema_flight_performance.json -o schema_flight_performance.json
```

### Create a table named flights_2014 in the cpb101_flight_data dataset

```bash
bq load --source_format=NEWLINE_DELIMITED_JSON $DEVSHELL_PROJECT_ID:cpb101_flight_data.flights_2014 gs://cloud-training/CPB200/BQ/lab4/domestic_2014_flights_*.json ./schema_flight_performance.json
```

```bash
bq ls $DEVSHELL_PROJECT_ID:cpb101_flight_data
```

### Use the cli to extract the table

```bash
bq extract cpb101_flight_data.AIRPORTS gs://$BUCKET/bq/airports2.csv
```

## Advanced capabilities

Data types: STRING, INT64, FLOAT64, BOOL, ARRAY, STRUCT, TIMESTAMP

```SQL
WITH WashingtonStations AS (
  SELECT
    weather.stn AS station_id,
    ANY_VALUE(station.name) AS name
  FROM
    `bigquery-public-data.noaa_gsod.stations` AS stations
  INNER JOIN
    `bigquery-public-data.noaa_gsod.gsod2015` AS weather
  ON
    station.usaf = weather.stn
  WHERE
    station.state = 'WA'
    AND station.usaf != '999999'
  GROUP BY
    station_id
)

SELECT washington_stations.name,
  (
    SELECT COUNT(*)
    FROM `bigquery-public-data.noaa_gsod.gsod2015` AS weather
    WHERE washington_stations.station_id = weather.stn
    AND prcp > 0 AND prcp < 99
  ) AS rainy_days
FROM WashingtonStations AS washington_stations
ORDER BY rainy_days DESC;
```

### Use of Structs

```SQL
WITH TitlesAndScores AS (
  SELECT
    ARRAY_AGG(STRUCT(title, score)) AS titles,
    EXTRACT(DATE FROM time_ts) AS date
  FROM
    `bigquery-public-data.hacker_news.stories`
  WHERE
    score IS NOT NULL
    AND title IS NOT NULL
  GROUP BY date
)

SELECT date,
  ARRAY(SELECT AS STRUCT title, score
        FROM UNNEST(titles) ORDER BY score DESC
        LIMIT 2) AS top_articles
FROM TitlesAndScores;
```

### Join condition and Window condition

```SQL
WITH TopNames AS (
  SELECT
    name, SUM(number) AS occurences,
  FROM
    `bigquery-public-data.usa_names.usa_1910_2013`
  GROUP BY name
  ORDER BY occurences DESC LIMIT 100
)

SELECT name,
  SUM(word_count) AS frequency
FROM TopNames
JOIN `bigquery-public-data.samples.shakespeare`
ON STARTS_WITH(word, name)
GROUP BY name
ORDER BY frequency DESC LIMIT 10;
```

### Standard SQL functions

- Aggregate functions
- String functions
- Analytic (window) functions
- Datetime functions
- Array functions
- Other functions and operators

String function example (Regex)

```SQL
SELECT word, COUNT(word) as count
FROM `bigquery-public-data.samples.shakespeare`
WHERE (
  REGEXP_CONTAINS(word, r"^\w\w'\w\w")
)
GROUP BY word
ORDER BY count DESC
LIMIT 3
```

### Analytical window functions

- Standard aggregations
  - SUM(), AVG(), MIN(), MAX(), COUNT(), ...
- Navigation functions
  - LEAD() - Returns the value of a row n rows ahead of the current row
  - LAG() - Return the value of a row n rows behind the current row
  - NTH_VALUE() - Returns the value of the nth row in the window
- Ranking and numbering functions
  - CUME_DIST() - Returns the cumulative distribution of a value in a group
  - DENSE_RANK() - Returns the integer rank of a value in a group
  - ROW_NUMBER() - Returns the current row number in the query result
  - RANK() - Returns the integer rank of a value in a group of values
  - PERCENT_RANK() - Returns the rank of the current row, relative to the other rows in the partition

### Window funtion example

```SQL
SELECT
  corpus,
  word,
  word_count,
  RANK() OVER (
    PARTITION BY corpus
    ORDER BY word_count DESC
  ) rank,
FROM `bigquery-public-data.samples.shakespeare`
WHERE
  length(word) > 10 AND word_count > 10
LIMIT 40
```

### Date functions

- DATE(year, month, day)
- DATE(timestamp)
- DATETIME(date, time)

### Bigquery supports User-defined functions, allow functionality not supported by standard SQL

```SQL
  CREATE TEMPORARY FUNCTION
    addFourAndDivide(x INT64, y INT64)
    AS ((x + 4) / y);

  WITH numbers AS (
    SELECT 1 as val
    UNION ALL
    SELECT 3 as val
    UNION ALL
    SELECT 4 as val
    UNION ALL
    SELECT 5 as val
  )

  SELECT val, addFourAndDivide(val, 2) AS result
  FROM numbers;
```

What is is the most popular programming language used on Github during weekends ?

#### Information about code commits

```SQL
SELECT
  author.email,
  diff.new_path AS path,
  author.date
FROM
  `bigquery-public-data.github_repos.commits`,
  UNNEST(difference) diff
WHERE
  EXTRACT(YEAR
  FROM
    author.date)=2016
LIMIT 10
```

#### Extract programming language

```SQL
SELECT
  author.email,
  LOWER(REGEXP_EXTRACT(diff.new_path, r'\.([^\./\(~_ \- #]*)$')) lang,
  diff.new_path AS path,
  author.date
FROM
  `bigquery-public-data.github_repos.commits`,
  UNNEST(difference) diff
WHERE
  EXTRACT(YEAR
  FROM
    author.date)=2016
LIMIT
  10
```

#### Group by language by desc order

```SQL
WITH
  commits AS (
  SELECT
    author.email,
    LOWER(REGEXP_EXTRACT(diff.new_path, r'\.([^\./\(~_ \- #]*)$')) lang,
    diff.new_path AS path,
    author.date
  FROM
    `bigquery-public-data.github_repos.commits`,
    UNNEST(difference) diff
  WHERE
    EXTRACT(YEAR
    FROM
      author.date)=2016 )
SELECT
  lang,
  COUNT(path) AS numcommits
FROM
  commits
WHERE
  LENGTH(lang)<8
  AND lang IS NOT NULL
  AND REGEXP_CONTAINS(lang, '[a-zA-Z]')
GROUP BY
  lang
HAVING
  numcommits > 100
ORDER BY
  numcommits DESC
```

#### Weekend or Weekday

```SQL
WITH
  commits AS (
  SELECT
    author.email,
    EXTRACT(DAYOFWEEK
    FROM
      author.date) BETWEEN 2
    AND 6 is_weekday,
    LOWER(REGEXP_EXTRACT(diff.new_path, r'\.([^\./\(~_ \- #]*)$')) lang,
    diff.new_path AS path,
    author.date
  FROM
    `bigquery-public-data.github_repos.commits`,
    UNNEST(difference) diff
  WHERE
    EXTRACT(YEAR
    FROM
      author.date)=2016)
SELECT
  lang,
  is_weekday,
  COUNT(path) AS numcommits
FROM
  commits
WHERE
  lang IS NOT NULL
GROUP BY
  lang,
  is_weekday
HAVING
  numcommits > 100
ORDER BY
  numcommits DESC
```

## [Lak medium article](https://medium.freecodecamp.org/exploring-a-powerful-sql-pattern-array-agg-struct-and-unnest-b7dcc6263e36)

```SQL
#standardsql
WITH hurricanes AS (
SELECT
  MIN(NAME) AS name,
  ARRAY_AGG(STRUCT(iso_time, latitude, longitude, usa_sshs) ORDER BY iso_time ASC) AS track
FROM
  `bigquery-public-data.noaa_hurricanes.hurricanes`
WHERE
  season = '2010' AND basin = 'NA'
GROUP BY
  sid
),

cat_hurricane AS (
SELECT name, track, (SELECT MAX(usa_sshs) FROM UNNEST(track))  AS category
from hurricanes
ORDER BY category DESC
)

SELECT
  name,
  category,
  (SELECT AS STRUCT iso_time, latitude, longitude
   FROM UNNEST(track)
   WHERE usa_sshs = category ORDER BY iso_time LIMIT 1).*
FROM cat_hurricane
ORDER BY category DESC, name ASC
```

An other solution in the comments

```SQL
SELECT
 MIN(NAME) AS name,
 ARRAY_AGG(STRUCT(usa_sshs as category, iso_time, latitude, longitude) ORDER BY usa_sshs desc, iso_time ASC)[ordinal(1)] as track
FROM
 `bigquery-public-data.noaa_hurricanes.hurricanes`
WHERE
 season = ‘2010’ AND basin = ‘NA’
GROUP BY
 sid
ORDER BY track.category DESC, name ASC
```

# Dataflow

- No-ops data pipeline for reliable, scalable data processing. (On batch or stream data)
- Java or Python

- small file => withoutSharding()
-

### Install packages

```bash
apt-get install python-pip
pip install google-cloud-dataflow oauth2client==3.0.0
pip install -U pip
```

### Execute the pipeline locally

```python
import apache_beam as beam
import sys

def my_grep(line, term):
   if line.startswith(term):
      yield line

if __name__ == '__main__':
   p = beam.Pipeline(argv=sys.argv)
   input = '../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java'
   output_prefix = '/tmp/output'
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()
```

```bash
ls -al /tmp
```

```bash
cat /tmp/output-*
```

### Execute grepc.py on the cloud

```bash
gsutil cp ../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java gs://$BUCKET/javahelp
```

```python
import apache_beam as beam

def my_grep(line, term):
   if line.startswith(term):
      yield line

PROJECT='qwiklabs-gcp-e3153433096d9532'
BUCKET='qwiklabs-gcp-e3153433096d9532'

def run():
   argv = [
      '--project={0}'.format(PROJECT),
      '--job_name=examplejob2',
      '--save_main_session',
      '--staging_location=gs://{0}/staging/'.format(BUCKET),
      '--temp_location=gs://{0}/staging/'.format(BUCKET),
      '--runner=DataflowRunner'
   ]

   p = beam.Pipeline(argv=argv)
   input = 'gs://{0}/javahelp/*.java'.format(BUCKET)
   output_prefix = 'gs://{0}/javahelp/output'.format(BUCKET)
   searchTerm = 'import'

   # find all lines that contain the searchTerm
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'Grep' >> beam.FlatMap(lambda line: my_grep(line, searchTerm) )
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run()

if __name__ == '__main__':
   run()
```

### With Java

```bash
mvn archetype:generate \
  -DarchetypeArtifactId=google-cloud-dataflow-java-archetypes-starter \
  -DarchetypeGroupId=com.google.cloud.dataflow \
  -DgroupId=com.example.pipelinesrus.newidea \
  -DartifactId=newidea \
  -Dversion="[1.0.0,2.0.0]" \
  -DinteractiveMode=false
```

```java
public class Grep {

	@SuppressWarnings("serial")
	public static void main(String[] args) {
		PipelineOptions options = PipelineOptionsFactory.fromArgs(args).withValidation().create();
		Pipeline p = Pipeline.create(options);

		String input = "src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java";
		String outputPrefix = "/tmp/output";
		final String searchTerm = "import";

		p //
				.apply("GetJava", TextIO.read().from(input)) //
				.apply("Grep", ParDo.of(new DoFn<String, String>() {
					@ProcessElement
					public void processElement(ProcessContext c) throws Exception {
						String line = c.element();
						if (line.contains(searchTerm)) {
							c.output(line);
						}
					}
				})) //
				.apply(TextIO.write().to(outputPrefix).withSuffix(".txt").withoutSharding());

		p.run();
	}
}
```

- run_locally.sh

```bash
#!/bin/bash
if [ "$#" -ne 1 ]; then
   echo "Usage:   ./run_locally.sh mainclass-basename"
   echo "Example: ./run_oncloud.sh Grep"
   exit
fi
MAIN=com.google.cloud.training.dataanalyst.javahelp.$1
export PATH=/usr/lib/jvm/java-8-openjdk-amd64/bin/:$PATH
mvn compile -e exec:java -Dexec.mainClass=$MAIN
```

```bash
cd ~/training-data-analyst/courses/data_analysis/lab2

export PATH=/usr/lib/jvm/java-8-openjdk-amd64/bin/:$PATH
cd ~/training-data-analyst/courses/data_analysis/lab2/javahelp
mvn compile -e exec:java \
 -Dexec.mainClass=com.google.cloud.training.dataanalyst.javahelp.Grep
```

### In the cloud

```bash
gsutil cp ../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/*.java gs://$BUCKET/javahelp
```

```bash
cd ~/training-data-analyst/courses/data_analysis/lab2/javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp
```

```java
echo $BUCKET
```

- run_oncloud1.sh

```bash
#!/bin/bash
if [ "$#" -ne 3 ]; then
   echo "Usage:   ./run_oncloud.sh project-name  bucket-name  mainclass-basename"
   echo "Example: ./run_oncloud.sh cloud-training-demos  cloud-training-demos  JavaProjectsThatNeedHelp"
   exit
fi
PROJECT=$1
BUCKET=$2
MAIN=com.google.cloud.training.dataanalyst.javahelp.$3
echo "project=$PROJECT  bucket=$BUCKET  main=$MAIN"
export PATH=/usr/lib/jvm/java-8-openjdk-amd64/bin/:$PATH
mvn compile -e exec:java \
 -Dexec.mainClass=$MAIN \
      -Dexec.args="--project=$PROJECT \
      --stagingLocation=gs://$BUCKET/staging/ \
      --tempLocation=gs://$BUCKET/staging/ \
      --runner=DataflowRunner"
```

```bash
bash run_oncloud1.sh $DEVSHELL_PROJECT_ID $BUCKET Grep
```

- beam.GroupByKey()
- Combine.globally(sum)
- Combine.perKey(sum)
- Count.perKey() for instance > GroupByKey() then processing count

## MapReduce in Dataflow

```bash
sudo apt-get install python-pip -y

pip install google-cloud-dataflow oauth2client==3.0.0
# downgrade as 1.11 breaks apitools
sudo pip install --force six==1.10
pip install -U pip
pip -V
sudo pip install apache_beam
```

```bash
cd ~/training-data-analyst/courses/data_analysis/lab2/python
nano is_popular.py
```

```python
import apache_beam as beam
import argparse

def startsWith(line, term):
   if line.startswith(term):
      yield line

def splitPackageName(packageName):
   """e.g. given com.example.appname.library.widgetname
           returns com
	           com.example
                   com.example.appname
      etc.
   """
   result = []
   end = packageName.find('.')
   while end > 0:
      result.append(packageName[0:end])
      end = packageName.find('.', end+1)
   result.append(packageName)
   return result

def getPackages(line, keyword):
   start = line.find(keyword) + len(keyword)
   end = line.find(';', start)
   if start < end:
      packageName = line[start:end].strip()
      return splitPackageName(packageName)
   return []

def packageUse(line, keyword):
   packages = getPackages(line, keyword)
   for p in packages:
      yield (p, 1)

def by_value(kv1, kv2):
   key1, value1 = kv1
   key2, value2 = kv2
   return value1 < value2

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='Find the most used Java packages')
   parser.add_argument('--output_prefix', default='/tmp/output', help='Output prefix')
   parser.add_argument('--input', default='../javahelp/src/main/java/com/google/cloud/training/dataanalyst/javahelp/', help='Input directory')

   options, pipeline_args = parser.parse_known_args()
   p = beam.Pipeline(argv=pipeline_args)

   input = '{0}*.java'.format(options.input)
   output_prefix = options.output_prefix
   keyword = 'import'

   # find most used packages
   (p
      | 'GetJava' >> beam.io.ReadFromText(input)
      | 'GetImports' >> beam.FlatMap(lambda line: startsWith(line, keyword))
      | 'PackageUse' >> beam.FlatMap(lambda line: packageUse(line, keyword))
      | 'TotalUse' >> beam.CombinePerKey(sum)
      | 'Top_5' >> beam.transforms.combiners.Top.Of(5, by_value)
      | 'write' >> beam.io.WriteToText(output_prefix)
   )

   p.run().wait_until_finish()
```

### With side inputs

- Convert the PCollection into a view (List or Map)
- Pass that view as a sideInput
- Retrieve that view for processing

- Java projects that need help

```python
import argparse
import logging
import datetime, os
import apache_beam as beam
import math

'''

This is a dataflow pipeline that demonstrates Python use of side inputs. The pipeline finds Java packages
on Github that are (a) popular and (b) need help. Popularity is use of the package in a lot of other
projects, and is determined by counting the number of times the package appears in import statements.
Needing help is determined by counting the number of times the package contains the words FIXME or TODO
in its source.

@author tomstern
based on original work by vlakshmanan

python JavaProjectsThatNeedHelp.py --project <PROJECT> --bucket <BUCKET> --DirectRunner or --DataFlowRunner

'''

# Global values
TOPN=1000


# ### Functions used for both main and side inputs

def splitPackageName(packageName):
   """e.g. given com.example.appname.library.widgetname
           returns com
	           com.example
                   com.example.appname
      etc.
   """
   result = []
   end = packageName.find('.')
   while end > 0:
      result.append(packageName[0:end])
      end = packageName.find('.', end+1)
   result.append(packageName)
   return result

def getPackages(line, keyword):
   start = line.find(keyword) + len(keyword)
   end = line.find(';', start)
   if start < end:
      packageName = line[start:end].strip()
      return splitPackageName(packageName)
   return []

def packageUse(record, keyword):
   if record is not None:
     lines=record.split('\n')
     for line in lines:
       if line.startswith(keyword):
         packages = getPackages(line, keyword)
         for p in packages:
           yield (p, 1)

def by_value(kv1, kv2):
   key1, value1 = kv1
   key2, value2 = kv2
   return value1 < value2

def is_popular(pcoll):
 return (pcoll
    | 'PackageUse' >> beam.FlatMap(lambda rowdict: packageUse(rowdict['content'], 'import'))
    | 'TotalUse' >> beam.CombinePerKey(sum)
    | 'Top_NNN' >> beam.transforms.combiners.Top.Of(TOPN, by_value) )


def packageHelp(record, keyword):
   count=0
   package_name=''
   if record is not None:
     lines=record.split('\n')
     for line in lines:
       if line.startswith(keyword):
         package_name=line
       if 'FIXME' in line or 'TODO' in line:
         count+=1
     packages = (getPackages(package_name, keyword) )
     for p in packages:
         yield (p,count)

def needs_help(pcoll):
 return (pcoll
    | 'PackageHelp' >> beam.FlatMap(lambda rowdict: packageHelp(rowdict['content'], 'package'))
    | 'TotalHelp' >> beam.CombinePerKey(sum)
    | 'DropZero' >> beam.Filter(lambda packages: packages[1]>0 ) )


# Calculate the final composite score
#
#    For each package that is popular
#    If the package is in the needs help dictionary, retrieve the popularity count
#    Multiply to get compositescore
#      - Using log() because these measures are subject to tournament effects
#

def compositeScore(popular, help):
    for element in popular:
      if help.get(element[0]):
         composite = math.log(help.get(element[0])) * math.log(element[1])
         if composite > 0:
           yield (element[0], composite)


# ### main

# Define pipeline runner (lazy execution)
def run():

  # Command line arguments
  parser = argparse.ArgumentParser(description='Demonstrate side inputs')
  parser.add_argument('--bucket', required=True, help='Specify Cloud Storage bucket for output')
  parser.add_argument('--project',required=True, help='Specify Google Cloud project')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--DirectRunner',action='store_true')
  group.add_argument('--DataFlowRunner',action='store_true')

  opts = parser.parse_args()

  if opts.DirectRunner:
    runner='DirectRunner'
  if opts.DataFlowRunner:
    runner='DataFlowRunner'

  bucket = opts.bucket
  project = opts.project

  #    Limit records if running local, or full data if running on the cloud
  limit_records=''
  if runner == 'DirectRunner':
     limit_records='LIMIT 3000'
  get_java_query='SELECT content FROM [fh-bigquery:github_extracts.contents_java_2016] {0}'.format(limit_records)

  argv = [
    '--project={0}'.format(project),
    '--job_name=javahelpjob',
    '--save_main_session',
    '--staging_location=gs://{0}/staging/'.format(bucket),
    '--temp_location=gs://{0}/staging/'.format(bucket),
    '--runner={0}'.format(runner),
    '--max_num_workers=5'
    ]

  p = beam.Pipeline(argv=argv)


  # Read the table rows into a PCollection (a Python Dictionary)
  bigqcollection = p | 'ReadFromBQ' >> beam.io.Read(beam.io.BigQuerySource(project=project,query=get_java_query))

  popular_packages = is_popular(bigqcollection) # main input

  help_packages = needs_help(bigqcollection) # side input

  # Use side inputs to view the help_packages as a dictionary
  results = popular_packages | 'Scores' >> beam.FlatMap(lambda element, the_dict: compositeScore(element,the_dict), beam.pvalue.AsDict(help_packages))

  # Write out the composite scores and packages to an unsharded csv file
  output_results = 'gs://{0}/javahelp/Results'.format(bucket)
  results | 'WriteToStorage' >> beam.io.WriteToText(output_results,file_name_suffix='.csv',shard_name_template='')

  # Run the pipeline (all operations are deferred until run() is called).


  if runner == 'DataFlowRunner':
     p.run()
  else:
     p.run().wait_until_finish()
  logging.getLogger().setLevel(logging.INFO)


if __name__ == '__main__':
  run()
```

- Executing the pipeline locally

```bash
python JavaProjectsThatNeedHelp.py --bucket $BUCKET --project $DEVSHELL_PROJECT_ID --DirectRunner
```

- Execution the pipeline on the cloud

```bash
python JavaProjectsThatNeedHelp.py --bucket $BUCKET --project $DEVSHELL_PROJECT_ID --DataFlowRunner
```

### Dataflow templates

- Options which are compile time parameters by default, have to be converted into runtime parameters so that they can be accessible to non developers
  -Dataflow templates:

- https://cloud.google.com/solutions/processing-logs-at-scale-using-dataflow
