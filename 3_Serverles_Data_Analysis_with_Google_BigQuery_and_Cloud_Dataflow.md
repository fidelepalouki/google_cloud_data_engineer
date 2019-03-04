# BigQuery

- No-ops data warehousing and analytics
- MapReduce => shard (split) data into nodes, parallelize what is called the map operations, then combine the results in a different set of machines compute nodes that are called the reduce operations
- Doesn't scale very well because we mixed Compute and Storage

* BigQuery allows to query very large datasets in seconds
* Is a columnar database
* To reduce costs => limit the number of columns on which we do our queries

Example

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

Can query from multiple tables

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

Join on fiels accross tables

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

Aggregate and Boolean functions

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

Get both the number of flights delayed and the total number of flights in a single quer

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

String operations

```SQL
SELECT
  CONCAT(CAST(year AS STRING), '-', LPAD(CAST(month AS STRING),2,'0'), '-', LPAD(CAST(day AS STRING),2,'0')) AS rainyday
FROM
  `bigquery-samples.weather_geo.gsod`
WHERE
  station_number = 725030
  AND total_precipitation > 0
```

Join on Date

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

Subquery

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

Cloud shell

```bash
curl https://storage.googleapis.com/cloud-training/CPB200/BQ/lab4/schema_flight_performance.json -o schema_flight_performance.json
```

Create a table named flights_2014 in the cpb101_flight_data dataset

```bash
bq load --source_format=NEWLINE_DELIMITED_JSON $DEVSHELL_PROJECT_ID:cpb101_flight_data.flights_2014 gs://cloud-training/CPB200/BQ/lab4/domestic_2014_flights_*.json ./schema_flight_performance.json
```

```bash
bq ls $DEVSHELL_PROJECT_ID:cpb101_flight_data
```

Use the cli to extract the table

```bash
bq extract cpb101_flight_data.AIRPORTS gs://$BUCKET/bq/airports2.csv
```

Advanced capabilities

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

Use of Structs

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

Join condition and Window condition

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

Standard SQL functions

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

Analytical window functions

- Standard aggregations
  - SUM, AVG, MIN, MAX, COUNT, ...
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

Window funtion example

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

Date functions

- DATE(year, month, day)
- DATE(timestamp)
- DATETIME(date, time)

Bigquery supports User-defined functions, allow functionality not supported by standard SQL

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

# Dataflow

- No-ops data pipeline for reliable, scalable data processing. (On batch or stream data)
- Java or Python
