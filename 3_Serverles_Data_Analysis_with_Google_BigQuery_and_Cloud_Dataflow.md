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

# Dataflow

- No-ops data pipeline for reliable, scalable data processing. (On batch or stream data)
- Java or Python

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

```SQL

```

```SQL

```

```SQL

```

```SQL

```
