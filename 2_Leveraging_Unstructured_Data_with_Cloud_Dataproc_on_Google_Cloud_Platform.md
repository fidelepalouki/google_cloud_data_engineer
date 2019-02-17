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

## Review

- Hadoop alternatives come with a lot of overhead
- Dataproc is designed to deal with those overhead
- Create a cluster specifically for one job
- Use Cloud Storage instead of HDFS (You lose your persitent disk if the node is shut down)
- Shutdown the cluster when it is not actually processing data
- Use custom machines to closely match the CPU and memory requirements for the job
- On non-critical jobs requiring huge clusters, use preemptible VMs to hasten results and cut costs at the same time
