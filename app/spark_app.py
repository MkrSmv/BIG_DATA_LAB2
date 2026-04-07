# coding: utf-8
"""
Spark Performance Application — NYC Yellow Taxi Analysis
Measures time and RAM across 6 pipeline stages.
Supports normal and optimized modes via CLI argument.
"""
import time
import logging
import psutil
import sys
import os
import contextlib

from pyspark.sql import SparkSession
from pyspark import SparkConf, StorageLevel
from pyspark.sql.functions import col, hour, dayofweek, count, avg, when
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ── CLI argument ────────────────────────────────────────────
OPTIMIZED = len(sys.argv) > 1 and sys.argv[1].lower() == "true"

# ── Logging setup ───────────────────────────────────────────
LOG_FILE = "/tmp/spark_app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)
# Suppress py4j noise
logging.getLogger("py4j").setLevel(logging.ERROR)

# ── Stage timing registry ───────────────────────────────────
stage_times: dict = {}


@contextlib.contextmanager
def log_stage(spark, name: str):
    """Time a logical Spark stage, log job start/end and active stage IDs."""
    spark.sparkContext.setJobDescription(name)
    logger.info(f"[JOB START] >>> {name}")
    t0 = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - t0
        stage_times[name] = elapsed
        tracker = spark.sparkContext.statusTracker()
        active = list(tracker.getActiveStageIds())
        logger.info(
            f"[JOB END]   <<< {name} | {elapsed:.3f}s | active stages: {active}"
        )
        spark.sparkContext.setJobDescription("")


# ── Main ─────────────────────────────────────────────────────
def main():
    conf = SparkConf()
    conf.set("spark.ui.showConsoleProgress", "false")
    conf.set("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
    conf.set("spark.executor.memory", "768m")  # container mem_limit=1g, leave room for JVM overhead
    conf.set("spark.driver.memory", "512m")   # driver only coordinates, doesn't hold data

    if OPTIMIZED:
        # Tune parallelism and shuffle for optimized runs
        conf.set("spark.default.parallelism", "12")
        conf.set("spark.sql.shuffle.partitions", "12")
    else:
        conf.set("spark.sql.shuffle.partitions", "100")

    spark = (
        SparkSession.builder
        .appName("NYCTaxiAnalysis")
        .master("spark://spark-master:7077")
        .config(conf=conf)
        .getOrCreate()
    )
    # Remove WARN and INFO noise from Spark internals; keep only ERROR
    spark.sparkContext.setLogLevel("ERROR")

    process = psutil.Process(os.getpid())
    total_start = time.time()

    logger.info("=" * 60)
    logger.info(f"NYCTaxiAnalysis | OPTIMIZED={OPTIMIZED}")
    logger.info(
        f"executor.memory=768m | driver.memory=512m | "
        f"shuffle.partitions={conf.get('spark.sql.shuffle.partitions')}"
    )
    logger.info("=" * 60)

    row_count = 0
    accuracy = 0.0
    df = None
    train_df = None
    test_df = None

    # ── Stage 1: Load ────────────────────────────────────────
    with log_stage(spark, "Load"):
        try:
            df = spark.read.parquet("hdfs:///nyc_taxi_2023.parquet")
            full_count = df.count()
            partitions = df.rdd.getNumPartitions()
            logger.info(f"Full dataset: {full_count:,} rows | {partitions} partitions")
            # Sample to ~500k rows — fits in 1g containers, still 5× above 100k requirement
            SAMPLE_FRACTION = 0.17
            df = df.sample(fraction=SAMPLE_FRACTION, seed=42)
            row_count = df.count()
            logger.info(f"Sampled {row_count:,} rows (fraction={SAMPLE_FRACTION})")
        except Exception as exc:
            logger.error(f"Cannot read HDFS data: {exc}")
            spark.stop()
            return

    # ── Stage 2: Preprocessing ───────────────────────────────
    with log_stage(spark, "Preprocessing"):
        df = df.select(
            col("VendorID").cast(IntegerType()),
            col("tpep_pickup_datetime").cast("timestamp"),
            col("passenger_count").cast(IntegerType()),
            col("trip_distance").cast(DoubleType()),
            col("RatecodeID").cast(IntegerType()),
            col("PULocationID").cast(IntegerType()),
            col("DOLocationID").cast(IntegerType()),
            col("payment_type").cast(IntegerType()),
            col("fare_amount").cast(DoubleType()),
            col("tip_amount").cast(DoubleType()),
            col("total_amount").cast(DoubleType()),
        ).filter(col("payment_type").isin([1, 2])).dropna()

        # Derived features
        df = (
            df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
              .withColumn("pickup_dow", dayofweek("tpep_pickup_datetime"))
              # Label: 1=credit card (1), 0=cash (2) — binary classification
              .withColumn("label", when(col("payment_type") == 1, 1.0).otherwise(0.0))
        )

        if OPTIMIZED:
            logger.info("OPTIMIZED: repartition(12) + persist(MEMORY_AND_DISK)")
            df = df.repartition(12).persist(StorageLevel.MEMORY_AND_DISK)

        clean_count = df.count()
        logger.info(
            f"After preprocessing: {clean_count:,} rows | {df.rdd.getNumPartitions()} partitions"
        )

    # ── Stage 3: Aggregations ────────────────────────────────
    with log_stage(spark, "Aggregations"):
        logger.info("[AGG 1/4] Trips and avg fare by VendorID")
        df.groupBy("VendorID").agg(
            count("*").alias("trips"),
            avg("trip_distance").alias("avg_distance"),
            avg("total_amount").alias("avg_total"),
        ).show()

        logger.info("[AGG 2/4] Trips by pickup hour")
        df.groupBy("pickup_hour").agg(
            count("*").alias("trips"),
            avg("fare_amount").alias("avg_fare"),
        ).orderBy("pickup_hour").show(24)

        logger.info("[AGG 3/4] Payment type distribution")
        df.groupBy("payment_type").count().orderBy("payment_type").show()

        logger.info("[AGG 4/4] Avg tip by day of week")
        df.groupBy("pickup_dow").agg(
            count("*").alias("trips"),
            avg("tip_amount").alias("avg_tip"),
        ).orderBy("pickup_dow").show()

    # ── Stage 4: ML Feature Engineering ─────────────────────
    with log_stage(spark, "ML-FeatureEng"):
        feature_cols = [
            "trip_distance", "passenger_count", "fare_amount",
            "tip_amount", "total_amount",
            "PULocationID", "DOLocationID",
            "RatecodeID", "pickup_hour", "pickup_dow",
        ]
        assembler = VectorAssembler(
            inputCols=feature_cols, outputCol="features", handleInvalid="skip"
        )
        df_ml = assembler.transform(df)
        train_df, test_df = df_ml.randomSplit([0.8, 0.2], seed=42)

        if OPTIMIZED:
            logger.info("OPTIMIZED: caching train/test splits")
            train_df = train_df.cache()
            test_df = test_df.cache()
            n_train = train_df.count()
            n_test = test_df.count()
            logger.info(f"Train: {n_train:,} | Test: {n_test:,}")

    # ── Stage 5: Model Training ──────────────────────────────
    with log_stage(spark, "ML-Training"):
        lr = LogisticRegression(
            featuresCol="features", labelCol="label",
            maxIter=15, regParam=0.01,  # 15 iterations is enough to converge on this dataset
        )
        logger.info("Training LogisticRegression (maxIter=15, regParam=0.01)...")
        model = lr.fit(train_df)
        logger.info(f"Training complete | iterations: {model.summary.totalIterations}")

    # ── Stage 6: Evaluation ──────────────────────────────────
    with log_stage(spark, "ML-Evaluation"):
        predictions = model.transform(test_df)
        evaluator = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        )
        accuracy = evaluator.evaluate(predictions)
        logger.info(f"Model accuracy: {accuracy:.4f}")

    # ── Cleanup: release cached DataFrames ───────────────────
    if OPTIMIZED and df is not None:
        df.unpersist()
        if train_df is not None:
            train_df.unpersist()
        if test_df is not None:
            test_df.unpersist()
        logger.info("Unpersisted all cached DataFrames")

    # ── Final summary ────────────────────────────────────────
    total_elapsed = time.time() - total_start
    ram_mb = process.memory_info().rss / (1024 * 1024)

    logger.info("=" * 60)
    logger.info(f"TOTAL TIME : {total_elapsed:.2f}s")
    logger.info(f"DRIVER RAM : {ram_mb:.2f} MB")
    logger.info(f"ACCURACY   : {accuracy:.4f}")
    logger.info("Stage breakdown:")
    for sname, st in stage_times.items():
        logger.info(f"  [{sname}]: {st:.3f}s")
    logger.info("=" * 60)

    spark.stop()

    # ── Machine-readable output parsed by plot_result.py ─────
    load_t = stage_times.get("Load", 0)
    agg_t  = stage_times.get("Aggregations", 0)
    ml_t   = stage_times.get("ML-Training", 0)

    print(f"[ROWS]{row_count}")
    print(f"[TIME]{total_elapsed}")
    print(f"[MEMORY]{ram_mb}mb")
    print(f"[LOAD_TIME]{load_t}")
    print(f"[AGG_TIME]{agg_t}")
    print(f"[ML_TIME]{ml_t}")
    print(f"[ACCURACY]{accuracy}")


if __name__ == "__main__":
    main()
