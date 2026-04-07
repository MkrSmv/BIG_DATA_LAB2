import urllib.request
import sys
import os

# NYC Yellow Taxi Trip Data — January 2023
# ~3 million rows, 19 columns: timestamps, ints, floats, categoricals
# Source: NYC Taxi & Limousine Commission
DATASET_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
OUTPUT_FILE = "nyc_taxi_2023.parquet"


def progress_hook(count, block_size, total_size):
    if total_size > 0:
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(f"\r  Downloading... {percent}%")
        sys.stdout.flush()


def main():
    if os.path.exists(OUTPUT_FILE):
        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"Dataset already exists: {OUTPUT_FILE} ({size_mb:.1f} MB). Skipping download.")
        return

    print("=" * 55)
    print("  NYC Yellow Taxi Dataset Downloader")
    print("=" * 55)
    print(f"  URL   : {DATASET_URL}")
    print(f"  Output: {OUTPUT_FILE}")
    print()

    try:
        urllib.request.urlretrieve(DATASET_URL, OUTPUT_FILE, reporthook=progress_hook)
        print()  # newline after progress

        size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
        print(f"  Download complete! File size: {size_mb:.1f} MB")
        print()
        print("  Dataset overview:")
        print("  - Rows   : ~3,000,000 (January 2023)")
        print("  - Columns: 19 (VendorID, datetime, distance, fare, tip, etc.)")
        print("  - Types  : integer, float, timestamp, string (categorical)")
        print("  - Format : Apache Parquet (columnar, efficient for Spark)")
        print()
        print("  Column types:")
        print("    Categorical : VendorID, RatecodeID, payment_type, store_and_fwd_flag")
        print("    Numeric     : trip_distance, fare_amount, tip_amount, total_amount")
        print("    Integer     : passenger_count, PULocationID, DOLocationID")
        print("    Timestamp   : tpep_pickup_datetime, tpep_dropoff_datetime")
        print()
        print("  Ready to upload to HDFS!")

    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        sys.exit(1)


if __name__ == "__main__":
    main()
