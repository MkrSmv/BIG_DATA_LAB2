$ErrorActionPreference = "Stop"

$FILE        = "nyc_taxi_2023.parquet"
$HDFS_PATH   = "/nyc_taxi_2023.parquet"
$METRIC_TAGS = @("[ROWS]","[TIME]","[MEMORY]","[LOAD_TIME]","[AGG_TIME]","[ML_TIME]","[ACCURACY]")

# -------------------------------------------------------------------
# Start run log (captures everything printed to console)
# -------------------------------------------------------------------
if (-Not (Test-Path "logs")) { New-Item -ItemType Directory -Path "logs" | Out-Null }
$RunTs = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = "logs\run_$RunTs.log"
Start-Transcript -Path $RunLog -Append
Write-Host "Run log: $RunLog" -ForegroundColor Magenta

# -------------------------------------------------------------------
# Install build tools + Python packages inside Spark containers
# -------------------------------------------------------------------
function Install-ClusterDeps {
    param([string[]]$Workers)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    # apk add on each worker — show only progress lines so user sees it's alive
    foreach ($w in $Workers) {
        Write-Host "  [apk] $w ..." -ForegroundColor DarkCyan
        docker exec $w apk add --no-cache make automake gcc g++ python3-dev linux-headers 2>&1 `
            | Where-Object { $_ -match "(Installing|Upgrading|OK:|ERROR|error)" } `
            | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
        Write-Host "  [apk] $w done." -ForegroundColor DarkCyan
    }

    Write-Host "  [apk] spark-master ..." -ForegroundColor DarkCyan
    docker exec spark-master apk add --no-cache make automake gcc g++ python3-dev linux-headers py3-pip 2>&1 `
        | Where-Object { $_ -match "(Installing|Upgrading|OK:|ERROR|error)" } `
        | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }
    Write-Host "  [apk] spark-master done." -ForegroundColor DarkCyan

    # IMPORTANT: use /usr/bin/python3 -m pip (not pip3) so packages install
    # for the exact Python interpreter that spark-submit /PYSPARK_PYTHON uses.
    Write-Host "  [pip] Installing psutil numpy on spark-master ..." -ForegroundColor DarkCyan
    docker exec spark-master /usr/bin/python3 -m pip install psutil numpy 2>&1 `
        | Where-Object { $_ -match "(Collecting|Installing|Successfully|already|ERROR|error)" } `
        | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkGray }

    # Verify psutil is importable by the same Python Spark will use
    $check = docker exec spark-master /usr/bin/python3 -c "import psutil; print('psutil OK')" 2>&1
    if ("$check" -match "psutil OK") {
        Write-Host "  [pip] psutil verified OK." -ForegroundColor Green
    } else {
        Write-Host "  [pip] ERROR: psutil not importable! Output: $check" -ForegroundColor Red
    }

    $ErrorActionPreference = $prev
}


# -------------------------------------------------------------------
# Upload dataset parquet to HDFS
# -------------------------------------------------------------------
function Upload-Dataset {
    Write-Host "Copying dataset to NameNode ..." -ForegroundColor Yellow
    docker cp ".\$FILE" namenode:/$FILE
    Write-Host "Uploading to HDFS ($HDFS_PATH) ..."
    docker exec namenode hdfs dfs -put -f /$FILE $HDFS_PATH
    docker exec namenode hdfs dfs -ls $HDFS_PATH
}

# -------------------------------------------------------------------
# Run spark-submit and stream output; parse metric lines
# -------------------------------------------------------------------
function Run-SparkJob {
    param(
        [string]$Label,
        [string]$OutFile,
        [string]$LogFile,
        [string]$Optimized
    )
    Write-Host " -> $Label" -ForegroundColor Green
    Write-Host "    (may take 2-5 min, output streams below)" -ForegroundColor DarkGray

    # Clear previous log inside container so each experiment gets a clean file
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    docker exec spark-master sh -c "rm -f /tmp/spark_app.log" *>&1 | Out-Null
    $ErrorActionPreference = $prev

    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    $metrics = [System.Collections.Generic.List[string]]::new()

    docker exec spark-master `
        /spark/bin/spark-submit `
        --master spark://spark-master:7077 `
        /tmp/spark_app.py $Optimized 2>&1 | ForEach-Object {
            $ln = $_.ToString().Trim()
            $hit = $false
            foreach ($tag in $METRIC_TAGS) {
                if ($ln.StartsWith($tag)) { $hit = $true; break }
            }
            if ($hit) {
                $metrics.Add($ln)
                Write-Host "    [METRIC] $ln" -ForegroundColor Cyan
            } elseif ($ln -ne "") {
                Write-Host "    $ln" -ForegroundColor DarkGray
            }
        }

    $ErrorActionPreference = $prev

    # Copy log file from container to host logs/ directory
    $ErrorActionPreference = "Continue"
    docker cp spark-master:/tmp/spark_app.log "logs\$LogFile" 2>&1 | Out-Null
    if (Test-Path "logs\$LogFile") {
        Write-Host "    Log saved: logs\$LogFile" -ForegroundColor DarkCyan
    } else {
        Write-Host "    WARNING: log file not found in container" -ForegroundColor Yellow
    }
    $ErrorActionPreference = $prev

    if ($metrics.Count -gt 0) {
        $metrics | Out-File -Encoding ASCII $OutFile
        Write-Host "    Saved $($metrics.Count) metrics to $OutFile" -ForegroundColor Green
    } else {
        Write-Host "    WARNING: no metrics captured for '$Label'" -ForegroundColor Yellow
        "" | Out-File -Encoding ASCII $OutFile
    }
}

# ===================================================================
#  0. Prepare
# ===================================================================
Write-Host ""
Write-Host "=============================" -ForegroundColor Cyan
Write-Host " 0. Setup"                     -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

if (Test-Path "results\res_*.txt")        { Remove-Item "results\res_*.txt" }
if (Test-Path "charts\comparison.png")    { Remove-Item "charts\comparison.png" }
if (Test-Path "charts\stages_breakdown.png") { Remove-Item "charts\stages_breakdown.png" }

# Ensure output dirs exist
foreach ($d in @("results", "charts", "logs")) {
    if (-Not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
}

if (-Not (Test-Path $FILE)) {
    Write-Host "Dataset not found. Downloading ..." -ForegroundColor Yellow
    pip install -r requirements.txt
    python .\app\create_dataset.py
}

# ===================================================================
#  1. Experiments -- 1 DataNode
# ===================================================================
Write-Host ""
Write-Host "=============================" -ForegroundColor Cyan
Write-Host " 1. Experiments: 1 DataNode"  -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
docker-compose -f config\docker-compose-1d.yml up -d --remove-orphans
$ErrorActionPreference = "Stop"

Write-Host "Waiting for 1DN cluster (30s) ..."
Start-Sleep -Seconds 30

docker exec namenode hdfs dfsadmin -safemode leave

Upload-Dataset

docker cp .\app\spark_app.py spark-master:/tmp/spark_app.py

Install-ClusterDeps -Workers @("spark-worker-1")

Run-SparkJob -Label "1DN Normal"    -OutFile "results\res_one_node.txt"     -LogFile "1dn_normal.log"    -Optimized "False"
Run-SparkJob -Label "1DN Optimized" -OutFile "results\res_one_node_opt.txt" -LogFile "1dn_optimized.log" -Optimized "True"

Write-Host "Stopping 1DN cluster ..."
$ErrorActionPreference = "Continue"
docker-compose -f config\docker-compose-1d.yml down --remove-orphans
$ErrorActionPreference = "Stop"

# ===================================================================
#  2. Experiments -- 3 DataNodes
# ===================================================================
Write-Host ""
Write-Host "=============================" -ForegroundColor Cyan
Write-Host " 2. Experiments: 3 DataNodes" -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
docker-compose -f config\docker-compose-3d.yml up -d --remove-orphans
$ErrorActionPreference = "Stop"

Write-Host "Waiting for 3DN cluster (40s) ..."
Start-Sleep -Seconds 40

docker exec namenode hdfs dfsadmin -safemode leave

Upload-Dataset

docker cp .\app\spark_app.py spark-master:/tmp/spark_app.py

Install-ClusterDeps -Workers @("spark-worker-1", "spark-worker-2", "spark-worker-3")

Run-SparkJob -Label "3DN Normal"    -OutFile "results\res_three_node.txt"     -LogFile "3dn_normal.log"    -Optimized "False"
Run-SparkJob -Label "3DN Optimized" -OutFile "results\res_three_node_opt.txt" -LogFile "3dn_optimized.log" -Optimized "True"

Write-Host "Stopping 3DN cluster ..."
$ErrorActionPreference = "Continue"
docker-compose -f config\docker-compose-3d.yml down --remove-orphans
$ErrorActionPreference = "Stop"

# ===================================================================
#  3. Visualize results
# ===================================================================
Write-Host ""
Write-Host "=============================" -ForegroundColor Cyan
Write-Host " 3. Generating charts"         -ForegroundColor Cyan
Write-Host "=============================" -ForegroundColor Cyan

python .\app\plot_result.py

Write-Host ""
Write-Host "All done! Open charts/comparison.png and charts/stages_breakdown.png." -ForegroundColor Green

Stop-Transcript
