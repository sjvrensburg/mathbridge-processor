#!/usr/bin/env python3
"""
Performance benchmarking script for mathbridge-processor.
Tests different combinations of batch size and worker count to find optimal settings.
"""

import json
import time
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

def run_benchmark(batch_size: int, max_workers: int, max_records: int = 500) -> Dict:
    """Run a single benchmark test."""
    output_dir = f"bench_b{batch_size}_w{max_workers}"
    
    # Clean up any existing output
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)
    
    # Build command
    cmd = [
        "python", "-m", "mathbridge_processor.cli", "process",
        "--max-records", str(max_records),
        "--batch-size", str(batch_size),
        "--max-workers", str(max_workers),
        "--output", output_dir,
        "--verbose"
    ]
    
    print(f"Testing batch_size={batch_size}, max_workers={max_workers}, records={max_records}")
    
    # Time the execution
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return {
                "batch_size": batch_size,
                "max_workers": max_workers,
                "records": max_records,
                "duration": float('inf'),
                "records_per_second": 0,
                "error": result.stderr
            }
        
        duration = end_time - start_time
        records_per_second = max_records / duration
        
        # Parse output for additional stats
        try:
            output_data = json.loads(result.stdout.split('\n')[-2])  # Get last JSON line
            cache_stats = output_data.get('cache_stats', {})
        except:
            cache_stats = {}
        
        # Clean up output directory
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        
        return {
            "batch_size": batch_size,
            "max_workers": max_workers,
            "records": max_records,
            "duration": duration,
            "records_per_second": records_per_second,
            "cache_hits": cache_stats.get('successful_cached', 0),
            "total_cached": cache_stats.get('total_expressions_cached', 0),
            "error": None
        }
        
    except subprocess.TimeoutExpired:
        return {
            "batch_size": batch_size,
            "max_workers": max_workers,
            "records": max_records,
            "duration": float('inf'),
            "records_per_second": 0,
            "error": "TIMEOUT"
        }
    except Exception as e:
        return {
            "batch_size": batch_size,
            "max_workers": max_workers,
            "records": max_records,
            "duration": float('inf'),
            "records_per_second": 0,
            "error": str(e)
        }

def main():
    print("=== MathBridge Processor Performance Benchmarking ===")
    
    # Test configurations
    batch_sizes = [100, 500, 1000, 2000, 5000]
    worker_counts = [1, 2, 4, 8, 12, 16, 20]  # Test up to system max
    test_records = 500  # Use 500 records for comprehensive testing
    
    results = []
    total_tests = len(batch_sizes) * len(worker_counts)
    current_test = 0
    
    print(f"Running {total_tests} benchmark tests with {test_records} records each...")
    print("This will take approximately 15-30 minutes...")
    
    # Run benchmarks
    for batch_size in batch_sizes:
        for max_workers in worker_counts:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}]", end=" ")
            
            result = run_benchmark(batch_size, max_workers, test_records)
            results.append(result)
            
            if result['error']:
                print(f"FAILED: {result['error']}")
            else:
                print(f"SUCCESS: {result['duration']:.2f}s ({result['records_per_second']:.1f} rec/s)")
    
    # Save raw results
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create analysis dataframe
    df = pd.DataFrame(results)
    df = df[df['error'].isna()]  # Filter out errors
    
    if df.empty:
        print("No successful benchmark results!")
        return
    
    # Analysis
    print("\n=== PERFORMANCE ANALYSIS ===")
    
    # Find top performing configurations
    top_configs = df.nlargest(5, 'records_per_second')
    print("\nTop 5 performing configurations:")
    for idx, row in top_configs.iterrows():
        print(f"  {row['records_per_second']:.1f} rec/s - batch_size={row['batch_size']}, workers={row['max_workers']} ({row['duration']:.2f}s)")
    
    # Optimal configuration
    best = df.loc[df['records_per_second'].idxmax()]
    print(f"\nOPTIMAL CONFIGURATION:")
    print(f"  Batch Size: {best['batch_size']}")
    print(f"  Max Workers: {best['max_workers']}")
    print(f"  Performance: {best['records_per_second']:.1f} records/second")
    print(f"  Duration: {best['duration']:.2f} seconds")
    
    # Extrapolation to 23M records
    records_23m = 23_000_000
    estimated_time = records_23m / best['records_per_second']
    hours = estimated_time // 3600
    minutes = (estimated_time % 3600) // 60
    print(f"  Estimated time for 23M records: {hours:.0f}h {minutes:.0f}m")
    
    # Analysis by batch size
    print("\n=== BATCH SIZE ANALYSIS ===")
    batch_analysis = df.groupby('batch_size')['records_per_second'].agg(['mean', 'max', 'std']).round(1)
    print("Batch Size | Avg Speed | Max Speed | Std Dev")
    print("-" * 45)
    for batch_size, stats in batch_analysis.iterrows():
        print(f"{batch_size:>9} | {stats['mean']:>9.1f} | {stats['max']:>9.1f} | {stats['std']:>7.1f}")
    
    # Analysis by worker count
    print("\n=== WORKER COUNT ANALYSIS ===")
    worker_analysis = df.groupby('max_workers')['records_per_second'].agg(['mean', 'max', 'std']).round(1)
    print("Workers | Avg Speed | Max Speed | Std Dev")
    print("-" * 40)
    for workers, stats in worker_analysis.iterrows():
        print(f"{workers:>7} | {stats['mean']:>9.1f} | {stats['max']:>9.1f} | {stats['std']:>7.1f}")
    
    # Save detailed results
    df.to_csv('benchmark_analysis.csv', index=False)
    print(f"\nDetailed results saved to:")
    print(f"  - benchmark_results.json (raw data)")
    print(f"  - benchmark_analysis.csv (analysis data)")

if __name__ == "__main__":
    main()