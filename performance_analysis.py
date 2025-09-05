#!/usr/bin/env python3
"""
Advanced performance analysis for optimal configuration determination.
"""

import json
import time
import subprocess
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

def test_scaling_performance():
    """Test performance scaling with different dataset sizes."""
    print("=== SCALING PERFORMANCE ANALYSIS ===")
    
    test_sizes = [500, 1000, 2000, 3000, 5000, 8000]
    batch_size = 1000  # Use optimal batch size
    max_workers = 20   # Use optimal worker count
    
    results = []
    
    for size in test_sizes:
        print(f"Testing {size} records...", end=" ")
        output_dir = f"scale_test_{size}"
        
        # Clean up
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        
        start_time = time.time()
        cmd = [
            "python", "-m", "mathbridge_processor.cli", "process",
            "--max-records", str(size),
            "--batch-size", str(batch_size),
            "--max-workers", str(max_workers),
            "--output", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            end_time = time.time()
            
            if result.returncode != 0:
                print("FAILED")
                continue
            
            duration = end_time - start_time
            records_per_second = size / duration
            
            # Parse cache stats
            try:
                output_data = json.loads(result.stdout.split('\n')[-2])
                cache_stats = output_data.get('cache_stats', {})
                stats = output_data.get('stats', {})
            except:
                cache_stats = {}
                stats = {}
            
            results.append({
                'records': size,
                'duration': duration,
                'records_per_second': records_per_second,
                'cache_size': cache_stats.get('total_expressions_cached', 0),
                'cache_hit_ratio': cache_stats.get('successful_cached', 0) / max(1, cache_stats.get('total_expressions_cached', 1)),
                'speech_success_rate': stats.get('speech_generated', 0) / max(1, stats.get('total_processed', 1))
            })
            
            print(f"{duration:.1f}s ({records_per_second:.1f} rec/s, {cache_stats.get('total_expressions_cached', 0)} cached)")
            
            # Clean up
            if Path(output_dir).exists():
                shutil.rmtree(output_dir)
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    return results

def analyze_resource_usage():
    """Analyze system resource usage during processing."""
    print("\n=== RESOURCE USAGE ANALYSIS ===")
    
    # Test I/O vs CPU bound characteristics
    print("Testing I/O characteristics...")
    
    # Test with different worker counts on fixed dataset
    sizes = [2000]
    worker_counts = [1, 4, 8, 16, 20, 24]
    batch_size = 1000
    
    results = []
    
    for workers in worker_counts:
        print(f"Testing {workers} workers...", end=" ")
        output_dir = f"resource_test_w{workers}"
        
        if Path(output_dir).exists():
            shutil.rmtree(output_dir)
        
        start_time = time.time()
        cmd = [
            "python", "-m", "mathbridge_processor.cli", "process",
            "--max-records", "2000",
            "--batch-size", str(batch_size),
            "--max-workers", str(workers),
            "--output", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            end_time = time.time()
            
            if result.returncode != 0:
                print("FAILED")
                continue
            
            duration = end_time - start_time
            records_per_second = 2000 / duration
            
            results.append({
                'workers': workers,
                'duration': duration,
                'records_per_second': records_per_second,
                'efficiency': records_per_second / workers  # Records per second per worker
            })
            
            print(f"{duration:.1f}s ({records_per_second:.1f} rec/s, {records_per_second/workers:.1f} eff)")
            
            if Path(output_dir).exists():
                shutil.rmtree(output_dir)
                
        except Exception as e:
            print(f"ERROR: {e}")
    
    return results

def estimate_23m_performance(scaling_results):
    """Estimate performance for 23M records based on scaling data."""
    print("\n=== 23 MILLION RECORD ESTIMATION ===")
    
    if not scaling_results:
        print("No scaling data available for estimation")
        return
    
    df = pd.DataFrame(scaling_results)
    
    # Analyze cache efficiency scaling
    print("Cache Analysis:")
    cache_ratios = df['cache_size'] / df['records']
    print(f"  Unique expressions ratio: {cache_ratios.mean():.3f} ± {cache_ratios.std():.3f}")
    print(f"  Cache hit efficiency: {df['cache_hit_ratio'].mean():.3f}")
    
    # Performance scaling analysis
    print("\nPerformance Scaling:")
    largest_test = df.loc[df['records'].idxmax()]
    print(f"  Best observed: {largest_test['records_per_second']:.1f} rec/s on {largest_test['records']} records")
    
    # Conservative estimate based on observed performance
    # Account for potential slowdown with larger datasets
    performance_estimates = {
        "conservative": largest_test['records_per_second'] * 0.8,  # 20% degradation buffer
        "optimistic": largest_test['records_per_second'] * 1.1,   # 10% improvement from better caching
        "realistic": largest_test['records_per_second'] * 0.95    # 5% degradation buffer
    }
    
    records_23m = 23_000_000
    
    print(f"\nEstimates for {records_23m:,} records:")
    for scenario, perf in performance_estimates.items():
        time_seconds = records_23m / perf
        hours = time_seconds // 3600
        minutes = (time_seconds % 3600) // 60
        
        print(f"  {scenario.capitalize():>12}: {perf:>6.1f} rec/s → {hours:>3.0f}h {minutes:>2.0f}m")
    
    # Memory and storage estimates
    avg_record_size = 1000  # Rough estimate in bytes per record
    total_size_gb = (records_23m * avg_record_size) / (1024**3)
    
    print(f"\nStorage Requirements:")
    print(f"  Estimated output size: ~{total_size_gb:.1f} GB")
    print(f"  Recommended free space: {total_size_gb * 2:.1f} GB")
    
    # Optimal configuration
    print(f"\nRECOMMENDED CONFIGURATION:")
    print(f"  --batch-size 1000")
    print(f"  --max-workers 20")
    print(f"  Expected completion: {performance_estimates['realistic']/largest_test['records_per_second']*100:.0f}% of test performance")

def main():
    print("Starting comprehensive performance analysis...")
    print("This will take 15-20 minutes to complete.\n")
    
    # Run scaling tests
    scaling_results = test_scaling_performance()
    
    # Run resource usage tests  
    resource_results = analyze_resource_usage()
    
    # Save results
    with open('scaling_analysis.json', 'w') as f:
        json.dump({
            'scaling_results': scaling_results,
            'resource_results': resource_results
        }, f, indent=2)
    
    # Generate estimates
    estimate_23m_performance(scaling_results)
    
    # Additional recommendations
    print(f"\n=== ADDITIONAL RECOMMENDATIONS ===")
    print(f"1. Monitor disk space - ensure at least 50GB free before starting")
    print(f"2. Run processing during off-peak hours for consistent performance")
    print(f"3. Consider processing in chunks of 1-5M records for safety")
    print(f"4. Use SSD storage for output directory if possible")
    print(f"5. Monitor system temperature during long runs")
    
    print(f"\nAnalysis complete. Results saved to scaling_analysis.json")

if __name__ == "__main__":
    main()