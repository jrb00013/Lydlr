# This file is part of the Lydlr project.
#
# Copyright (C) 2025 Joseph Ronald Black
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

"""
System Monitor Utility
- Monitor system resources
- Track compression performance
- Generate performance reports
"""

import psutil
import torch
import time
from typing import Dict, List
from collections import deque
import json
from pathlib import Path


class SystemMonitor:
    """Monitors system resources and performance"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # Resource tracking
        self.cpu_history = deque(maxlen=history_size)
        self.memory_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.network_history = deque(maxlen=history_size)
        
        # Performance tracking
        self.compression_history = deque(maxlen=history_size)
        self.latency_history = deque(maxlen=history_size)
        
        # Timestamps
        self.start_time = time.time()
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        stats = {
            'cpu': {
                'percent': psutil.cpu_percent(interval=0.1),
                'count': psutil.cpu_count(),
                'freq': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'memory': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available,
                'percent': psutil.virtual_memory().percent,
                'used': psutil.virtual_memory().used
            },
            'disk': {
                'total': psutil.disk_usage('/').total,
                'used': psutil.disk_usage('/').used,
                'free': psutil.disk_usage('/').free,
                'percent': psutil.disk_usage('/').percent
            },
            'network': {
                'bytes_sent': psutil.net_io_counters().bytes_sent,
                'bytes_recv': psutil.net_io_counters().bytes_recv,
                'packets_sent': psutil.net_io_counters().packets_sent,
                'packets_recv': psutil.net_io_counters().packets_recv
            }
        }
        
        # GPU stats if available
        if torch.cuda.is_available():
            stats['gpu'] = {
                'available': True,
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'memory_reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
                'memory_free': (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_reserved()) / 1024**3  # GB
            }
        else:
            stats['gpu'] = {'available': False}
        
        return stats
    
    def record_compression(self, compression_ratio: float, latency_ms: float):
        """Record compression performance"""
        self.compression_history.append({
            'compression_ratio': compression_ratio,
            'latency_ms': latency_ms,
            'timestamp': time.time()
        })
        self.latency_history.append(latency_ms)
    
    def record_system_state(self):
        """Record current system state"""
        stats = self.get_system_stats()
        
        self.cpu_history.append({
            'percent': stats['cpu']['percent'],
            'timestamp': time.time()
        })
        
        self.memory_history.append({
            'percent': stats['memory']['percent'],
            'used_gb': stats['memory']['used'] / 1024**3,
            'timestamp': time.time()
        })
        
        if stats['gpu']['available']:
            self.gpu_history.append({
                'memory_allocated_gb': stats['gpu']['memory_allocated'],
                'memory_reserved_gb': stats['gpu']['memory_reserved'],
                'timestamp': time.time()
            })
        
        self.network_history.append({
            'bytes_sent': stats['network']['bytes_sent'],
            'bytes_recv': stats['network']['bytes_recv'],
            'timestamp': time.time()
        })
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        if not self.compression_history:
            return {}
        
        compression_ratios = [h['compression_ratio'] for h in self.compression_history]
        latencies = list(self.latency_history)
        
        return {
            'compression': {
                'avg_ratio': sum(compression_ratios) / len(compression_ratios),
                'min_ratio': min(compression_ratios),
                'max_ratio': max(compression_ratios),
                'total_samples': len(compression_ratios)
            },
            'latency': {
                'avg_ms': sum(latencies) / len(latencies),
                'min_ms': min(latencies),
                'max_ms': max(latencies),
                'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
            },
            'uptime_seconds': time.time() - self.start_time
        }
    
    def export_report(self, filepath: str):
        """Export performance report to JSON"""
        report = {
            'system_stats': self.get_system_stats(),
            'performance_summary': self.get_performance_summary(),
            'compression_history': list(self.compression_history),
            'latency_history': list(self.latency_history),
            'cpu_history': list(self.cpu_history),
            'memory_history': list(self.memory_history),
            'gpu_history': list(self.gpu_history),
            'network_history': list(self.network_history),
            'export_timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath

