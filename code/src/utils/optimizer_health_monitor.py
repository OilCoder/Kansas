"""
🛡️ Optimizer Health Monitor - Sistema de Monitoreo Ultra-Estable
================================================================

Monitoreo continuo de salud del sistema específico para el optimizer
con gestión inteligente de workers paralelos.

Autor: Sistema de IA
Versión: 1.0 - Ultra-Estable
"""

import psutil
import time
import gc
import tensorflow as tf
from src.utils.memory_manager import get_memory_usage, clean_memory_for_trial


class OptimizerHealthMonitor:
    """Monitoreo continuo de salud específico del optimizer."""
    
    def __init__(self):
        self.health_history = []
        self.last_check_time = time.time()
        
    def check_system_health(self):
        """Monitoreo continuo de métricas del sistema."""
        try:
            # Memoria RAM
            ram = psutil.virtual_memory()
            ram_percent = ram.percent
            
            # Memoria GPU
            gpu_memory_percent = self._get_gpu_memory_usage()
            
            # CPU y temperatura
            cpu_percent = psutil.cpu_percent(interval=1)
            temperature = self._get_system_temperature()
            
            health_status = {
                'timestamp': time.time(),
                'ram_percent': ram_percent,
                'gpu_memory_percent': gpu_memory_percent,
                'cpu_percent': cpu_percent,
                'temperature': temperature,
                'status': self._determine_health_status(ram_percent, gpu_memory_percent, temperature)
            }
            
            self.health_history.append(health_status)
            return health_status
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _get_gpu_memory_usage(self):
        """Obtener uso de memoria GPU."""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                return (gpu_details['current'] / gpu_details['peak']) * 100
            return 0.0
        except:
            return 0.0
    
    def _get_system_temperature(self):
        """Obtener temperatura del sistema."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                return max([temp.current for sensor in temps.values() for temp in sensor])
            return 50.0  # Default safe temperature
        except:
            return 50.0
    
    def _determine_health_status(self, ram_percent, gpu_percent, temperature):
        """Determinar estado de salud del sistema."""
        if ram_percent > 95 or gpu_percent > 95 or temperature > 80:
            return 'critical'
        elif ram_percent > 85 or gpu_percent > 90 or temperature > 75:
            return 'warning'
        elif ram_percent > 70 or gpu_percent > 80:
            return 'medium'
        else:
            return 'healthy'


class OptimizerWorkerManager:
    """Gestión inteligente de workers específicos del optimizer."""
    
    def __init__(self, initial_workers=6):
        self.current_workers = initial_workers
        self.max_workers = 8
        self.min_workers = 1
        self.failed_workers = []
        
    def adjust_workers_based_on_health(self, health_status):
        """Ajuste automático de workers basado en salud del sistema."""
        status = health_status.get('status', 'healthy')
        
        if status == 'critical':
            self.scale_down('critical_memory')
            return self.current_workers
        elif status == 'warning':
            self.scale_down('high_memory')
            return self.current_workers
        elif status == 'healthy' and self.current_workers < self.max_workers:
            self.scale_up('good_health')
            return self.current_workers
        
        return self.current_workers
    
    def scale_up(self, reason):
        """Aumentar número de workers."""
        if self.current_workers < self.max_workers:
            self.current_workers = min(self.current_workers + 1, self.max_workers)
            print(f"   📈 Workers aumentados a {self.current_workers} (razón: {reason})")
    
    def scale_down(self, reason):
        """Reducir número de workers."""
        if self.current_workers > self.min_workers:
            self.current_workers = max(self.current_workers - 1, self.min_workers)
            print(f"   📉 Workers reducidos a {self.current_workers} (razón: {reason})")
    
    def restart_failed_workers(self):
        """Reiniciar workers fallidos."""
        if self.failed_workers:
            print(f"   🔄 Reiniciando {len(self.failed_workers)} workers fallidos")
            self.failed_workers.clear()
            gc.collect() 