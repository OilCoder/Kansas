"""Verifies GPU availability and performs basic operation tests without modifying system settings. Provides non-intrusive GPU capability checking, device information reporting, and compatibility validation for neural network operations."""

import os
import logging
import tensorflow as tf
import sys

logger = logging.getLogger(__name__)

def check_gpu_availability(memory_limit=None, gpu_memory_fraction=0.9):
    """
    Verifies GPU availability without modifying settings.
    Only checks and reports rather than configuring.
    
    Parameters:
    -----------
    memory_limit : int, optional
        Not used, kept for API compatibility.
    gpu_memory_fraction : float, optional
        Not used, kept for API compatibility.
    
    Returns:
    --------
    bool
        True if GPU is available and working, False otherwise.
    """
    logger.info("Step GPU-1: Verificando dispositivos GPU...")
    
    # List physical devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        logger.error("⚠️ ERROR: No se encontró GPU. Este proyecto requiere ejecución en GPU.")
        logger.error("⚠️ ERROR: Terminando ejecución.")
        return False
    
    try:
        # Verify basic operations on GPU without modifying memory
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        
        logger.info("✓ Verificación de operaciones GPU completada con éxito")
        
        # Log GPU properties without attempting to modify them
        for i, gpu in enumerate(gpus):
            try:
                details = tf.config.experimental.get_device_details(gpu)
                compute_capability = details.get('compute_capability', 'Unknown')
                if isinstance(compute_capability, tuple):
                    compute_capability = f"{compute_capability[0]}.{compute_capability[1]}"
                logger.info(f"GPU #{i}: {details.get('device_name', gpu.name)}")
                logger.info(f"  Modelo: {details.get('device_name', 'Unknown')}")
                logger.info(f"  Compute Capability: {compute_capability}")
                logger.info(f"  Dispositivo TF: {tf.test.gpu_device_name()}")
            except Exception as e:
                logger.warning(f"Could not get details for GPU #{i}: {str(e)}")
        
        logger.info("✓ Configuración de GPU verificada correctamente")
        return True
        
    except RuntimeError as e:
        logger.error(f"⚠️ ERROR en la verificación de GPU: {e}")
        return False
