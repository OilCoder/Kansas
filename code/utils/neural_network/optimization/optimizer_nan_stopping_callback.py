"""
Implements TensorFlow callback for early detection and stopping of training when NaN or infinite values appear.

Provides patience-based monitoring to prevent wasted computational resources during 
hyperparameter optimization.

• NaNStoppingCallback - Custom TensorFlow callback class
• Early detection of NaN and infinite values in training metrics
• Configurable patience for robust NaN detection
• Integration with TensorFlow training loops
• Prevents wasted computational resources in optimization
• Automatic training termination on numerical instability
"""

import numpy as np
from tensorflow.keras.callbacks import Callback


class NaNStoppingCallback(Callback):
    """Callback para detener entrenamiento si aparecen valores NaN."""
    
    def __init__(self, patience=2, verbose=True):
        """
        Inicializar el callback.
        
        Args:
            patience (int): Número de epochs consecutivos con NaN antes de parar
            verbose (bool): Si mostrar mensajes de debug
        """
        super().__init__()
        self.patience = patience
        self.verbose = verbose
        self.nan_count = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Verificar NaN al final de cada epoch."""
        logs = logs or {}
        
        # Verificar si hay NaN en cualquier métrica
        has_nan = any(np.isnan(value) or np.isinf(value) for value in logs.values())
        
        if has_nan:
            self.nan_count += 1
            if self.verbose:
                print(f"⚠️ NaN detectado en epoch {epoch + 1} (count: {self.nan_count})")
            
            if self.nan_count >= self.patience:
                if self.verbose:
                    print(f"🛑 Parando trial por NaN persistente después de {self.nan_count} epochs")
                self.model.stop_training = True
        else:
            # Reset counter si no hay NaN
            self.nan_count = 0
    
    def on_train_begin(self, logs=None):
        """Reset al inicio del entrenamiento."""
        self.nan_count = 0 