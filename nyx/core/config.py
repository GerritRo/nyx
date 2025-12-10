"""
NYX Core Config
"""
import threading
import jax.numpy as jnp
import numpy as np
from contextlib import contextmanager
from typing import Optional, Any, List, Union, Callable

class ConfigParameter:
    """A configuration parameter with validation"""
    
    def __init__(self, name: str, default: Any, validator: Optional[Callable] = None, 
                 description: str = "", immutable: bool = False):
        self.name = name
        self.default = default
        self.validator = validator
        self.description = description
        self.immutable = immutable
        self._value = default
        
    def set(self, value):
        if self.immutable:
            raise ValueError(f"Parameter {self.name} is immutable")
        if self.validator:
            if not self.validator(value):
                raise ValueError(f"Invalid value {value} for parameter {self.name}")
        self._value = value
        
    def get(self):
        return self._value
    
    def reset(self):
        self._value = self.default


class GlobalConfig:
    """Global configuration manager for NYX"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._parameters = {}
        self._contexts = []
        
        # Register default parameters
        self._register_defaults()
        self._initialized = True
        
    def _register_defaults(self):
        """Register default global parameters"""
        
        # Wavelength configuration
        self.register('wavelengths', 
                     default=jnp.linspace(300, 700, 50),
                     validator=lambda x: len(x) > 1 and jnp.all(x > 0),
                     description="Wavelength grid in nm")
        
        # HEALPix configuration
        self.register('healpix_nside',
                     default=16,
                     validator=lambda x: x > 0 and (x & (x-1)) == 0,  # Power of 2
                     description="HEALPix nside parameter")
        
        # Atmospheric configuration
        self.register('airmass_formula',
                     default='kasten_young_1989',
                     validator=lambda x: x in ['plane_parallel', 'kasten_young_1989'],
                     description="Airmass calculation formula")

        # Rendering configuration
        self.register('grid_dim',
                      default=2,
                      description="Image interpolation grid dimension")
        
        self.register('use_jit',
                     default=True,
                     description="Whether to use JAX JIT compilation")
        
        self.register('precision',
                     default='float32',
                     validator=lambda x: x in ['float16', 'float32', 'float64'],
                     description="Numerical precision for calculations")
        
        # Spectral handling
        self.register('spectral_method',
                     default='conserve',
                     validator=lambda x: x in ['linear', 'conserve'],
                     description="Spectral interpolation method")
        
        self.register('spectral_resolution_warning',
                     default=True,
                     description="Warn when spectral resolution is degraded")
        
    def register(self, name: str, default: Any, validator: Optional[Callable] = None,
                description: str = "", immutable: bool = False):
        """Register a new configuration parameter"""
        if name in self._parameters:
            raise ValueError(f"Parameter {name} already registered")
        self._parameters[name] = ConfigParameter(name, default, validator, description, immutable)
        
    def set(self, name: str, value: Any):
        """Set a configuration parameter"""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        self._parameters[name].set(value)
        
    def get(self, name: str) -> Any:
        """Get a configuration parameter"""
        if name not in self._parameters:
            raise KeyError(f"Unknown parameter: {name}")
        return self._parameters[name].get()
    
    def reset(self, name: Optional[str] = None):
        """Reset parameter(s) to default values"""
        if name:
            if name not in self._parameters:
                raise KeyError(f"Unknown parameter: {name}")
            self._parameters[name].reset()
        else:
            for param in self._parameters.values():
                param.reset()
                
    @contextmanager
    def context(self, **kwargs):
        """Context manager for temporary configuration changes"""
        old_values = {}
        for name, value in kwargs.items():
            if name in self._parameters:
                old_values[name] = self.get(name)
                self.set(name, value)
        try:
            yield self
        finally:
            for name, value in old_values.items():
                self.set(name, value)
                
    def summary(self):
        """Print configuration summary"""
        print("NYX Global Configuration")
        print("=" * 60)
        for name, param in self._parameters.items():
            value = param.get()
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                if len(value) > 5:
                    value_str = f"array(shape={value.shape}, dtype={value.dtype})"
                else:
                    value_str = str(value)
            else:
                value_str = str(value)
            print(f"{name:30s}: {value_str}")
            if param.description:
                print(f"{'':30s}  ({param.description})")


# Global configuration instance
_config = GlobalConfig()


# Public API functions
def set_wavelengths(wavelengths: Union[np.ndarray, jnp.ndarray, List[float]]):
    """Set the global wavelength grid"""
    _config.set('wavelengths', jnp.asarray(wavelengths))


def get_wavelengths() -> jnp.ndarray:
    """Get the global wavelength grid"""
    return _config.get('wavelengths')


def set_healpix_nside(nside: int):
    """Set the HEALPix nside parameter"""
    _config.set('healpix_nside', nside)


def get_healpix_nside() -> int:
    """Get the HEALPix nside parameter"""
    return _config.get('healpix_nside')


def set_grid_dim(dim: int):
    """Set the interpolation grid fineness parameter"""
    _config.set('grid_dim', dim)


def get_grid_dim() -> int:
    """Get the interpolation grid fineness parameter"""
    return _config.get('grid_dim')


def set_parameter(name: str, value: Any):
    """Set a global configuration parameter"""
    _config.set(name, value)


def get_parameter(name: str) -> Any:
    """Get a global configuration parameter"""
    return _config.get(name)


def config_context(**kwargs):
    """Context manager for temporary configuration changes"""
    return _config.context(**kwargs)


def config_summary():
    """Print configuration summary"""
    _config.summary()
