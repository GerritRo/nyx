import jax
import chex
import jax.numpy as jnp
import numpy as np

from enum import Enum
from dataclasses import field
from typing import Dict, Any, List, Callable, NamedTuple, Optional, Union


@chex.dataclass
class InstrumentQuery:
    centers: jnp.array
    hp_pixels: jnp.array
    hp_weight: jnp.array
    weight: jnp.array
    grid: jnp.array
    values: jnp.array
    bandpass: jnp.array


@chex.dataclass
class AtmosphereQuery:
    YiXi: jnp.array
    tau: jnp.array
    scattering: jnp.array
    extinction: jnp.array


@chex.dataclass
class DiffuseQuery:
    flux_map: jnp.array


@chex.dataclass
class CatalogQuery:
    sec_Z: jnp.array
    image_coords: jnp.array
    flux_values: jnp.array
    flux_map: jnp.array


# Scene summarization
@chex.dataclass
class SceneComponents:
    """Container for all render queries components."""
    # Emitter queries
    diffuse: List[DiffuseQuery] = field(default_factory=list)
    catalog: List[CatalogQuery] = field(default_factory=list)
    
    # Atmosphere query
    atmosphere: AtmosphereQuery = None
    
    # Instrument query
    instrument: InstrumentQuery = None

class ComponentType(Enum):
    """Types of scene components."""
    ATMOSPHERE = "atmosphere"
    INSTRUMENT = "instrument"
    DIFFUSE = "diffuse"
    CATALOG = "catalog"


class ParameterSpec(NamedTuple):
    """Specification for a parameter."""
    shape: tuple
    initial_value: Union[float, np.ndarray, jnp.ndarray]
    dtype: type = jnp.float32
    description: str = ""
    bounds: Optional[tuple] = None  # (min, max) bounds for optimization


class Scene:
    """
    A parameterized scene that can be realized with different parameters.
    """
    
    def __init__(self):
        self._generators = {}
        self._generator_types = {}
        self._param_specs = {}
        self._param_slices = {}
        self._initial_values = {}
        self._total_params = 0
        
    def add_generator(self, 
                     name: str,
                     component_type: ComponentType,
                     generator_fn: Callable,
                     param_specs: Dict[str, ParameterSpec]):
        """
        Add a generator function for a scene component.
        
        Args:
            name: Unique name for this generator (e.g., 'main_atmosphere', 'diffuse_1')
            component_type: Type of component (determines how it's treated in SceneComponents)
            generator_fn: Function that takes parameters dict and returns component Query object
            param_specs: Dictionary of parameter specifications with initial values
        """
        if name in self._generators:
            raise ValueError(f"Generator '{name}' already exists")
            
        self._generators[name] = generator_fn
        self._generator_types[name] = component_type
        
        # Calculate parameter slicing and store initial values
        start_idx = self._total_params
        component_slices = {}
        component_initial = {}
        
        for param_name, spec in param_specs.items():
            param_size = np.prod(spec.shape)
            end_idx = start_idx + param_size
            component_slices[param_name] = (start_idx, end_idx, spec.shape)
            
            # Store initial values
            initial = spec.initial_value
            if np.isscalar(initial):
                initial = jnp.full(spec.shape, initial, dtype=spec.dtype)
            else:
                initial = jnp.asarray(initial, dtype=spec.dtype)
            
            if initial.shape != spec.shape:
                raise ValueError(f"Initial value shape {initial.shape} doesn't match "
                               f"specified shape {spec.shape} for {name}.{param_name}")
            
            component_initial[param_name] = initial
            start_idx = end_idx
            
        self._param_slices[name] = component_slices
        self._param_specs[name] = param_specs
        self._initial_values[name] = component_initial
        self._total_params = start_idx
    
    @property
    def parameters(self):
        """Return a structured view of all parameters."""
        param_info = {}
        for name, specs in self._param_specs.items():
            param_info[name] = {
                'type': self._generator_types[name].value,
                'parameters': {}
            }
            for param_name, spec in specs.items():
                slice_info = self._param_slices[name][param_name]
                param_info[name]['parameters'][param_name] = {
                    'shape': spec.shape,
                    'dtype': spec.dtype,
                    'description': spec.description,
                    'bounds': spec.bounds,
                    'flat_indices': (slice_info[0], slice_info[1])
                }
        return param_info
    
    @property
    def n_parameters(self):
        """Total number of parameters."""
        return self._total_params
    
    def get_initial_parameters(self):
        """Get the initial parameter values as a flat array."""
        params = jnp.zeros(self.n_parameters)
        
        for name, initial_dict in self._initial_values.items():
            for param_name, initial_value in initial_dict.items():
                start, end, shape = self._param_slices[name][param_name]
                flat_initial = initial_value.flatten()
                params = params.at[start:end].set(flat_initial)
                
        return params
    
    def unpack_parameters(self, flat_params):
        """Unpack flat parameter array into structured dict."""
        unpacked = {}
        
        for name, slices in self._param_slices.items():
            unpacked[name] = {}
            for param_name, (start, end, shape) in slices.items():
                param_values = flat_params[start:end].reshape(shape)
                unpacked[name][param_name] = param_values
                
        return unpacked
    
    def pack_parameters(self, params_dict):
        """Pack structured parameter dict into flat array."""
        flat_params = jnp.zeros(self.n_parameters)
        
        for name, params in params_dict.items():
            for param_name, param_value in params.items():
                if name in self._param_slices and param_name in self._param_slices[name]:
                    start, end, shape = self._param_slices[name][param_name]
                    flat_params = flat_params.at[start:end].set(param_value.flatten())
                    
        return flat_params
        
    def realize(self, parameters):
        """
        Realize the scene with given parameters.
        
        Args:
            parameters: Flat array of parameters or structured dict
            
        Returns:
            SceneComponents object
        """
        # Convert to structured dict if needed
        if isinstance(parameters, (jnp.ndarray, np.ndarray)):
            params_dict = self.unpack_parameters(parameters)
        else:
            params_dict = parameters
            
        # Generate components by type
        components = {
            'atmosphere': None,
            'instrument': None,
            'diffuse': [],
            'catalog': []
        }
        
        for name, generator in self._generators.items():
            component_type = self._generator_types[name]
            component_params = params_dict.get(name, {})
            generated = generator(component_params)
            
            if component_type == ComponentType.DIFFUSE:
                components['diffuse'].append(generated)
            elif component_type == ComponentType.CATALOG:
                components['catalog'].append(generated)
            elif component_type == ComponentType.ATMOSPHERE:
                components['atmosphere'] = generated
            elif component_type == ComponentType.INSTRUMENT:
                components['instrument'] = generated
        
        # Add dummy emitters if lists are empty (for JAX compatibility)
        if not components['diffuse']:
            npix, n_wvl = components['atmosphere'].extinction.shape
            dummy_diffuse = DiffuseQuery(flux_map=jnp.zeros((npix, n_wvl)))
            components['diffuse'].append(dummy_diffuse)
        
        if not components['catalog']:
            n_wvl = components['atmosphere'].tau.shape[0]
            dummy_catalog = CatalogQuery(
                sec_Z=jnp.zeros((1, 1)),
                image_coords=jnp.zeros((1, 2)),
                flux_values=jnp.zeros((1, n_wvl)),
                flux_map=jnp.zeros((components['atmosphere'].extinction.shape[0], n_wvl))
            )
            components['catalog'].append(dummy_catalog)
        
        # Create SceneComponents
        return SceneComponents(
            diffuse=components['diffuse'],
            catalog=components['catalog'],
            instrument=components['instrument'],
            atmosphere=components['atmosphere']
        )
    
    def print_parameters(self, values=None, format='table'):
        """
        Print parameters in a nice table format.
        
        Args:
            values: Optional flat parameter array to show current values
            format: 'table' for tabular format
        """
        if values is not None:
            current_params = self.unpack_parameters(values)
        else:
            current_params = None
            
        if format == 'table':
            self._print_table(current_params)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _print_table(self, current_params=None):
        """Print parameters in table format."""
        # Calculate column widths
        headers = ['Component', 'Type', 'Parameter', 'Shape', 'Initial', 'Current', 'Bounds', 'Description']
        
        rows = []
        for comp_name, specs in self._param_specs.items():
            comp_type = self._generator_types[comp_name].value
            for param_name, spec in specs.items():
                initial_val = self._initial_values[comp_name][param_name]
                
                # Format initial value
                if initial_val.size == 1:
                    initial_str = f"{float(initial_val[0]):.4f}"
                else:
                    initial_str = f"-"
                
                # Format current value
                if current_params and comp_name in current_params:
                    current_val = current_params[comp_name][param_name]
                    if current_val.size == 1:
                        current_str = f"{float(current_val[0]):.4f}"
                    else:
                        mean_val = float(jnp.mean(current_val))
                        std_val = float(jnp.std(current_val))
                        current_str = f"μ={mean_val:.3f}, σ={std_val:.3f}"
                else:
                    current_str = "-"
                
                # Format bounds
                bounds_str = f"{spec.bounds}" if spec.bounds else "-"
                
                rows.append([
                    comp_name,
                    comp_type,
                    param_name,
                    str(spec.shape),
                    initial_str,
                    current_str,
                    bounds_str,
                    spec.description[:30] + "..." if len(spec.description) > 30 else spec.description
                ])
        
        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Print table
        def print_row(cells, widths):
            row_str = "│ "
            for cell, width in zip(cells, widths):
                row_str += str(cell).ljust(width) + " │ "
            print(row_str)
        
        # Print header
        print("┌" + "─" * (sum(col_widths) + 3 * len(col_widths) - 1) + "┐")
        print_row(headers, col_widths)
        print("├" + "─" * (sum(col_widths) + 3 * len(col_widths) - 1) + "┤")
        
        # Print rows
        for row in rows:
            print_row(row, col_widths)
        
        print("└" + "─" * (sum(col_widths) + 3 * len(col_widths) - 1) + "┘")
        
        # Print summary
        print(f"\nTotal parameters: {self.n_parameters}")