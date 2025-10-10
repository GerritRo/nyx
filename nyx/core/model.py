import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, SkyOffsetFrame

from .config import get_grid_dim
from .scene import Scene, ComponentType
from typing import Optional, Dict, Any, List, Tuple, Protocol

class AtmosphereProtocol(Protocol):
    """Protocol for atmosphere objects"""
    def get_generator(self, wavelengths, observation):
        """Return JAX generator function and parameter specs"""

class InstrumentProtocol(Protocol):
    """Protocol for instrument objects"""
    def get_generator(self, wavelengths,  observation):
        """Return JAX generator function and parameter specs"""

class EmitterProtocol(Protocol):
    """Protocol for emitter objects"""
    def get_generator(self, wavelengths, observation):
        """Return JAX generator function, parameter specs, and component type"""

class Observation:
    def __init__(self, location, obstime, target, rotation, fov=None, **kwargs):
        self.AltAz = AltAz(obstime=obstime, location=location)
        self.target = target.transform_to(self.AltAz)
        self.frame = self.target.skyoffset_frame(rotation=rotation)
        
        if fov != None:
            self.fov = fov.to(u.radian)
        else:
            self.fov = None

    def set_fov(self, fov):
        self.fov = fov.to(u.radian)
    
    def calculate_fov(self, instrument, maxshift=600*u.arcsec):
        raise NotImplementedError

    def get_eval_coordinates(self, altaz=True):
        grid = np.linspace(-self.fov, self.fov, get_grid_dim())
        X, Y = np.meshgrid(grid, grid)
        if altaz == True:
            return SkyCoord(X, Y, unit='rad', frame=self.frame).transform_to(self.AltAz)
        else:
            return grid

class Model:
    """
    Main model class - agnostic to specific implementations
    Just needs objects that follow the protocols
    """
    
    def __init__(self, 
                 instrument: InstrumentProtocol,
                 atmosphere: AtmosphereProtocol,
                 emitters: List[EmitterProtocol]):
        
        self.instrument = instrument
        self.atmosphere = atmosphere
        self.emitters = emitters
        
    def query(self, observation: Any) -> 'SceneResult':        
        # Build scene
        scene = Scene()
        
        # Add atmosphere
        atmo_gen, atmo_specs = self.atmosphere.get_generator(observation)
        scene.add_generator(
            name="atmosphere",
            component_type=ComponentType.ATMOSPHERE,
            generator_fn=atmo_gen,
            param_specs=atmo_specs
        )
        
        # Add instrument  
        inst_gen, inst_specs = self.instrument.get_generator(observation)
        scene.add_generator(
            name="instrument",
            component_type=ComponentType.INSTRUMENT,
            generator_fn=inst_gen,
            param_specs=inst_specs
        )
        
        # Add emitters
        for i, emitter in enumerate(self.emitters):
            em_gen, em_specs, em_type = emitter.get_generator(observation)
            scene.add_generator(
                name=f"emitter_{i}",
                component_type=em_type,
                generator_fn=em_gen,
                param_specs=em_specs
            )
        
        return scene
