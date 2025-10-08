import jax

@jax.jit
def rayleigh_phase(theta):
    return 1/(4*jax.numpy.pi)*3/4*(1+jax.numpy.cos(theta)**2)

@jax.jit
def henyey_greenstein_phase(theta, g):
    gsq = g**2
    return 1/(4*jax.numpy.pi) * (1-gsq) / (1+gsq-2*g*jax.numpy.cos(theta))**1.5

@jax.jit
def gradation_function(tau, sec_Z, sec_z):
    sec_diff = sec_Z/(sec_z - sec_Z)
    exp_diff = (jax.numpy.exp(-tau*sec_Z) - jax.numpy.exp(-tau*sec_z))
    return sec_diff * exp_diff