import sys
from pathlib import Path
from typing import Dict, Tuple

_EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import jax
import jax.numpy as jnp
import immrax as irx

_M = 35.89e-3
_G = 9.81
_IXX = 2.3951e-5
_IYY = 2.3951e-5
_IZZ = 3.2346e-5

_HOVER_THRUST = _M * _G

QPS_DT = 0.02
ARENA_X_HALF = 1.5
ARENA_Y_HALF = 1.2
ARENA_Z_RANGE = (0.1, 1.6)

class CrazyflieSystem(irx.System):

    def __init__(self, m: float = _M, g: float = _G,
                 Ixx: float = _IXX, Iyy: float = _IYY, Izz: float = _IZZ):
        self.evolution = 'continuous'
        self.xlen = 12
        self.m = m
        self.g = g
        self.Ixx = Ixx
        self.Iyy = Iyy
        self.Izz = Izz

    def f(self, t, x, u, p):
        alpha = p
        U1, U2, U3, U4 = alpha * u

        phi, theta, psi = x[3], x[4], x[5]
        vx, vy, vz = x[6], x[7], x[8]
        pr, qr, rr = x[9], x[10], x[11]

        sphi, cphi = jnp.sin(phi), jnp.cos(phi)
        stheta, ctheta, ttheta = jnp.sin(theta), jnp.cos(theta), jnp.tan(theta)
        spsi, cpsi = jnp.sin(psi), jnp.cos(psi)

        xdot = vx
        ydot = vy
        zdot = vz

        phidot = pr + sphi * ttheta * qr + cphi * ttheta * rr
        thetadot = cphi * qr - sphi * rr
        psidot = (sphi / ctheta) * qr + (cphi / ctheta) * rr

        z_b_x = cphi * stheta * cpsi + sphi * spsi
        z_b_y = cphi * stheta * spsi - sphi * cpsi
        z_b_z = cphi * ctheta

        vxdot = (U1 / self.m) * z_b_x
        vydot = (U1 / self.m) * z_b_y
        vzdot = (U1 / self.m) * z_b_z - self.g

        prdot = ((self.Iyy - self.Izz) / self.Ixx) * qr * rr + U2 / self.Ixx
        qrdot = ((self.Izz - self.Ixx) / self.Iyy) * pr * rr + U3 / self.Iyy
        rrdot = ((self.Ixx - self.Iyy) / self.Izz) * pr * qr + U4 / self.Izz

        return jnp.array([
            xdot, ydot, zdot, phidot, thetadot, psidot,
            vxdot, vydot, vzdot, prdot, qrdot, rrdot,
        ])

_EMB_CACHE: Dict[Tuple[float, float, float, float, float], Tuple[CrazyflieSystem, object]] = {}

def get_system_and_embedding(m: float = _M, g: float = _G, Ixx: float = _IXX,
                             Iyy: float = _IYY, Izz: float = _IZZ) -> Tuple[CrazyflieSystem, object]:
    key = (m, g, Ixx, Iyy, Izz)
    if key not in _EMB_CACHE:
        sys_ = CrazyflieSystem(m, g, Ixx, Iyy, Izz)
        emb = irx.natemb(sys_)
        _EMB_CACHE[key] = (sys_, emb)
    return _EMB_CACHE[key]

def euler_step(emb_sys, x_ivl: irx.Interval, u: jnp.ndarray,
               p_ivl: irx.Interval, dt: float) -> irx.Interval:
    _t = jnp.zeros(())
    x_ut = irx.i2ut(x_ivl)
    dx_ut = emb_sys.f(_t, x_ut, u, p_ivl)
    return irx.ut2i(dx_ut * dt + x_ut)

if __name__ == "__main__":
    sys_, emb = get_system_and_embedding()
    x0 = jnp.zeros(12)
    u_hover = jnp.array([_HOVER_THRUST, 0.0, 0.0, 0.0])
    p_nominal = jnp.ones(4)
    xdot = sys_.f(jnp.zeros(()), x0, u_hover, p_nominal)
    print("Hover residual (should be ~0):", xdot)
    assert jnp.allclose(xdot, 0.0, atol=1e-5), "Hover point is not an equilibrium!"
    print("OK: x=0, u=[m*g,0,0,0] is an exact fixed point.")
