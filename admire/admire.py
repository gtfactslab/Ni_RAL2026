import jax
import jax.numpy as jnp
import immrax as irx
from typing import Tuple

class AdmireNineDoFLinAct(irx.system.System):
    def __init__(self) -> None:
        self.evolution = 'continuous'
        self.xlen = 9
        self.ulen = 10
        self.mass = 9100. 
        self.Ix = 21000.
        self.Iy = 81000.
        self.Iz = 101000.
        self.Ixz = 2500.
        self.Sref = 45.00
        self.bref = 10.00
        self.cref = 5.20
        self.gravity = 9.81
        self.dmass = 0.
        self.dIx = 0.0
        self.dIy = 0.0
        self.dIz = 0.0
        self.dIzz = 0.0
        self.dIxx = 0.0
        self.dIyy = 0.0
        self.dIxz = 0.0
        self.Gama = self.Ix*(1+self.dIxx)*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz)						
        self.C1	 = ((self.Iy*(1+self.dIyy)-self.Iz*(1+self.dIzz))*self.Iz*(1+self.dIzz)-self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/self.Gama        
        self.C2	 = ((self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy)+self.Iz*(1+self.dIzz))*self.Ixz*(1+self.dIxz))/self.Gama
        self.C3   = self.Iz*(1+self.dIzz)/self.Gama
        self.C4	 = self.Ixz*(1+self.dIxz)/self.Gama
        self.C5   = (self.Iz*(1+self.dIzz)-self.Ix*(1+self.dIxx))/(self.Iy*(1+self.dIyy))
        self.C6	 = self.Ixz*(1+self.dIxz)/(self.Iy*(1+self.dIyy))
        self.C7	 = 1/(self.Iy*(1+self.dIyy))
        self.C8	 = (self.Ix*(1+self.dIxx)*(self.Ix*(1+self.dIxx)-self.Iy*(1+self.dIyy))+self.Ixz*(1+self.dIxz)*self.Ixz*(1+self.dIxz))/self.Gama
        self.C9	 = (self.Ix*(1+self.dIxx))/self.Gama
        

        data_list = [
            [-.62, -.62, -.4, -.62, -.62, -.4, -.16, .08, -.53, -1.78],
            [0., 0., -.02, -.04, -.04, -.02, 0., 0., 0., -.11],
            [0., 0., 0., .01, -.01, 0., .02, 0., .11, .0],
            [.37, -.37, -2.27, -1.96, 1.96, 2.27, 1.59, 0., -.64, 0.],
            [.67, .67, -.55, -.88, -.88, -.55, 0., -.2, .01, -6.63],
            [-.19, .19, -.1, -.22, .22, .1, -.96, .0, -5.34, 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]
        self.B_bar = jnp.array(data_list)
                    

    def f(self, t: float, x: jax.Array, u: jax.Array, p: jax.Array) -> jax.Array:
        Vt_st, alpha_st, beta_st, pb_st, qb_st, rb_st, psi_st, theta_st, phi_st = x[:9]
        ubody = Vt_st * jnp.cos(alpha_st) * jnp.cos(beta_st)
        vbody = Vt_st * jnp.sin(beta_st)
        wbody = Vt_st * jnp.sin(alpha_st) * jnp.cos(beta_st)
        uderiv = rb_st * vbody - qb_st * wbody - self.gravity * jnp.sin(theta_st) 
        vderiv = - rb_st * ubody + pb_st * wbody + self.gravity * jnp.sin(phi_st) * jnp.cos(theta_st)
        wderiv = qb_st * ubody - pb_st * vbody + self.gravity * jnp.cos(phi_st) * jnp.cos(theta_st)
        Vt_der = (ubody*uderiv+vbody*vderiv+wbody*wderiv)/Vt_st
        alpha_der = (ubody*wderiv-wbody*uderiv)/(ubody*ubody+wbody*wbody)
        beta_der  = (vderiv*Vt_st-vbody*Vt_der)/(Vt_st*Vt_st*jnp.cos(beta_st))
        pb_der    = (self.C1*rb_st+self.C2*pb_st)*qb_st
        qb_der    =  self.C5*pb_st*rb_st-self.C6*(pb_st*pb_st-rb_st*rb_st)
        rb_der    = (self.C8*pb_st-self.C2*rb_st)*qb_st

        psi_der   = (qb_st*jnp.sin(phi_st)+rb_st*jnp.cos(phi_st))/jnp.cos(theta_st)
        theta_der =  qb_st*jnp.cos(phi_st)-rb_st*jnp.sin(phi_st)
        phi_der   =  pb_st+jnp.tan(theta_st)*(qb_st*jnp.sin(phi_st)+rb_st*jnp.cos(phi_st))

        internal_dynamics = jnp.array([
            Vt_der,
            alpha_der,
            beta_der,
            pb_der,
            qb_der,
            rb_der,
            psi_der,
            theta_der,
            phi_der
        ])

        actuator_dynamics = self.B_bar @ (jnp.diag(p) @ u[:self.ulen])

        return internal_dynamics + actuator_dynamics 
    
    def g(self, t: float, x: jax.Array, v: jax.Array) -> jax.Array:
        Vt_st, alpha_st, beta_st, pb_st, qb_st, rb_st, psi_st, theta_st, phi_st = x[:9]
        
        ubody = Vt_st * jnp.cos(alpha_st) * jnp.cos(beta_st)
        vbody = Vt_st * jnp.sin(beta_st)
        wbody = Vt_st * jnp.sin(alpha_st) * jnp.cos(beta_st)
        uv = jnp.cos(phi_st) * jnp.cos(psi_st) * ubody + \
            (jnp.sin(phi_st) * jnp.sin(theta_st) * jnp.cos(psi_st) - jnp.cos(phi_st) * jnp.sin(psi_st)) * vbody + \
            (jnp.cos(phi_st) * jnp.sin(theta_st) * jnp.cos(psi_st) + jnp.sin(phi_st) * jnp.sin(psi_st)) * wbody
        vv = jnp.cos(phi_st) * jnp.sin(psi_st) * ubody + \
            (jnp.sin(phi_st) * jnp.sin(theta_st) * jnp.sin(psi_st) + jnp.cos(phi_st) * jnp.cos(psi_st)) * vbody + \
            (jnp.cos(phi_st) * jnp.sin(theta_st) * jnp.sin(psi_st) - jnp.sin(phi_st) * jnp.cos(psi_st)) * wbody
        wv = -jnp.sin(theta_st) * ubody + jnp.sin(phi_st) * jnp.cos(theta_st) * vbody + jnp.cos(phi_st) * jnp.cos(theta_st) * wbody
        return jnp.concatenate(x, jnp.array([uv, vv, wv, pb_st, qb_st, rb_st])) + v
        
