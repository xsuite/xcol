# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import time
start_time = time.time()

import xpart as xp
import xtrack as xt
import xcoll as xc
from xobjects.test_helpers import for_all_test_contexts


num_part = int(1e6)


@for_all_test_contexts(
    excluding=('ContextCupy', 'ContextPyopencl')  # Rutherford RNG not on GPU
)
def test_transfer_line(test_context):
    # Create the line
    line = _create_transfer_line()
    air = _add_air_regions(line)
    assert xt.line._dicts_equal(line["Air 1"].material.to_dict(), air.to_dict())
    assert xt.line._dicts_equal(line["Air 2"].material.to_dict(), air.to_dict())
    _add_monitors(line)
    line.build_tracker(_context=test_context)
    line.scattering.disable()  # Scattering need to be disabled to be able to twiss
    part = _generate_matched_particles(line)
    line.scattering.enable()   # Re-enable scattering
    line.track(part)
    nemitt_x = np.array([el.nemitt_x for el in line.get_elements_of_type(xc.EmittanceMonitor)[0]])
    nemitt_y = np.array([el.nemitt_y for el in line.get_elements_of_type(xc.EmittanceMonitor)[0]])
    assert np.allclose(nemitt_x[0:2],  7.7e-6, atol=1e-7)
    assert np.allclose(nemitt_x[2:2], 10.7e-6, atol=1e-7)
    assert np.allclose(nemitt_x[4:2], 15.7e-6, atol=1e-7)
    assert np.allclose(nemitt_y[0:2],  3.5e-6, atol=1e-7)
    assert np.allclose(nemitt_y[2:2],  6.6e-6, atol=1e-7)
    assert np.allclose(nemitt_y[4:2], 10.9e-6, atol=1e-7)


def _create_transfer_line():
    k_qf_1 = 0.32730047
    k_qd_2 = -0.36102915
    k_qf_3 = 0.32789126
    k_qd_4 = -0.1991137
    l_quad = 2
    elements = [
                xt.Quadrupole(k1=k_qf_1, length=l_quad),
                xt.Drift(length=1.),
                xt.Quadrupole(k1=k_qd_2, length=l_quad),
                xt.Drift(length=1.),
                xt.Quadrupole(k1=k_qf_3, length=l_quad),
                xt.Drift(length=1.),
                xt.Quadrupole(k1=k_qd_4, length=l_quad),
                xt.Drift(length=89.),
                xt.Marker()
               ]
    element_names = ["QF1", "drift_1", "QD2", "drift_2", "QF3", "drift_3", "QD4",
                     "drift_4", "END"]
    particle_ref = xp.Particles(energy0=24.e9)
    line = xt.Line(elements=elements, element_names=element_names, particle_ref=particle_ref)
    return line


def _add_air_regions(line):
    X0_air = 301
    air = xc.Material(radiation_length=X0_air, name="Air (1 atm 20C)")
    line.insert_element(element=xc.EverestBlock(length=10, material=air), name="Air 1", at_s=20)
    line.insert_element(element=xc.EverestBlock(length=10, material=air), name="Air 2", at_s=50)
    return air


def _add_monitors(line):
    xc.EmittanceMonitor.install(line, name="monitor start", at_s=0, longitudinal=False)
    xc.EmittanceMonitor.install(line, name="monitor air 1 start", at_s=20, longitudinal=False)
    xc.EmittanceMonitor.install(line, name="monitor air 1 end", at_s=30, longitudinal=False)
    xc.EmittanceMonitor.install(line, name="monitor air 2 start", at_s=50, longitudinal=False)
    xc.EmittanceMonitor.install(line, name="monitor air 2 end", at_s=60, longitudinal=False)
    xc.EmittanceMonitor.install(line, name="monitor end", at_s=100, longitudinal=False)

def _generate_matched_particles(line):
    # Matched initial parameters
    betx0 = 154.0835045206266
    bety0 = 5.222566527078791
    alfx0 = -36.90472944993891
    alfy0 = 0.2523074897915478
    dx0 = 0.13
    dy0 = 0.0
    dpx0 = 0.02
    dpy0 = 0.0
    tw_init = xt.TwissInit(betx=betx0, bety=bety0, alfx=alfx0, alfy=alfy0, dx=dx0, dy=dy0, dpx=dpx0, dpy=dpy0)
    tw = line.twiss(method='4d', start="monitor start", end="END", init=tw_init)
    nemitt_x = 7.639770207283603e-06
    nemitt_y = 3.534081877201574e-06
    x_norm, px_norm = xp.generate_2D_gaussian(num_part)
    y_norm, py_norm = xp.generate_2D_gaussian(num_part)
    part = line.build_particles(x_norm=x_norm, px_norm=px_norm, y_norm=y_norm, py_norm=py_norm,
                                W_matrix=tw.W_matrix[0], particle_on_co=line.particle_ref,
                                nemitt_x=nemitt_x,nemitt_y=nemitt_y)
    return part
