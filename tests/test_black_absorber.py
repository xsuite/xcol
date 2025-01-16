# copyright ############################### #
# This file is part of the Xcoll package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import numpy as np
import pytest

import xobjects as xo
import xpart as xp
import xtrack as xt
import xcoll as xc

from xpart.test_helpers import flaky_assertions, retry
from xobjects.test_helpers import for_all_test_contexts, fix_random_seed

n_part = int(2.e6)

@for_all_test_contexts(
    excluding=('ContextCupy', 'ContextPyopencl')  # BlackAbsorber not on GPU
)
def test_with_parallel_beam(test_context):
    jaw_L, jaw_R, _, _, _, _, _, _, L, coll = _make_absorber(_context=test_context)
    # Create particles
    part = _generate_particles(_context=test_context)
    part_init = part.copy()
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)
    # As the angles are zero, only particles that started in front of the jaw are lost
    mask_hit = (part_init.x >= jaw_L) | (part_init.x <= jaw_R)

    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_lost = np.unique(part.s[mask_hit])
    s_alive = np.unique(part.s[~mask_hit])
    assert len(s_lost)==1 and s_lost[0]==0
    assert len(s_alive)==1 and s_alive[0]==L
    assert np.allclose(part.x, part_init.x, atol=1e-12, rtol=0)


# Should test jaws with different angles
@pytest.mark.parametrize("angle_L, angle_R, tilt_L, tilt_R", [
                        [0,  0,  0,       0],
                        [0,  0,  0.05,    0.05],
                        [0,  0,  -0.035,  -0.035],
                        [0,  0,  -0.045,  0.015],
                        [0,  0,  0.045,   -0.015],
                        [90, 90, 0,       0],
                        [90, 90, 0.05,    0.05],
                        [90, 90, -0.035,  -0.035],
                        [90, 90, -0.045,  0.015],
                        [90, 90, 0.045,   -0.015],
                        [41, 39, 0,       0],
                        [41, 39, 0.05,    0.05],
                        [41, 39, -0.035,  -0.035],
                        [41, 39, -0.045,  0.015],
                        [41, 39, 0.045,   -0.015],
                    ], ids=["H", "H tilt pos", "H tilt neg", "H tilt neg pos", "H tilt pos neg",
                            "V", "V tilt pos", "V tilt neg", "V tilt neg pos", "V tilt pos neg",
                            "S", "S tilt pos", "S tilt neg", "S tilt neg pos", "S tilt pos neg"])
@for_all_test_contexts(
    excluding=('ContextCupy', 'ContextPyopencl')  # BlackAbsorber not on GPU
)
def test_with_generic_beam(test_context, angle_L, angle_R, tilt_L, tilt_R):
    # Create collimator
    j_LU, j_RU, j_LD, j_RD, s_LU, s_RU, s_LD, s_RD, L, coll = _make_absorber(angle=[angle_L, angle_R], tilts=[tilt_L, tilt_R], _context=test_context)
    # Create particles
    part = _generate_particles(four_dim=True, _context=test_context)
    part_init = part.copy()
    # Track
    coll.track(part)
    part.move(_context=xo.ContextCpu())
    part.sort(interleave_lost_particles=True)

    # Get all expected hit locations
    # # TODO adapt when exact drifts are implemented
    dri = xt.Drift(length=s_LU)
    dri.track(part_init)
    assert np.allclose(part_init.s, s_LU, atol=1e-12, rtol=0)
    x_rot = part_init.x * np.cos(np.deg2rad(angle_L)) + part_init.y * np.sin(np.deg2rad(angle_L))
    mask_hit_LU = x_rot >= j_LU
    dri.length = s_LD - s_LU
    dri.track(part_init)
    assert np.allclose(part_init.s, s_LD, atol=1e-12, rtol=0)
    x_rot = part_init.x * np.cos(np.deg2rad(angle_L)) + part_init.y * np.sin(np.deg2rad(angle_L))
    mask_hit_L = ~mask_hit_LU & (x_rot >= j_LD)
    dri.length = s_RU - s_LD
    dri.track(part_init)
    assert np.allclose(part_init.s, s_RU, atol=1e-12, rtol=0)
    x_rot = part_init.x * np.cos(np.deg2rad(angle_R)) + part_init.y * np.sin(np.deg2rad(angle_R))
    mask_hit_RU = x_rot <= j_RU
    dri.length = s_RD - s_RU
    dri.track(part_init)
    assert np.allclose(part_init.s, s_RD, atol=1e-12, rtol=0)
    x_rot = part_init.x * np.cos(np.deg2rad(angle_R)) + part_init.y * np.sin(np.deg2rad(angle_R))
    mask_hit_R = ~mask_hit_RU & (x_rot <= j_RD)
    mask_hit = mask_hit_LU | mask_hit_RU | mask_hit_L | mask_hit_R
    # Correct for potential double hits (the upstream jaw is first, so that is kept):
    mask_hit_L = mask_hit_L & ~mask_hit_RU
    mask_hit_R = mask_hit_R & ~mask_hit_LU

    # Check lost and alive
    lost = np.unique(part.state[mask_hit])
    alive = np.unique(part.state[~mask_hit])
    assert len(lost)==1 and lost[0]==-340
    assert len(alive)==1 and alive[0]==1
    s_alive = np.unique(part.s[~mask_hit])

    # Check s positions
    assert len(s_alive)==1 and s_alive[0]==L
    sgn_L = np.sign(tilt_L)
    sgn_R = np.sign(tilt_R)
    assert np.all([sgn_L*s <= sgn_L*s_LU for s in part.s[mask_hit_LU]])
    assert np.all([sgn_R*s >= sgn_R*s_RU for s in part.s[mask_hit_RU]])
    assert np.all([s > s_LU and s <= s_LD for s in part.s[mask_hit_L]])
    assert np.all([s > s_RU and s <= s_RD for s in part.s[mask_hit_R]])

    # Check that the particles are at the border of the jaws
    x_rot = part.x * np.cos(np.deg2rad(angle_L)) + part.y * np.sin(np.deg2rad(angle_L))
    if tilt_L == 0:
        assert np.allclose(x_rot[mask_hit_L], j_LU, atol=1e-12, rtol=0)
    else:
        assert np.allclose(x_rot[mask_hit_LU], (part.s[mask_hit_LU] - s_LU)*np.tan(tilt_L + np.pi/2) + j_LU, atol=1e-12, rtol=0)
        assert np.allclose(x_rot[mask_hit_L], (part.s[mask_hit_L] - s_LU)*np.tan(tilt_L) + j_LU, atol=1e-12, rtol=0)
    x_rot = part.x * np.cos(np.deg2rad(angle_R)) + part.y * np.sin(np.deg2rad(angle_R))
    if tilt_R == 0:
        assert np.allclose(x_rot[mask_hit_R], j_RU, atol=1e-12, rtol=0)
    else:
        assert np.allclose(x_rot[mask_hit_RU], (part.s[mask_hit_RU] - s_RU)*np.tan(tilt_R - np.pi/2) + j_RU, atol=1e-12, rtol=0)
        assert np.allclose(x_rot[mask_hit_R], (part.s[mask_hit_R] - s_RU)*np.tan(tilt_R) + j_RU, atol=1e-12, rtol=0)


def _make_absorber(angle=0, tilts=[0,0], _context=None):
    if _context is None:
        _context = xo.ContextCpu()
    jaws = [0.05 + 0.01*np.random.normal(), -0.05 + 0.001*np.random.normal()]
    L = 0.873 + 0.1*np.random.normal()
    coll = xc.BlackAbsorber(length=L, angle=angle, jaw=jaws, tilt=tilts, _context=_context)
    return coll.jaw_LU, coll.jaw_RU, coll.jaw_LD, coll.jaw_RD, coll.jaw_s_LU, coll.jaw_s_RU, coll.jaw_s_LD, coll.jaw_s_RD, L, coll


def _generate_particles(four_dim=False, angle=0, _context=None):
    if _context is None:
        _context = xo.ContextCpu()
    # Make particles
    x = np.random.uniform(-0.1, 0.1, n_part)
    y = np.random.uniform(-0.1, 0.1, n_part)
    if four_dim:
        px = np.random.uniform(-0.1, 0.1, n_part)
        py = np.random.uniform(-0.1, 0.1, n_part)
    else:
        px = 0
        py = 0
    ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12, _context=_context)
    part = xp.build_particles(x=x, y=y, px=px, py=py, particle_ref=ref, _context=_context)
    part_init = part.copy()
    part_init.move(_context=xo.ContextCpu())
    return part


@pytest.mark.parametrize("side, sign_R", [
                        ['+', 1], ['-', 1], ['+', -1], ['-', -1]]
                        , ids=["L R>0", "R R>0", "L R<0", "R R<0"])
@for_all_test_contexts(
    excluding=('ContextCupy', 'ContextPyopencl')  # BlackAbsorber not on GPU
)
@fix_random_seed(3482634)
def test_black_crystal(test_context, side, sign_R):
    ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
    x = np.random.uniform(-1, 1, n_part)
    px = np.random.uniform(-1, 1, n_part)
    y = np.random.uniform(-1, 1, n_part)
    py = np.random.uniform(-1, 1, n_part)
    part_init = xp.build_particles(x=x, px=px, y=y, py=py, particle_ref=ref)
    drift_length = 1  # To send the surviving particles further out
    dri = xt.Drift(length=drift_length)

    for R in [0.87, 3.3, 17]:
        R = sign_R*R
        for tilt in [-89, -65, -42, -22, -8, 0, 11, 27, 48, 72, 89]:
            print(f"{R=}{tilt=}")
            length = 0.87
            sign = 1 if side == '+' else -1
            jaw = sign*0.13
            width = 0.2
            height = 0.5
            coll = xc.BlackCrystal(length=length, bending_radius=R, jaw=jaw, width=width, height=height, side=side, tilt=np.deg2rad(tilt), _context=test_context)
            part = part_init.copy()
            coll.track(part)
            dri.track(part)

            Rpos = R - (sign_R-1)/2*width # full radius when positive
            Rneg = R - (sign_R+1)/2*width # shorter radius when positive (full radius when negative)

            # Four corners:
            x_TU = jaw + (1+sign)/2*width
            s_TD = Rneg*length/R
            x_TD = x_TU + Rneg*(1 - np.sqrt(1 - length**2/R**2))
            x_BU = jaw - (1-sign)/2*width
            s_BD = Rpos*length/R
            x_BD = x_BU + Rpos*(1 - np.sqrt(1 - length**2/R**2))
            Rs   = 0
            Rx   = x_BU + R if sign_R == 1 else x_TU + R

            mask_alive = part.state > 0
            assert np.allclose(part.s[mask_alive], length+drift_length)

            # Rotate particles to tilted frame
            part_s =  part.s * np.cos(np.deg2rad(tilt)) + (part.x - x_BU) * np.sin(np.deg2rad(tilt))
            part_x = -part.s * np.sin(np.deg2rad(tilt)) + (part.x - x_BU)  * np.cos(np.deg2rad(tilt)) + x_BU

            mask_not_end = part.s < length + drift_length
            mask_height = (part.y < height/2) & (part.y > -height/2)
            mask_front = np.isclose(part_s, 0) & mask_height & mask_not_end
            assert np.all(part.state[mask_front] < 1)
            assert np.all(part_x[mask_front] <= x_TU)
            assert np.all(part_x[mask_front] >= x_BU)
            mask_upper_curve = np.isclose((part_x - Rx)**2 + (part_s - Rs)**2, Rneg**2) & mask_not_end & mask_height
            assert np.all(part.state[mask_upper_curve] < 1)
            mask_back = np.isclose(part_x, (x_TD - x_BD)/(s_TD - s_BD) * (part_s - s_BD) + x_BD) & mask_not_end & mask_height
            assert np.all(part.state[mask_back] < 1)
            mask_lower_curve = np.isclose((part_x - Rx)**2 + (part_s - Rs)**2, Rpos**2) & mask_not_end & mask_height
            assert np.all(part.state[mask_lower_curve] < 1)
            mask_top_face = np.isclose(part.y, height/2) & mask_not_end
            assert np.all(part.state[mask_top_face] < 1)
            mask_bottom_face = np.isclose(part.y, -height/2) & mask_not_end
            assert np.all(part.state[mask_bottom_face] < 1)
            assert np.all(mask_alive == ~mask_front & ~mask_upper_curve & ~mask_back & ~mask_lower_curve & ~mask_top_face & ~mask_bottom_face)

            # _, ax = plt.subplots(3, 2, figsize=(10, 5))
            # ax[0][0].scatter(part.s[~mask_alive], part.x[~mask_alive], s=0.1)
            # # ax[0][0].set_aspect('equal')
            # ax[0][1].scatter(part.s[mask_front], part.x[mask_front], s=0.1)
            # ax[0][1].scatter(part.s[mask_upper_curve], part.x[mask_upper_curve], s=0.1)
            # ax[0][1].scatter(part.s[mask_back], part.x[mask_back], s=0.1)
            # ax[0][1].scatter(part.s[mask_lower_curve], part.x[mask_lower_curve], s=0.1)
            # ax[1][0].scatter(part.s[mask_front], part.x[mask_front], s=0.1)
            # ax[1][1].scatter(part.s[mask_upper_curve], part.x[mask_upper_curve], s=0.1)
            # ax[2][0].scatter(part.s[mask_back], part.x[mask_back], s=0.1)
            # ax[2][1].scatter(part.s[mask_lower_curve], part.x[mask_lower_curve], s=0.1)
