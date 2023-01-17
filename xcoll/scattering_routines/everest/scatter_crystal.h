#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*gpufun*/
void drift_4d_single(LocalParticle* part, double length) {
    double const rpp    = LocalParticle_get_rpp(part);
    double const xp     = LocalParticle_get_px(part) * rpp;
    double const yp     = LocalParticle_get_py(part) * rpp;
    LocalParticle_add_to_x(part, xp * length );
    LocalParticle_add_to_y(part, yp * length );
}

/*gpufun*/
double drift_zeta_single(double rvv, double xp, double yp, double length){
    double const rv0v = 1./rvv;
    double const dzeta = 1 - rv0v * (1. + (pow(xp,2.) + pow(yp,2.))/2.);
    return length * dzeta;
}

/*gpufun*/
void shift_4d_single(LocalParticle* part, double dx, double dpx, double dy, double dpy) {
    LocalParticle_add_to_x(part, dx);
    LocalParticle_add_to_px(part, dpx);
    LocalParticle_add_to_y(part, dy);
    LocalParticle_add_to_py(part, dpy);
}

/*gpufun*/
void scatter_cry(EverestCrystalData el, LocalParticle* part) {

    // Crystal properties
    double length  = EverestCrystalData_get_active_length(el);
    // if collimator.jaw_F_L != collimator.jaw_B_L or collimator.jaw_F_R != collimator.jaw_B_R:
    //     raise NotImplementedError
    double const c_aperture = EverestCrystalData_get_jaw_F_L(el) - EverestCrystalData_get_jaw_F_R(el);
    double const c_offset   = EverestCrystalData_get_offset(el) + ( EverestCrystalData_get_jaw_F_L(el) + EverestCrystalData_get_jaw_F_R(el) )/2;
    double const c_tilt0    = EverestCrystalData_get_tilt(el, 0);
    double const c_tilt1    = EverestCrystalData_get_tilt(el, 1);
    double const onesided   = EverestCrystalData_get_onesided(el);
    double const c_rotation = atan2(EverestCrystalData_get_sin_z(el), EverestCrystalData_get_cos_z(el) );
    double const co_x       = EverestCrystalData_get_dx(el);
    double const co_px      = EverestCrystalData_get_dpx(el);
    double const co_y       = EverestCrystalData_get_dy(el);
    double const co_py      = EverestCrystalData_get_dpy(el);
    double const bend       = EverestCrystalData_get_bend(el);
    double const cry_tilt   = EverestCrystalData_get_align_angle(el) + EverestCrystalData_get_crytilt(el);
    double const bend_ang   = length/bend;    // temporary value
    if (cry_tilt >= -bend_ang) {
        length = bend*(sin(bend_ang + cry_tilt) - sin(cry_tilt));
    } else {
        length = bend*(sin(bend_ang - cry_tilt) + sin(cry_tilt));
    }
    double const cry_rcurv  = bend;
    double const cry_bend   = length/cry_rcurv; //final value (with corrected length) 
    double const cry_alayer = EverestCrystalData_get_thick(el);
    double const cry_xmax   = EverestCrystalData_get_xdim(el);
    double const cry_ymax   = EverestCrystalData_get_ydim(el);
    double const cry_orient = EverestCrystalData_get_orient(el);
    double const cry_miscut = EverestCrystalData_get_miscut(el);
    double const cry_cBend  = cos(cry_bend); 
    double const cry_sBend  = sin(cry_bend); 
    double const cry_cpTilt = cos(cry_tilt);
    double const cry_spTilt = sin(cry_tilt);
    double const cry_cnTilt = cos(-cry_tilt);
    double const cry_snTilt = sin(-cry_tilt);


    // Store initial coordinates for updating later
    double const rpp_in  = LocalParticle_get_rpp(part);
    double const rvv_in  = LocalParticle_get_rvv(part);
    double const e0      = LocalParticle_get_energy0(part) / 1e9; // Reference energy in GeV
    double const beta0   = LocalParticle_get_beta0(part);
    double const ptau_in = LocalParticle_get_ptau(part);
    double const x_in2    = LocalParticle_get_x(part);
    double const px_in2   = LocalParticle_get_px(part);
    double const y_in2    = LocalParticle_get_y(part);
    double const py_in2   = LocalParticle_get_py(part);
    double p0 = LocalParticle_get_p0c(part) / 1e9;

    // Drift to centre of collimator
    drift_4d_single(part, length/2);
    // Move to closed orbit  (dpx = dxp because ref. particle has delta = 0)
    shift_4d_single(part, -co_x, -co_px/rpp_in, -co_y, -co_py/rpp_in );
    // Backtrack to start of collimator
    drift_4d_single(part, -length/2);


    double x_in  = LocalParticle_get_x(part);
    double xp_in = LocalParticle_get_px(part)*rpp_in;
    double y_in  = LocalParticle_get_y(part);
    double yp_in = LocalParticle_get_py(part)*rpp_in;
    double s_in = 0;   // s at start of collimator

    // TODO: missing correction due to m/m0 (but also wrong in xpart...)
    double p_in = p0*ptau_in + e0; // energy, not momentum, in GeV

    int val_part_hit = 0;
    int val_part_abs = 0;
    int val_part_impact = -1;
    double val_part_indiv = -1.;
    double val_part_linteract = -1.;  // not used?
    double val_nabs_type = 0;         // not used?
    double val_linside = 0;

    p0 = e0;
    double x0     = 0;
    double xp0    = 0;
    double nhit   = 0;
    double nabs   = 0;
    double fracab = 0;
    double nnuc0  = 0;
    double ien0   = 0;
    double nnuc1  = 0;
    double ien1   = 0;
    double iProc  = 0;
    double n_chan = 0;
    double n_VR   = 0;
    double n_amorphous = 0;
    double s_imp  = 0;

    double x = x_in;
    double xp = xp_in;
    double z = y_in;
    double zp = yp_in;
    double p = p_in;
    double sp = 0;
    double dpop = (p - p0)/p0;
    double tiltangle = 0.;

    double mirror = 1.;

    // TODO: use xtrack C-code for rotation element

    // Compute rotation factors for collimator rotation
    double cRot   = cos(c_rotation);
    double sRot   = sin(c_rotation);
    double cRRot  = cos(-c_rotation);
    double sRRot  = sin(-c_rotation);

    // Transform particle coordinates to get into collimator coordinate  system
    // First do rotation into collimator frame
    x  =  x_in*cRot + sRot*y_in;
    z  =  y_in*cRot - sRot*x_in;
    xp = xp_in*cRot + sRot*yp_in;
    zp = yp_in*cRot - sRot*xp_in;

    // Output variables
    double x_out = x_in;
    double y_out = y_in;
    double xp_out = xp_in;
    double yp_out = yp_in;
    double p_out = p_in;


    // For one-sided collimators consider only positive X. For negative X jump to the next particle
    if (!onesided || (x >= 0.)) {
        // Log input energy + nucleons as per the FLUKA coupling
        nnuc0 = nnuc0 + 1.;
        ien0 = ien0 + p_in * 1.0e3;

        // Now mirror at the horizontal axis for negative X offset
        if (x < 0) {
            mirror    = -1;
            tiltangle = -1*c_tilt1;
        } else {
            mirror    = 1;
            tiltangle = c_tilt0;
        }
        x  = mirror*x;
        xp = mirror*xp;

        // Shift with opening and offset
        x = (x - c_aperture/2.) - mirror*c_offset;

        // Include collimator tilt
        if (tiltangle > 0.) {
            xp = xp - tiltangle;
        }
        if (tiltangle < 0.) {
            x  = x + sin(tiltangle) * length;
            xp = xp - tiltangle;
        }

        // particle passing above the jaw are discarded => take new event
        // entering by the face, shorten the length (zlm) and keep track of
        // entrance longitudinal coordinate (keeps) for histograms

        // The definition is that the collimator jaw is at x>=0.

        // 1) Check whether particle hits the collimator
        int isimp = 0;
        double s = 0.;
        double zlm = -1*length; 


        double* crystal_result = crystal(x,
                                xp,
                                z,
                                zp,
                                s,
                                p,
                                x0,
                                xp0,
                                zlm,
                                s_imp,
                                isimp,
                                val_part_hit,
                                val_part_abs,
                                val_part_impact,
                                val_part_indiv,
                                length,
                                EverestCrystalData_getp_material(el),
                                nhit,
                                nabs,
                                cry_tilt,
                                cry_rcurv,
                                cry_bend,
                                cry_alayer,
                                cry_xmax,
                                cry_ymax,
                                cry_orient,
                                cry_miscut,
                                cry_cBend,
                                cry_sBend,
                                cry_cpTilt,
                                cry_spTilt,
                                cry_cnTilt,
                                cry_snTilt,
                                iProc,
                                n_chan,
                                n_VR,
                                n_amorphous
                                );

        val_part_hit = crystal_result[0];
        val_part_abs = crystal_result[1];
        val_part_impact = crystal_result[2];
        val_part_indiv = crystal_result[3];
        nhit = crystal_result[4];
        nabs = crystal_result[5];
        s_imp = crystal_result[6];
        isimp = crystal_result[7];
        s = crystal_result[8];
        zlm = crystal_result[9];
        x = crystal_result[10];
        xp = crystal_result[11];
        x0 = crystal_result[12];
        xp0 = crystal_result[13];
        z = crystal_result[14];
        zp = crystal_result[15];
        p = crystal_result[16];
        iProc = crystal_result[17];
        n_chan = crystal_result[18];
        n_VR = crystal_result[19];
        n_amorphous = crystal_result[20];

        if (nabs != 0.) {
            val_part_abs = 1.;
            val_part_linteract = zlm;
        }
        s_imp  = (s - length) + s_imp;

        // Transform back to particle coordinates with opening and offset
        if (x < 99.0e-3) {
            // Include collimator tilt
            if (tiltangle > 0) {
                x  = x  + tiltangle*length;
                xp = xp + tiltangle;
            } else if (tiltangle < 0) {
                x  = x  + tiltangle*length;
                xp = xp + tiltangle;
                x  = x  - sin(tiltangle) * length;
            }

            // Transform back to particle coordinates with opening and offset
            x   = (x + c_aperture/2) + mirror*c_offset;

            // Now mirror at the horizontal axis for negative X offset
            x  = mirror * x;
            xp = mirror * xp;

            // Last do rotation into collimator frame
            x_out  =  x*cRRot +  z*sRRot;
            y_out  =  z*cRRot -  x*sRRot;
            xp_out = xp*cRRot + zp*sRRot;
            yp_out = zp*cRRot - xp*sRRot;

            // Log output energy + nucleons as per the FLUKA coupling
            // Do not log dead particles
            nnuc1       = nnuc1 + 1;                           // outcoming nucleons
            ien1        = ien1  + p_out * 1e3;                 // outcoming energy

            p_out = p;
            s_in = s_in + s;
        } else {
            x_out = x;
            y_out = z;
        }
    }

    LocalParticle_set_x(part, x_out);
    LocalParticle_set_px(part, xp_out/rpp_in);
    LocalParticle_set_y(part, y_out);
    LocalParticle_set_py(part, yp_out/rpp_in);

    // Backtrack to centre of collimator
    drift_4d_single(part, -length/2);
    // Return from closed orbit  (dpx = dxp because ref. particle has delta = 0)
    shift_4d_single(part, co_x, co_px/rpp_in, co_y, co_py/rpp_in );

    double energy_out = p_out;       //  Cannot assign energy directly to LocalParticle as it would update dependent variables, but needs to be corrected first!

    // Update energy    ---------------------------------------------------
    // Only particles that hit the jaw and survived need to be updated
    if (val_part_hit>0 && val_part_abs==0){
        double ptau_out = (energy_out - e0) / (e0 * beta0);
        LocalParticle_update_ptau(part, ptau_out);
    }

    // Drift to end of collimator
    drift_4d_single(part, length/2);

    // Update 4D coordinates    -------------------------------------------
    // Absorbed particles get their coordinates set to the entrance of collimator
    if (val_part_abs>0){
        LocalParticle_set_x(part, x_in2);
        LocalParticle_set_px(part, px_in2);
        LocalParticle_set_y(part, y_in2);
        LocalParticle_set_py(part, py_in2);
    }

    // Update longitudinal coordinate zeta    -----------------------------
    // Absorbed particles keep coordinates at the entrance of collimator, others need correcting:
    // Non-hit particles are just drifting (zeta not yet drifted in K2, so do here)
    if (val_part_hit==0){
        LocalParticle_add_to_zeta(part, drift_zeta_single(rvv_in, px_in2*rpp_in, py_in2*rpp_in, length) );
    }
    // Hit and survived particles need correcting:
    if (val_part_hit>0 && val_part_abs==0){
        double px  = LocalParticle_get_px(part);
        double py  = LocalParticle_get_py(part);
        double rvv = LocalParticle_get_rvv(part);
        double rpp = LocalParticle_get_rpp(part);
        // First we drift half the length with the old angles:
        LocalParticle_add_to_zeta(part, drift_zeta_single(rvv_in, px_in2*rpp_in, py_in2*rpp_in, length/2) );
        // then half the length with the new angles:
        LocalParticle_add_to_zeta(part, drift_zeta_single(rvv, px*rpp, py*rpp, length/2) );
    }

    // Update s    --------------------------------------------------------
    // TODO: move absorbed particles to last impact location
    if (val_part_abs==0){
        LocalParticle_add_to_s(part, length);
    }

    // Update state    ----------------------------------------------------
    if (val_part_abs > 0){
        LocalParticle_set_state(part, -333);
    }

    return;

}