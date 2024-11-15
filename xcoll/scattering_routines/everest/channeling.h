// copyright ############################### #
// This file is part of the Xcoll package.   #
// Copyright (c) CERN, 2023.                 #
// ######################################### #

#ifndef XCOLL_EVEREST_CHANNELING_H
#define XCOLL_EVEREST_CHANNELING_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


// Convention:
//       L: length of curved trajectory
//       s: longitudinal coordinate (in collimator frame)  -  projection of curved trajectory
//       t: opening angle of curve ('t' for theta)
//       r: radius from position to bending centre (including t_I):  L = t*r


/*gpufun*/
double channeling_average_density(EverestData restrict everest, CrystalGeometry restrict cg, LocalParticle* part, double pc) {

    // Material properties
    double const anuc = everest->coll->anuc;
    double const rho  = everest->coll->rho;
    double const eum  = everest->coll->eum;

    double bend_r = cg->bending_radius;
    double ratio  = everest->Rc_over_R;

    //Rescale the total and inelastic cross-section accordigly to the average density seen
    double x_i = LocalParticle_get_x(part);
    double xp  = LocalParticle_get_xp(part);
    int np     = x_i/XC_PLANE_DISTANCE;          //Calculate in which crystalline plane the particle enters
    x_i       -= (np + 0.5)*XC_PLANE_DISTANCE;   //Rescale the incoming x to the middle of the crystalline plane

    double pv   = pow(pc, 2.)/sqrt(pow(pc, 2.) + pow(XC_PROTON_MASS*1.0e-3, 2.))*1.0e9; //Calculate pv=P/E   TODO: this is beta?
    double Ueff = eum*4.*pow(x_i/XC_PLANE_DISTANCE, 2.) + pv*x_i/bend_r; //Calculate effective potential
    double Et   = pv*pow(xp, 2.)/2. + Ueff;       //Calculate transverse energy
    double Ec   = eum*pow(1. - ratio, 2.);         //Calculate critical energy in bent crystals

    //To avoid negative Et
    double xminU = -pow(XC_PLANE_DISTANCE, 2.)*pc*1.0e9/(8.*eum*bend_r);
    double Umin  = fabs(eum*4.*pow(xminU/XC_PLANE_DISTANCE, 2.) + pv*xminU/bend_r);
    Et    = Et + Umin;
    Ec    = Ec + Umin;

    //Calculate min e max of the trajectory between crystalline planes
    double x_min = -0.5*XC_PLANE_DISTANCE*ratio - 0.5*XC_PLANE_DISTANCE*sqrt(Et/Ec);
    double x_max = -0.5*XC_PLANE_DISTANCE*ratio + 0.5*XC_PLANE_DISTANCE*sqrt(Et/Ec);

    //Change ref. frame and go back with 0 on the crystalline plane on the left
    x_min = x_min - XC_PLANE_DISTANCE/2.;
    x_max = x_max - XC_PLANE_DISTANCE/2.;

    //Calculate the "normal density" in m^-3
    double N_am  = rho*XC_AVOGADRO*1.0e6/anuc;

    //Calculate atomic density at min and max of the trajectory oscillation
    // erf returns the error function of complex argument
    double rho_max = erf(x_max/sqrt(2*pow(XC_THERMAL_VIBRATIONS, 2.)));
    rho_max       -= erf((XC_PLANE_DISTANCE-x_max)/sqrt(2*pow(XC_THERMAL_VIBRATIONS, 2.)));
    rho_max       *= N_am*XC_PLANE_DISTANCE/2.;
    double rho_min = erf(x_min/sqrt(2*pow(XC_THERMAL_VIBRATIONS, 2.)));
    rho_min       -= erf((XC_PLANE_DISTANCE-x_min)/sqrt(2*pow(XC_THERMAL_VIBRATIONS, 2.)));
    rho_min       *= N_am*XC_PLANE_DISTANCE/2.;

    //"zero-approximation" of average nuclear density seen along the trajectory
    double avrrho  = (rho_max - rho_min)/(x_max - x_min);
    avrrho  = 2.*avrrho/N_am;

    return avrrho;
}


/*gpufun*/
double* channel_transport(EverestData restrict everest, LocalParticle* part, double pc, double L_chan, double t_I, double t_P) {
    // Channeling: happens over an arc length L_chan (potentially less if dechanneling)
    //             This equates to an opening angle t_P wrt. to the point P (center of miscut if at start of crystal)
    //             The chord angle xp at the start of channeling (I) is t_P/2 + t_I
    //             The angle xp at the end of channeling (F) is t_P + t_I
    //             In practice: we drift from start to end, but overwrite the angle afterwards
    // TODO: why does channeling only have 50% energy loss?

    double* result = (double*)malloc(2 * sizeof(double));

    InteractionRecordData record = everest->coll->record;
    RecordIndex record_index     = everest->coll->record_index;
    int8_t sc = everest->coll->record_scatterings;

    // First log particle at start of channeling
    int64_t i_slot = -1;
    if (sc) i_slot = InteractionRecordData_log(record, record_index, part, XC_CHANNELING);

    // Do channeling.
    // The distance from I to F is the chord length of the angle t_P: d = 2 r sin(t_P/2)
    // Hence the longitudinal distance (the length to be drifted) is the projection of this using the
    // xp at the start of channeling: s = 2 r sin(t_P/2)cos(t_P/2 + t_I)
    double t_chord= t_I + t_P/2.;
    double drift_length = 2.*L_chan/t_P * sin(t_P/2.) * cos(t_chord);
    LocalParticle_set_xp(part, t_chord);          // Angle at start of channeling
    Drift_single_particle_4d(part, drift_length);
    // In reality,the particle oscillates horizontally between the planes while channeling.
    // This effect is mimicked by giving a random angle spread at the exit
    double sigma_ran = 0.5*everest->t_c;
    double ran_angle = RandomNormal_generate(part)*sigma_ran;
    LocalParticle_set_xp(part, t_I + t_P + ran_angle); // Angle at end of channeling

    // Apply energy loss along trajectory
    double energy_loss = 0.5*calcionloss(everest, part, L_chan);
    // TODO: LocalParticle_add_to_energy(part, - energy_loss*L_chan*1.e9, change_angle);
    // if change_angle = 0  => LocalParticle_scale_px(part, old_rpp / new_rpp) such that xp remains the same
    // It is done in K2, so we should do it. Though, it seems that with the current implementation in xtrack
    // this is no longer correct with exact drifts...?
    pc = pc - energy_loss*L_chan; //energy loss to ionization [GeV]

    // Finally log particle at end of channeling
    if (sc) InteractionRecordData_log_child(record, i_slot, part);

    result[0] = drift_length;
    result[1] = pc;
    return result;
}


double Channel(EverestData restrict everest, LocalParticle* part, CrystalGeometry restrict cg, double pc, double length) {

    if (LocalParticle_get_state(part) < 1){
        // Do nothing if already absorbed
        return pc;
    }

    InteractionRecordData record = everest->coll->record;
    RecordIndex record_index     = everest->coll->record_index;
    int8_t sc = everest->coll->record_scatterings;

    calculate_initial_angle(everest, part, cg);
#ifdef XCOLL_REFINE_ENERGY
    calculate_critical_angle(everest, part, cg, pc);
#endif

    // Do we channel, or are we in the transition between channeling and VR?
    double xp = LocalParticle_get_xp(part);
    double alpha = fabs(xp - everest->t_I) / everest->t_c;
    double ratio = everest->Rc_over_R;
    double xi = RandomUniform_generate(part)/(1 - ratio)/sqrt(everest->coll->eta);

    if (xi > 1 || alpha > 2*sqrt(xi)*sqrt(1-xi)) {
        // No channelling
        pc = Amorphous(everest, part, cg, pc, length, 1);

    } else {
        // Channelling
        calculate_opening_angle(everest, part, cg);
        double t_I = everest->t_I;
        double t_P = everest->t_P;
        double L_chan = everest->r*t_P;

        // ------------------------------------------------
        // Calculate curved length L_dechan of dechanneling
        // ------------------------------------------------
        double const_dech = calculate_dechanneling_length(everest, pc);
        double TLdech1 = const_dech*pc*pow(1. - ratio, 2.); //Updated calculate typical dech. length(m)
        double N_atom = 1.0e-1;
        if(RandomUniform_generate(part) <= N_atom) {
            TLdech1 /= 200.;   // Updated dechanneling length (m)      
        }
        double L_dechan = TLdech1*RandomExponential_generate(part);   // Actual dechan. length

        // -----------------------------------------------------
        // Calculate curved length L_nucl of nuclear interaction
        // -----------------------------------------------------
        // Nuclear interaction length is rescaled in this case, because channeling
        double collnt = everest->coll->collnt;
        double avrrho = channeling_average_density(everest, cg, part, pc);
        if (avrrho == 0) {
            collnt = 1.e10;  // very large because essentially 1/0
        } else {
            collnt = collnt/avrrho;
        }
        double L_nucl = collnt*RandomExponential_generate(part);

// printf("Channeling (%f -> %f):   L_c: %f,  L_d: %f,  L_n: %f\n", t_I, t_P, L_chan, L_dechan, L_nucl);
        // ------------------------------------------------------------------------
        // Compare the 3 lengths: the first one encountered is what will be applied
        // ------------------------------------------------------------------------
        if (L_chan <= fmin(L_dechan, L_nucl)){
            // Channel full length
            double* result_chan = channel_transport(everest, part, pc, L_chan, t_I, t_P);
            // double channeled_length = result_chan[0];
            pc               = result_chan[1];
            free(result_chan);

        } else if (L_dechan < L_nucl) {
            // Channel up to L_dechan, then amorphous
            double* result_chan = channel_transport(everest, part, pc, L_dechan, t_I, t_P*L_dechan/L_chan);
            double channeled_length = result_chan[0];
            pc               = result_chan[1];
            free(result_chan);
            if (sc) InteractionRecordData_log(record, record_index, part, XC_DECHANNELING);
            pc = Amorphous(everest, part, cg, pc, length - channeled_length, 1);

        } else {
            // Channel up to L_nucl, then scatter, then amorphous
            double* result_chan = channel_transport(everest, part, pc, L_nucl, t_I, t_P*L_nucl/L_chan);
            double channeled_length = result_chan[0];
            pc               = result_chan[1];
            free(result_chan);
            // Rescale nuclear interaction parameters
            everest->rescale_scattering = avrrho;
#ifndef XCOLL_REFINE_ENERGY
            calculate_scattering(everest, pc);
#endif
            pc = nuclear_interaction(everest, part, pc);
            if (LocalParticle_get_state(part) == XC_LOST_ON_EVEREST_COLL){
                LocalParticle_set_state(part, XC_LOST_ON_EVEREST_CRYSTAL);
            } else {
                // We call the main Amorphous function for the leftover
                everest->rescale_scattering = 1;
#ifndef XCOLL_REFINE_ENERGY
                calculate_scattering(everest, pc);
#endif
                pc = Amorphous(everest, part, cg, pc, length - channeled_length, 1);
            }
        }
    }

    return pc;
}

#endif /* XCOLL_EVEREST_CHANNELING_H */
