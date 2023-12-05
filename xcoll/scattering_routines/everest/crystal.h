// copyright ############################### #
// This file is part of the Xcoll Package.   #
// Copyright (c) CERN, 2023.                 #
// ######################################### #

#ifndef XCOLL_EVEREST_CRYSTAL_INTERACT_H
#define XCOLL_EVEREST_CRYSTAL_INTERACT_H
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double const tlcut_cry = 0.0009982;

double const aTF = 0.194e-10; // Screening function [m]
double const dP  = 1.920e-10; // Distance between planes (110) [m]
double const u1  = 0.075e-10; // Thermal vibrations amplitude

// pp cross-sections and parameters for energy dependence
double const pptref_cry = 0.040;
double const freeco_cry = 1.618;

// Processes
int const proc_out         =  -1;     // Crystal not hit
int const proc_AM          =   1;     // Amorphous
int const proc_VR          =   2;     // Volume reflection
int const proc_CH          =   3;     // Channeling
int const proc_VC          =   4;     // Volume capture
int const proc_absorbed    =   5;     // Absorption
int const proc_DC          =   6;     // Dechanneling
int const proc_pne         =   7;     // Proton-neutron elastic interaction
int const proc_ppe         =   8;     // Proton-proton elastic interaction
int const proc_diff        =   9;     // Single diffractive
int const proc_ruth        =  10;     // Rutherford scattering
int const proc_ch_absorbed =  15;     // Channeling followed by absorption
int const proc_ch_pne      =  17;     // Channeling followed by proton-neutron elastic interaction
int const proc_ch_ppe      =  18;     // Channeling followed by proton-proton elastic interaction
int const proc_ch_diff     =  19;     // Channeling followed by single diffractive
int const proc_ch_ruth     =  20;     // Channeling followed by Rutherford scattering
int const proc_TRVR        = 100;     // Volume reflection in VR-AM transition region
int const proc_TRAM        = 101;     // Amorphous in VR-AM transition region


/*gpufun*/
double* movech(RandomRutherfordData rng, LocalParticle* part, double nam, double dz, double x, double xp, double yp, double pc, double r, double rc, double rho, double anuc, double zatom, double emr, double hcut, double bnref, double csref0, double csref1, double csref5, double eUm, double collnt, double iProc) {

    double* result = (double*)malloc(5 * sizeof(double));

//     // Material properties
//     double const exenergy = CrystalMaterialData_get_excitation_energy(material);
//     double const rho      = CrystalMaterialData_get_density(material);
//     double const anuc     = CrystalMaterialData_get_A(material);
//     double const zatom    = CrystalMaterialData_get_Z(material);
//     double const emr      = CrystalMaterialData_get_nuclear_radius(material);
//     double const dlri     = CrystalMaterialData_get_crystal_radiation_length(material);
//     double const dlyi     = CrystalMaterialData_get_crystal_nuclear_length(material);
//     double const ai       = CrystalMaterialData_get_crystal_plane_distance(material);
//     double const eum      = CrystalMaterialData_get_crystal_potential(material);
//     double const collnt   = CrystalMaterialData_get_nuclear_collision_length(material);
//     double const hcut     = CrystalMaterialData_get_hcut(material);
//     double const bnref    = CrystalMaterialData_get_nuclear_elastic_slope(material);
//     double const csref0   = CrystalMaterialData_get_cross_section(material, 0);
//     double const csref1   = CrystalMaterialData_get_cross_section(material, 1);
//     double const csref5   = CrystalMaterialData_get_cross_section(material, 5);

    double pmap = 938.271998;
    double pc_in = pc;

    double cs[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double cprob[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double xran_cry[1] = {0.0};

    //New treatment of scattering routine based on standard sixtrack routine

    //Useful calculations for cross-section and event topology calculation
    double ecmsq  = ((2.*pmap)*1.0e-3)*pc;
    double xln15s = log(0.15*ecmsq);

    //New models, see Claudia's thesis
    double pptot = (0.041084 - 0.0023302*log(ecmsq)) + 0.00031514 * pow(log(ecmsq),2.);
    double ppel  = (11.7 - 1.59*log(ecmsq) + 0.134 * pow(log(ecmsq),2.))/1.0e3;
    double ppsd  = (4.3 + 0.3*log(ecmsq))/1.0e3;
    double bpp   = 7.156 + 1.439*log(sqrt(ecmsq));

    xran_cry[0] = RandomRutherford_generate(rng, part);

    //Rescale the total and inelastic cross-section accordigly to the average density seen
    double x_i = x;
    int np  = x_i/dP;    //Calculate in which crystalline plane the particle enters
    x_i = x_i - np*dP;   //Rescale the incoming x at the left crystalline plane
    x_i = x_i - (dP/2.); //Rescale the incoming x in the middle of crystalline planes

    double pv   = pow(pc,2.)/sqrt(pow(pc,2.) + pow((pmap*1.0e-3),2.))*1.0e9;          //Calculate pv=P/E
    double Ueff = eUm*((2.*x_i)/dP)*((2.*x_i)/dP) + pv*x_i/r; //Calculate effective potential
    double Et   = (pv*pow(xp,2.))/2. + Ueff;                  //Calculate transverse energy
    double Ec   = (eUm*(1.-rc/r))*(1.-rc/r);                  //Calculate critical energy in bent crystals

    //To avoid negative Et
    double xminU = ((-pow(dP,2.)*pc)*1.0e9)/(8.*eUm*r);
    double Umin  = fabs((eUm*((2.*xminU)/dP))*((2.*xminU)/dP) + pv*xminU/r);
    Et    = Et + Umin;
    Ec    = Ec + Umin;

    //Calculate min e max of the trajectory between crystalline planes
    double x_min = (-(dP/2.)*rc)/r - (dP/2.)*sqrt(Et/Ec);
    double x_max = (-(dP/2.)*rc)/r + (dP/2.)*sqrt(Et/Ec);

    //Change ref. frame and go back with 0 on the crystalline plane on the left
    x_min = x_min - dP/2.;
    x_max = x_max - dP/2.;

    //Calculate the "normal density" in m^-3
    double N_am  = ((rho*6.022e23)*1.0e6)/anuc;

    //Calculate atomic density at min and max of the trajectory oscillation
    // erf returns the error function of complex argument
    double rho_max = ((N_am*dP)/2.)*(erf(x_max/sqrt(2*pow(u1,2.))) - erf((dP-x_max)/sqrt(2*pow(u1,2.))));
    double rho_min = ((N_am*dP)/2.)*(erf(x_min/sqrt(2*pow(u1,2.))) - erf((dP-x_min)/sqrt(2*pow(u1,2.))));

    //"zero-approximation" of average nuclear density seen along the trajectory
    double avrrho  = (rho_max - rho_min)/(x_max - x_min);
    avrrho  = (2.*avrrho)/N_am;

    double csref_tot_rsc  = csref0*avrrho; //Rescaled total ref cs
    double csref_inel_rsc = csref1*avrrho; //Rescaled inelastic ref cs

    //Cross-section calculation
    double freep = freeco_cry * pow(anuc,1./3.);

    //compute pp and pn el+single diff contributions to cross-section (both added : quasi-elastic or qel later)
    cs[3] = freep*ppel;
    cs[4] = freep*ppsd;

    //correct TOT-CSec for energy dependence of qel
    //TOT CS is here without a Coulomb contribution
    cs[0] = csref_tot_rsc + freep*(pptot - pptref_cry);

    //Also correct inel-CS
    if(csref_tot_rsc == 0) {
        cs[1] = 0;
    } else {
        cs[1] = (csref_inel_rsc*cs[0])/csref_tot_rsc;
    }
        
    //Nuclear Elastic is TOT-inel-qel ( see definition in RPP)
    cs[2] = ((cs[0] - cs[1]) - cs[3]) - cs[4];
    cs[5] = csref5;

    //Now add Coulomb
    cs[0] = cs[0] + cs[5];

    //Calculate cumulative probability
    //cprob[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    cprob[5] = 1;
    
    if (cs[0] == 0) {
        int i;
        for (i = 1; i < 5; ++i) {
            cprob[i] = cprob[i-1];
        }
    } else {
        int i;
        for (i = 1; i < 5; ++i) {
            cprob[i] = cprob[i-1] + cs[i]/cs[0];
        }
    }

    //Multiple Coulomb Scattering
    xp = xp*1.0e3;
    yp = yp*1.0e3;

    //Turn on/off nuclear interactions
    if (nam == 0) {
        result[0] = x;
        result[1] = xp;
        result[2] = yp;
        result[3] = pc;
        result[4] = iProc;
        return result;
    }
    
    double nuc_cl_l;
    //Can nuclear interaction happen?
    //Rescaled nuclear collision length
    if (avrrho == 0) {
        nuc_cl_l = 1.0e6;
    } else {
        nuc_cl_l = collnt/avrrho;
    }

    double zlm = nuc_cl_l*RandomExponential_generate(part);

    //write(889,*) x_i,pv,Ueff,Et,Ec,N_am,avrrho,csref_tot_rsc,csref_inel_rsc,nuc_cl_l

    if (zlm < dz) {

        //Choose nuclear interaction
        double aran = RandomUniform_generate(part);
        int i = 1;
        double bn;
        double xm2;
        double bsd = 0.0;
        double teta;
        double tx;
        double tz;

        while (aran > cprob[i]) {
            i=i+1;
        }
        
        int ichoix = i;

        //Do the interaction
        double t = 0 ; //default value to cover ichoix=1
        
        if (ichoix==1) {
            iProc = proc_ch_absorbed; //deep inelastic, impinging p disappeared
        } else if (ichoix==2) { //p-n elastic
            iProc = proc_ch_pne;
            bn    = (bnref*cs[0])/csref_tot_rsc;
            t     = RandomExponential_generate(part)/bn;
        } else if (ichoix==3) { //p-p elastic
            iProc = proc_ch_ppe;
            t     = RandomExponential_generate(part)/bpp;
        } else if (ichoix==4) { //Single diffractive
            iProc = proc_ch_diff;
            xm2   = exp(RandomUniform_generate(part)*xln15s);
            pc    = pc*(1 - xm2/ecmsq);

            if (xm2 < 2.) {
                bsd = 2*bpp;
            } else if (xm2 >= 2. && xm2 <= 5.) {
                bsd = ((106.0 - 17.0*xm2)*bpp)/36.0;
            } else if (xm2 > 5.) {
                bsd = (7*bpp)/12.0;
            }
            //end if
            t = RandomExponential_generate(part)/bsd;
        } else { //(ichoix==5)
            iProc      = proc_ch_ruth;
            xran_cry[0] = RandomRutherford_generate(rng, part);
            t = xran_cry[0];
        }


        //Calculate the related kick -----------
        if (ichoix == 4) {
            teta = sqrt(t)/pc_in; //DIFF has changed PC!!!
        } else {
            teta = sqrt(t)/pc;
        }

        tx = teta*RandomNormal_generate(part)*1.0e3;
        tz = teta*RandomNormal_generate(part)*1.0e3;

        //Change p angle
        xp = xp + tx;
        yp = yp + tz;
    }

    xp = xp/1.0e3;
    yp = yp/1.0e3;

    result[0] = x;
    result[1] = xp;
    result[2] = yp;
    result[3] = pc;
    result[4] = iProc;
    return result;
}


/*gpufun*/
double* moveam(RandomRutherfordData rng, LocalParticle* part, double nam, double dz, double dei, double dlr, double xp, double yp, double pc, double anuc, double zatom, double emr, double hcut, double bnref, double csref0, double csref1, double csref5, double collnt, double iProc) {

    double* result = (double*)malloc(4 * sizeof(double));

//     // Material properties
//     double const exenergy = CrystalMaterialData_get_excitation_energy(material);
//     double const rho      = CrystalMaterialData_get_density(material);
//     double const anuc     = CrystalMaterialData_get_A(material);
//     double const zatom    = CrystalMaterialData_get_Z(material);
//     double const emr      = CrystalMaterialData_get_nuclear_radius(material);
//     double const dlri     = CrystalMaterialData_get_crystal_radiation_length(material);
//     double const dlyi     = CrystalMaterialData_get_crystal_nuclear_length(material);
//     double const ai       = CrystalMaterialData_get_crystal_plane_distance(material);
//     double const eum      = CrystalMaterialData_get_crystal_potential(material);
//     double const collnt   = CrystalMaterialData_get_nuclear_collision_length(material);
//     double const hcut     = CrystalMaterialData_get_hcut(material);
//     double const bnref    = CrystalMaterialData_get_nuclear_elastic_slope(material);
//     double const csref0   = CrystalMaterialData_get_cross_section(material, 0);
//     double const csref1   = CrystalMaterialData_get_cross_section(material, 1);
//     double const csref5   = CrystalMaterialData_get_cross_section(material, 5);

    double pc_in = pc;

    double pmap = 938.271998;

    double cs[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    double cprob[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

    // New treatment of scattering routine based on standard sixtrack routine
    // useful calculations for cross-section and event topology calculation
    double ecmsq  = 2.*pmap*1.0e-3*pc;
    double xln15s = log(0.15*ecmsq);

    // New models, see Claudia's thesis
    double pptot = 0.041084 - 0.0023302*log(ecmsq) + 0.00031514 * pow(log(ecmsq),2.);
    double ppel  = (11.7 - 1.59*log(ecmsq) + 0.134 * pow(log(ecmsq),2.))/1.0e3;
    double ppsd  = (4.3 + 0.3*log(ecmsq))/1.0e3;
    double bpp   = 7.156 + 1.439*log(sqrt(ecmsq));

    // Cross-section calculation
    // freep: number of nucleons involved in single scattering
    double freep = freeco_cry * pow(anuc,1./3.);

    // Compute pp and pn el+single diff contributions to cross-section (both added : quasi-elastic or qel later)
    cs[3] = freep*ppel;
    cs[4] = freep*ppsd;

    // Correct TOT-CSec for energy dependence of qel
    // TOT CS is here without a Coulomb contribution
    cs[0] = csref0 + freep*(pptot - pptref_cry);
    double bn = (bnref*cs[0])/csref0;

    // Also correct inel-CS
    cs[1] = (csref1*cs[0])/csref0;

    // Nuclear Elastic is TOT-inel-qel ( see definition in RPP)
    cs[2] = ((cs[0] - cs[1]) - cs[3]) - cs[4];
    cs[5] = csref5;

    // Now add Coulomb
    cs[0] = cs[0] + cs[5];

    // Calculate cumulative probability
    cprob[5] = 1;

    int i;
    for (i = 1; i < 5; ++i) {
        cprob[i] = cprob[i-1] + cs[i]/cs[0];
    }

    // Multiple Coulomb Scattering
    xp  = xp*1.0e3;
    yp  = yp*1.0e3;
    pc  = pc - dei*dz; // Energy lost because of ionization process[GeV]

    double dya   = (13.6/pc)*sqrt(dz/dlr); // RMS of coloumb scattering MCS (mrad)
    double kxmcs = dya*RandomNormal_generate(part);
    double kymcs = dya*RandomNormal_generate(part);

    xp = xp+kxmcs;
    yp = yp+kymcs;

    if (nam == 0) {

        result[0] = xp;
        result[1] = yp;
        result[2] = pc;
        result[3] = iProc;
        return result; // Turn on/off nuclear interactions
    }

    // Can nuclear interaction happen?
    double zlm = collnt*RandomExponential_generate(part);

    if (zlm < dz) {
        // Choose nuclear interaction
        double aran = RandomUniform_generate(part);
        int i=1;

        while (aran > cprob[i]) {
            i = i+1;
            //goto 10
        }

        int ichoix = i;

        // Do the interaction
        double t = 0 ;// default value to cover ichoix=1
        if (ichoix==1) {
        //case(1) // Deep inelastic, impinging p disappeared
            iProc = proc_absorbed;
        } else if (ichoix==2) { // p-n elastic
            iProc = proc_pne;
            t     = RandomExponential_generate(part)/bn;
        } else if (ichoix==3) { // p-p elastic
            iProc = proc_ppe;
            t     = RandomExponential_generate(part)/bpp;
        } else if (ichoix==4) { // Single diffractive
            iProc = proc_diff;
            double xm2 = exp(RandomUniform_generate(part)*xln15s);
            double bsd = 0.0;
            pc    = pc*(1 - xm2/ecmsq);

            if(xm2 < 2) {
                bsd = 2*bpp;
            } else if(xm2 >= 2 && xm2 <= 5) {
                bsd = ((106.0 - 17.0*xm2)*bpp)/36.0;
            } else if(xm2 > 5) {
                bsd = 7.0*bpp/12.0;
            }
        
            t = RandomExponential_generate(part)/bsd;
        } else { //(ichoix==5)
            iProc = proc_ruth;
            t = RandomRutherford_generate(rng, part);
        }

        // end select

        double teta;

        // Calculate the related kick
        if(ichoix == 4) {
            teta = sqrt(t)/pc_in ;// DIFF has changed PC
        } else {
            teta = sqrt(t)/pc;
        }

        double tx = teta*RandomNormal_generate(part)*1.0e3;
        double tz = teta*RandomNormal_generate(part)*1.0e3;

        // Change p angle
        xp = xp + tx;
        yp = yp + tz;
        
    }

    xp = xp/1.0e3;
    yp = yp/1.0e3;

    result[0] = xp;
    result[1] = yp;
    result[2] = pc;
    result[3] = iProc;
    return result; // Turn on/off nuclear interactions
}


/*gpufun*/
double calcionloss_cry(LocalParticle* part, double dz, double EnLo, double betar, double bgr, double gammar, double tmax, double plen, double exenergy, double zatom, double rho, double anuc) {

//     // Material properties
//     double const exenergy = CrystalMaterialData_get_excitation_energy(material);
//     double const rho      = CrystalMaterialData_get_density(material);
//     double const anuc     = CrystalMaterialData_get_A(material);
//     double const zatom    = CrystalMaterialData_get_Z(material);
//     double const emr      = CrystalMaterialData_get_nuclear_radius(material);
//     double const dlri     = CrystalMaterialData_get_crystal_radiation_length(material);
//     double const dlyi     = CrystalMaterialData_get_crystal_nuclear_length(material);
//     double const ai       = CrystalMaterialData_get_crystal_plane_distance(material);
//     double const eum      = CrystalMaterialData_get_crystal_potential(material);
//     double const collnt   = CrystalMaterialData_get_nuclear_collision_length(material);
//     double const hcut     = CrystalMaterialData_get_hcut(material);
//     double const bnref    = CrystalMaterialData_get_nuclear_elastic_slope(material);
//     double const csref0   = CrystalMaterialData_get_cross_section(material, 0);
//     double const csref1   = CrystalMaterialData_get_cross_section(material, 1);
//     double const csref5   = CrystalMaterialData_get_cross_section(material, 5);

    double k = 0.307075; // Constant in front bethe-bloch [mev g^-1 cm^2]
    double pmae = 0.51099890;
    double pmap = 938.271998;

    double thl  = (((((4.*k)*zatom)*dz)*1.0e2)*rho)/(anuc* pow(betar,2)); // [MeV]
    EnLo = ((k*zatom)/(anuc* pow(betar,2.))) * (0.5*log(((((2.*pmae)*bgr)*bgr)*tmax)/(1.0e6* pow(exenergy,2.))) -
            pow(betar,2.) - log(plen/(exenergy*1.0e3)) - log(bgr) + 0.5);
    EnLo = ((EnLo*rho)*1.0e-1)*dz; // [GeV]
    double Tt   = (EnLo*1.0e3)+thl; // [MeV]

    double cs_tail   = ((k*zatom)/(anuc* pow(betar,2.))) * ((0.5*((1/Tt)-(1./tmax))) - (log(tmax/Tt)*(pow(betar,2.))/(2.*tmax)) 
                        + ((tmax-Tt)/((4.*(pow(gammar,2.)))*(pow(pmap,2.)))));
    double prob_tail = ((cs_tail*rho)*dz)*1.0e2;

    if (RandomUniform_generate(part) < prob_tail) {
        EnLo = ((k*zatom)/(anuc*pow(betar,2.))) * (0.5*log((2*pmae*bgr*bgr*tmax)/(1.0e6*pow(exenergy,2.))) - pow(betar,2.) - 
                log(plen/(exenergy*1.0e3)) - log(bgr) + 0.5 + pow(tmax,2.)/(8.*(pow(gammar,2.))*(pow(pmap,2.))));

        EnLo = (EnLo*rho)*1.0e-1; // [GeV/m]
    } else {
        EnLo = EnLo/dz; // [GeV/m]
    }

    return EnLo;
}


/*gpufun*/
double* interact(RandomRutherfordData rng, LocalParticle* part, double x, double xp, double y, double yp, double pc, double length, double s_P, double x_P, double exenergy, double rho, double anuc, double zatom, double emr, double dlri, double dlyi, 
                double ai, double eUm, double collnt, double hcut, double csref0, double csref1, double csref5, double bnref,
                 double cry_tilt, double cry_rcurv, double cry_alayer, double cry_xmax, 
                double cry_ymax, double cry_orient, double cry_miscut, double cry_bend, double cry_cBend, double cry_sBend, double cry_cpTilt, double cry_spTilt, double cry_cnTilt, double cry_snTilt, double iProc) {

    double* result = (double*)malloc(6 * sizeof(double));

//     // Material properties
//     double const exenergy = CrystalMaterialData_get_excitation_energy(material);
//     double const rho      = CrystalMaterialData_get_density(material);
//     double const anuc     = CrystalMaterialData_get_A(material);
//     double const zatom    = CrystalMaterialData_get_Z(material);
//     double const emr      = CrystalMaterialData_get_nuclear_radius(material);
//     double const dlri     = CrystalMaterialData_get_crystal_radiation_length(material);
//     double const dlyi     = CrystalMaterialData_get_crystal_nuclear_length(material);
//     double const ai       = CrystalMaterialData_get_crystal_plane_distance(material);
//     double const eum      = CrystalMaterialData_get_crystal_potential(material);
//     double const collnt   = CrystalMaterialData_get_nuclear_collision_length(material);
//     double const hcut     = CrystalMaterialData_get_hcut(material);
//     double const bnref    = CrystalMaterialData_get_nuclear_elastic_slope(material);
//     double const csref0   = CrystalMaterialData_get_cross_section(material, 0);
//     double const csref1   = CrystalMaterialData_get_cross_section(material, 1);
//     double const csref5   = CrystalMaterialData_get_cross_section(material, 5);

    double dest = 0.;
    double pmap = 938.271998;
    double pmae = 0.51099890;
    double crade = 2.817940285e-15;

    double c_v1 =  1.7;   // Fitting coefficient
    double c_v2 = -1.5;   // Fitting coefficient

    int nam = 1; // Switch on/off the nuclear interaction (NAM) and the MCS (ZN)
    int zn  = 1;

    // dE/dX and dechanneling length calculation
    double mom    = pc*1.0e3;                // [GeV]
    double enr    = sqrt(pow(mom,2.) + pow(pmap,2.)); // [MeV]
    double gammar = enr/pmap;
    double betar  = mom/enr;
    double bgr    = betar*gammar;
    double mep    = pmae/pmap;  // Electron/proton

    double tmax = (2.*pmae*pow(bgr,2.))/(1. + 2.*gammar*mep + pow(mep,2.));  // [MeV]
    double plen = sqrt((rho*zatom)/anuc)*28.816e-6; // [MeV]

    double const_dech = ((256.0/(9*pow(M_PI,2.))) * (1./(log(((2.*pmae)*gammar)/(exenergy*1.0e3)) - 1.))) * ((aTF*dP)/(crade*pmae)); // [m/MeV]
    const_dech = const_dech*1.0e3; // [m/GeV]

    double s = 0.;
    double s_length = cry_rcurv*sin(length/cry_rcurv);
    double L_chan   = length;

    // MISCUT second step: fundamental coordinates (crystal edges and plane curvature radius)
    double s_K = cry_rcurv*sin(length/cry_rcurv);
    double x_K = cry_rcurv*(1.-cos(length/cry_rcurv));
    double s_M = (cry_rcurv-cry_xmax)*sin(length/cry_rcurv);
    double x_M = cry_xmax + (cry_rcurv-cry_xmax)*(1.-cos(length/cry_rcurv));
    double r   = sqrt(pow(s_P,2.) + pow((x-x_P),2.));

    // MISCUT third step: F coordinates (exit point) on crystal exit face
    double A_F = pow((tan(length/cry_rcurv)),2.) + 1.;
    double B_F = ((-2)*pow((tan(length/cry_rcurv)),2.))*cry_rcurv + (2.*tan(length/cry_rcurv))*s_P - 2.*x_P;
    double C_F = pow((tan(length/cry_rcurv)),2.)*(pow(cry_rcurv,2.)) - ((2.*tan(length/cry_rcurv))*s_P)*cry_rcurv + pow(s_P,2.) + pow(x_P,2.) - pow(r,2.);

    double x_F = (-B_F-sqrt(pow(B_F,2.)-4.*(A_F*C_F)))/(2.*A_F);
    double s_F = (-tan(length/cry_rcurv))*(x_F-cry_rcurv);

    if (x_F < x_K || x_F > x_M || s_F < s_M || s_F > s_K) {

        double alpha_F;
        double beta_F;
        
        if (cry_miscut == 0 && fabs(x_F-x_K) <= 1.0e-13 && fabs(s_F-s_K) <= 1.0e3) {
        // no miscut, entrance from below: exit point is K (lower edge)
            x_F = x_K;
            s_F = s_K;
        } else if (cry_miscut == 0 && fabs(x_F-x_M) <= 1.0e3 && fabs(s_F-s_M) <= 1.0e3) {
        // no miscut, entrance from above: exit point is M (upper edge)
            x_F = x_M;
            s_F = s_M;
        } else {
        // MISCUT Third step (bis): F coordinates (exit point)  on bent side
            if (cry_miscut < 0) {
            // Intersect with bottom side
                alpha_F = (cry_rcurv-x_P)/x_P;
                beta_F = -(pow(r,2.)-pow(s_P,2.)-pow(x_P,2.))/(2*s_P);
                A_F = pow(alpha_F,2.) + 1.;
                B_F = 2.*(alpha_F*beta_F) - 2.*cry_rcurv;
                C_F = pow(beta_F,2.);
            } else {
            // Intersect with top side
                alpha_F = (cry_rcurv-x_P)/s_P;
                beta_F = -(pow(r,2.)+cry_xmax*(cry_xmax-(2.*cry_rcurv))-pow(s_P,2.)-pow(x_P,2.))/(2.*s_P);
                A_F = pow(alpha_F,2.) + 1.;
                B_F = 2.*(alpha_F*beta_F) - 2.*cry_rcurv;
                C_F = pow(beta_F,2.) - cry_xmax*(cry_xmax-2.*cry_rcurv);
            }
            
            x_F = (-B_F-sqrt(pow(B_F,2.)-4.*(A_F*C_F)))/(2.*A_F);
            s_F = alpha_F*x_F + beta_F;
        }
    }

    // MISCUT 4th step: deflection and length calculation
    double a = sqrt(pow(s_F,2.)+pow((x-x_F),2.));
    double tP = acos((2.*pow(r,2.)-pow(a,2.))/(2.*pow(r,2.)));
    double tdefl = asin((s_F-s_P)/r);
    L_chan = r*tP;

    double xp_rel = xp - cry_miscut;

    double ymin = -cry_ymax/2.;
    double ymax =  cry_ymax/2.;

    // FIRST CASE: p don't interact with crystal
    if (y < ymin || y > ymax || x > cry_xmax) {
        x  = x + xp*s_length;
        y     = y + yp*s_length;
        iProc = proc_out;
        result[0] = x;
        result[1] = xp;
        result[2] = y;
        result[3] = yp;
        result[4] = pc;
        result[5] = iProc;
        return result;

    } else if (x < cry_alayer || y-ymin < cry_alayer || ymax-y < cry_alayer) {
    // SECOND CASE: p hits the amorphous layer
        double x0    = x;
        double y0    = y;
        double a_eq  = 1. + pow(xp,2.);
        double b_eq  = (2.*x)*xp - (2.*xp)*cry_rcurv;
        double c_eq  = pow(x,2.) - (2.*x)*cry_rcurv;
        double delta = pow(b_eq,2.) - (4.*a_eq)*c_eq;
        s = (-b_eq+sqrt(delta))/(2.*a_eq);
        if (s >= s_length) {
            s = s_length;
        }
        x   =  xp*s + x0;
        double len_xs = sqrt(pow((x-x0),2.) + pow(s,2.));
        double len_ys;
        if (yp >= 0 && y + yp*s <= ymax) {
            len_ys = yp*len_xs;
        } else if (yp < 0 && y + yp*s >= ymin) {
            len_ys = yp*len_xs;
        } else {
            s      = (ymax-y)/yp;
            len_ys = sqrt(pow((ymax-y),2.) + pow(s,2.));
            x   = x0 + xp*s;
            len_xs = sqrt(pow((x-x0),2.) + pow(s,2.));
        }
        
        double am_len = sqrt(pow(len_xs,2.) + pow(len_ys,2.));
        s     = s/2;
        x  = x0 + xp*s;
        y     = y0 + yp*s;
        iProc = proc_AM;

        dest = calcionloss_cry(part,s_length,dest,betar,bgr,gammar,tmax,plen,
                            exenergy,zatom,rho,anuc);

        double* result_am = moveam(rng, part,nam,am_len,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,
                            bnref,csref0,csref1,csref5,collnt,iProc);
        xp = result_am[0];
        yp = result_am[1];
        pc = result_am[2];
        iProc = result_am[3];
        free(result_am);

        x = x + xp*(s_length-s);
        y = y + yp*(s_length-s);

        result[0] = x;
        result[1] = xp;
        result[2] = y;
        result[3] = yp;
        result[4] = pc;
        result[5] = iProc;
        return result;

    } else if (x > cry_xmax-cry_alayer && x < cry_xmax) {
        iProc = proc_AM;
        
        dest = calcionloss_cry(part,s_length,dest,betar,bgr,gammar,tmax,plen,
                            exenergy,zatom,rho,anuc);

        double* result_am = moveam(rng, part,nam,s_length,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,
                        csref1,csref5,collnt,iProc);
        xp = result_am[0];
        yp = result_am[1];
        pc = result_am[2];
        iProc = result_am[3];
        free(result_am);

        result[0] = x;
        result[1] = xp;
        result[2] = y;
        result[3] = yp;
        result[4] = pc;
        result[5] = iProc;
        return result;
    }
    
    //THIRD CASE: the p interacts with the crystal.
    //Define typical angles/probabilities for orientation 110
    double xc      = ai*1.e-3*0.95;               // 95% saturation
    double Rcrit   = pc*1.e9/(2.0*eUm)*xc;  // Critical curvature radius [m]
    double xpcrit0 = sqrt(2.0e-9*eUm/pc); // Critical angle (rad) for straight crystals

    //If R>Rcritical=>no channeling is possible (ratio<1)
    double ratio  = cry_rcurv/Rcrit;
    double xpcrit = xpcrit0*(1-Rcrit/cry_rcurv); //Critical angle for curved crystal
    double lambda = sqrt(xc*Rcrit);  // absorbed factor 2pi

    double Ang_rms;
    double Ang_avr;
    double Vcapt;
    if (ratio <= 1.) { //no possibile channeling
        Ang_rms = c_v1*0.42*xpcrit0*sin(1.4*ratio);           // RMS scattering
        Ang_avr = c_v2*xpcrit0*5.0e-2*ratio;                  // Average angle reflection
        Vcapt   = 0.;                                         // Probability of VC
    } else if (ratio <= 3.) { //Strongly bent crystal
        Ang_rms = c_v1*0.42*xpcrit0*sin(0.4713*ratio + 0.85); // RMS scattering
        Ang_avr = c_v2*xpcrit0*(0.1972*ratio - 0.1472);       // Average angle reflection
        Vcapt   = 7.0e-4*(ratio - 0.7)/pow(pc,2.0e-1);        // Correction by sasha drozdin/armen
        //K=0.0007 is taken based on simulations using CATCH.f (V.Biryukov)
    } else { //Rcry >> Rcrit
        Ang_rms = c_v1*xpcrit0/ratio;                         // RMS scattering
        Ang_avr = c_v2*xpcrit0*(1. - 1.6667/ratio);           // Average angle for VR
        Vcapt   = 7.0e-4*(ratio - 0.7)/pow(pc,2.0e-1);        // Probability for VC correction by sasha drozdin/armen
        //K=0.0007 is taken based on simulations using CATCH.f (V.Biryukov)
    }
    if (cry_orient == 2) {
        Ang_avr = Ang_avr*0.93;
        Ang_rms = Ang_rms*1.05;
        xpcrit  = xpcrit*0.98;
    }
    if (fabs(xp_rel) < xpcrit) {

        //if they can channel: 2 options
        double xi1 = RandomUniform_generate(part)*cry_rcurv/(cry_rcurv-Rcrit)*ai/xc;
        if (xi1 <= 1 && fabs(xp_rel)/xpcrit <= 2*sqrt(xi1)*sqrt(1-xi1)) {
            //option 1:channeling

            double TLdech1 = (const_dech*pc)*pow((1.-1./ratio),2.); //Updated calculate typical dech. length(m)
            double N_atom = 1.0e-1;
            if(RandomUniform_generate(part) <= N_atom) {
                TLdech1 = ((const_dech/2.0e2)*pc)*pow((1.-1./ratio),2.);  //Updated dechanneling length (m)      
            }
            double Dechan = RandomExponential_generate(part); //Probability of dechanneling
            double Ldech  = TLdech1*Dechan;   //Actual dechan. length

            //careful: the dechanneling lentgh is along the trajectory
            //of the particle -not along the longitudinal coordinate...
            if (Ldech < L_chan) {
                iProc = proc_DC;
                double Dxp   = Ldech/r; //Change angle from channeling [mrad]
                double Sdech = Ldech*cos(cry_miscut + 0.5*Dxp);
                x  = x  + Ldech*(sin(0.5*Dxp+cry_miscut)); //Trajectory at channeling exit
                xp = xp + Dxp + (2*(RandomUniform_generate(part)-0.5))*xpcrit;
                y  = y  + yp * Sdech;

                dest = calcionloss_cry(part,Ldech,dest,betar,bgr,gammar,tmax,plen,
                                    exenergy,zatom,rho,anuc);
                pc = pc - 0.5*dest*Ldech; //Energy loss to ionization while in CH [GeV]
                x  = x  + (0.5*(s_length-Sdech))*xp;
                y  = y  + (0.5*(s_length-Sdech))*yp;

                dest = calcionloss_cry(part,s_length-Sdech,dest,betar,bgr,gammar,tmax,plen,exenergy,zatom,rho,anuc);

                double* result_am = moveam(rng,part,nam,s_length-Sdech,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,csref1,csref5,collnt,iProc);
                xp = result_am[0];
                yp = result_am[1];
                pc = result_am[2];
                iProc = result_am[3];
                free(result_am);

                x = x + (0.5*(s_length-Sdech))*xp;
                y = y + (0.5*(s_length-Sdech))*yp;
            } else {
                iProc = proc_CH;
                double xpin  = xp;
                double ypin  = yp;

                //check if a nuclear interaction happen while in CH
                double* result_ch = movech(rng,part,nam,L_chan,x,xp,yp,pc,cry_rcurv,Rcrit,rho,anuc,zatom,emr,hcut,bnref,
                                csref0,csref1,csref5,eUm,collnt,iProc);
                x = result_ch[0];
                xp = result_ch[1];
                yp = result_ch[2];
                pc = result_ch[3];
                iProc = result_ch[4];
                free(result_ch);

                if (iProc != proc_CH) {
                    //if an nuclear interaction happened, move until the middle with initial xp,yp:
                    //propagate until the "crystal exit" with the new xp,yp accordingly with the rest
                    //of the code in "thin lens approx"
                    x = x + (0.5*L_chan)*xpin;
                    y = y + (0.5*L_chan)*ypin;
                    x = x + (0.5*L_chan)*xp;
                    y = y + (0.5*L_chan)*yp;

                    dest = calcionloss_cry(part,length,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    pc = pc - dest*length; //energy loss to ionization [GeV]
                } else {
                    double xi2 = RandomUniform_generate(part)*2*M_PI;
//                     double t_out = xpcrit*cos(xi2)*sqrt(pow(2*xi1-1, 2) + pow(xp_rel/xpcrit, 2));
                    double t_out = xp_rel*cos(L_chan/lambda) - xpcrit*(2*xi1-1)*sin(L_chan/lambda);
//                     t_out = xp_rel*cos(xi2) - xpcrit*(2*xi1-1)*sin(xi2);
                    xp = tdefl + t_out; //Change angle[rad]
                    x = x + L_chan*(sin(0.5*xp)); //Trajectory at channeling exit
                    y   = y + s_length * yp;

                    dest = calcionloss_cry(part,length,dest,betar,bgr,gammar,tmax,plen,exenergy,zatom,rho,anuc);
                    pc = pc - (0.5*dest)*length; //energy loss to ionization [GeV]     
                } 
            }
        } else { //Option 2: VR
            //good for channeling but don't channel (1-2)
            iProc = proc_VR;

            xp = xp + (0.45*(xp_rel/xpcrit + 1))*Ang_avr;
            x  = x  + (0.5*s_length)*xp;
            y  = y  + (0.5*s_length)*yp;

            dest = calcionloss_cry(part,s_length,dest,betar,bgr,gammar,tmax,plen,exenergy,zatom,rho,anuc);
            double* result_am = moveam(rng, part,nam,s_length,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,csref1,csref5,collnt,iProc);
            xp = result_am[0];
            yp = result_am[1];
            pc = result_am[2];
            iProc = result_am[3];
            free(result_am);

            x = x + (0.5*s_length)*xp;
            y = y + (0.5*s_length)*yp;  
        }
    } else { //case 3-2: no good for channeling. check if the  can VR
        double Lrefl = xp_rel*r; //Distance of refl. point [m]
        double Srefl = sin(xp_rel/2. + cry_miscut)*Lrefl;

        if (Lrefl > 0 && Lrefl < L_chan) { //VR point inside

        //2 options: volume capture and volume reflection

            if (RandomUniform_generate(part) > Vcapt || zn == 0) { //Option 1: VR
                iProc = proc_VR;
                x  = x + xp*Srefl;
                y     = y + yp*Srefl;
                double Dxp   = Ang_avr;
                xp    = xp + Dxp + Ang_rms*RandomNormal_generate(part);
                x  = x  + (0.5*xp)*(s_length - Srefl);
                y     = y  + (0.5*yp)*(s_length - Srefl);

                dest = calcionloss_cry(part,s_length-Srefl,dest,betar,bgr,gammar,tmax,plen,exenergy,zatom,rho,anuc);
                double* result_am = moveam(rng,part,nam,s_length-Srefl,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,csref1,csref5,collnt,iProc);
                xp = result_am[0];
                yp = result_am[1];
                pc = result_am[2];
                iProc = result_am[3];
                free(result_am);

                x = x + (0.5*xp)*(s_length - Srefl);
                y = y + (0.5*yp)*(s_length - Srefl);

            } else { //Option 2: VC
                x = x + xp*Srefl;
                y = y + yp*Srefl;

                double TLdech2 = (const_dech/1.0e1)*pc*pow((1-1/ratio),2.) ;         //Updated typical dechanneling length(m)
                double Ldech   = TLdech2 * pow((sqrt(1.0e-2 + RandomExponential_generate(part)) - 1.0e-1),2.); //Updated DC length
                double tdech   = Ldech/cry_rcurv;
                double Sdech   = Ldech*cos(xp + 0.5*tdech);

                double xi1 = RandomUniform_generate(part);
                double xi2 = RandomUniform_generate(part)*2*M_PI;
                double t_out = xpcrit*cos(xi2)*sqrt(pow(2*xi1-1, 2) + pow(xp_rel/xpcrit, 2));
                t_out = (0.5*RandomNormal_generate(part))*xpcrit;
                if (Ldech < length-Lrefl) {
                    iProc = proc_DC;
                    double Dxp   = Ldech/cry_rcurv + t_out;
                    x  = x + Ldech*(sin(0.5*Dxp+xp)); //Trajectory at channeling exit
                    y     = y + Sdech*yp;
                    xp    =  Dxp;
                    double Red_S = (s_length - Srefl) - Sdech;
                    x  = x + (0.5*xp)*Red_S;
                    y     = y + (0.5*yp)*Red_S;

                    dest = calcionloss_cry(part,Srefl,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);

                    pc = pc - dest*Srefl; //"added" energy loss before capture

                    dest = calcionloss_cry(part,Sdech,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    pc = pc - (0.5*dest)*Sdech; //"added" energy loss while captured

                    dest = calcionloss_cry(part,Red_S,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    double* result_am = moveam(rng,part,nam,Red_S,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,
                        csref1,csref5,collnt,iProc);
                    xp = result_am[0];
                    yp = result_am[1];
                    pc = result_am[2];
                    iProc = result_am[3];
                    free(result_am);
                    
                    x = x + (0.5*xp)*Red_S;
                    y = y + (0.5*yp)*Red_S;
                } else {
                    iProc   = proc_VC;
                    double Rlength = length - Lrefl;
                    double tchan   = Rlength/cry_rcurv;
                    double Red_S   = Rlength*cos(xp + 0.5*tchan);

                    dest = calcionloss_cry(part,Lrefl,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    pc   = pc - dest*Lrefl; //"added" energy loss before capture
                    double xpin = xp;
                    double ypin = yp;

                    //Check if a nuclear interaction happen while in ch
                    double* result_ch = movech(rng,part,nam,Rlength,x,xp,yp,pc,cry_rcurv,Rcrit,rho,anuc,zatom,emr,hcut,bnref,
                                    csref0,csref1,csref5,eUm,collnt,iProc);
                    x = result_ch[0];
                    xp = result_ch[1];
                    yp = result_ch[2];
                    pc = result_ch[3];
                    iProc = result_ch[4];
                    free(result_ch);
                                    
                    if (iProc != proc_VC) {
                        //if an nuclear interaction happened, move until the middle with initial xp,yp: propagate until
                        //the "crystal exit" with the new xp,yp aciordingly with the rest of the code in "thin lens approx"
                        x = x + (0.5*Rlength)*xpin;
                        y = y + (0.5*Rlength)*ypin;
                        x = x + (0.5*Rlength)*xp;
                        y = y + (0.5*Rlength)*yp;

                        dest = calcionloss_cry(part,Rlength,dest,betar,bgr,gammar,tmax,plen,
                                            exenergy,zatom,rho,anuc);
                        pc = pc - dest*Rlength;
                    } else {
                        double Dxp = (length-Lrefl)/cry_rcurv;
                        x = x + sin(0.5*Dxp+xp)*Rlength; //Trajectory at channeling exit
                        y   = y + Red_S*yp;
                        xp  = tdefl + t_out; //[mrad]

                        dest = calcionloss_cry(part,Rlength,dest,betar,bgr,gammar,tmax,plen,
                                            exenergy,zatom,rho,anuc);
                        pc = pc - (0.5*dest)*Rlength;  //"added" energy loss once captured
                    }
                }
            }
        } else {
            //Case 3-3: move in amorphous substance (big input angles)
            //Modified for transition vram daniele
            if (xp_rel > tdefl-cry_miscut + 2*xpcrit || xp_rel < -xpcrit) {
                iProc = proc_AM;
                x  = x + (0.5*s_length)*xp;
                y     = y + (0.5*s_length)*yp;
                if(zn > 0) {
                    dest = calcionloss_cry(part,s_length,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    double* result_am = moveam(rng,part,nam,s_length,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,
                        csref1,csref5,collnt,iProc);
                    xp = result_am[0];
                    yp = result_am[1];
                    pc = result_am[2];
                    iProc = result_am[3];
                    free(result_am);
                }
            
                x = x + (0.5*s_length)*xp;
                y = y + (0.5*s_length)*yp;
            } else {
                double Pvr = (xp_rel-(tdefl-cry_miscut))/(2.*xpcrit);
                if(RandomUniform_generate(part) > Pvr) {

                    iProc = proc_TRVR;
                    x  = x + xp*Srefl;
                    y     = y + yp*Srefl;

                    double Dxp = (((-3.*Ang_rms)*xp_rel)/(2.*xpcrit) + Ang_avr) + ((3.*Ang_rms)*(tdefl-cry_miscut))/(2.*xpcrit);
                    xp  = xp + Dxp;
                    x = x + (0.5*xp)*(s_length-Srefl);
                    y   = y + (0.5*yp)*(s_length-Srefl);

                    dest = calcionloss_cry(part,s_length-Srefl,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    double* result_am = moveam(rng,part,nam,s_length-Srefl,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,
                        csref1,csref5,collnt,iProc);
                    xp = result_am[0];
                    yp = result_am[1];
                    pc = result_am[2];
                    iProc = result_am[3];
                    free(result_am);

                    x = x + (0.5*xp)*(s_length - Srefl);
                    y = y + (0.5*yp)*(s_length - Srefl);
                } else {
                    iProc = proc_TRAM;
                    x = x + xp*Srefl;
                    y = y + yp*Srefl;
                    double Dxp = ((((-1.*(13.6/pc))*sqrt(s_length/dlri))*1.0e-3)*xp_rel)/(2.*xpcrit) + (((13.6/pc)*sqrt(s_length/dlri))*1.0e-3)*(1.+(tdefl-cry_miscut)/(2.*xpcrit));
                    xp = xp+Dxp;
                    x  = x + (0.5*xp)*(s_length-Srefl);
                    y  = y + (0.5*yp)*(s_length-Srefl);

                    dest = calcionloss_cry(part,s_length-Srefl,dest,betar,bgr,gammar,tmax,plen,
                                        exenergy,zatom,rho,anuc);
                    double* result_am = moveam(rng,part,nam,s_length-Srefl,dest,dlri,xp,yp,pc,anuc,zatom,emr,hcut,bnref,csref0,
                        csref1,csref5,collnt,iProc);
                    xp = result_am[0];
                    yp = result_am[1];
                    pc = result_am[2];
                    iProc = result_am[3];
                    free(result_am);

                    x = x + (0.5*xp)*(s_length - Srefl);
                    y = y + (0.5*yp)*(s_length - Srefl);
                }
            }
        }            
    }

    result[0] = x;
    result[1] = xp;
    result[2] = y;
    result[3] = yp;
    result[4] = pc;
    result[5] = iProc;
    return result;
}


/*gpufun*/
double* crystal(RandomRutherfordData rng, LocalParticle* part, double x, double xp, double z, double zp, double s, double p, 
                double x0, double xp0, double zlm, double s_imp, int isimp, double val_part_hit, double val_part_abs, 
                double val_part_impact, double val_part_indiv, double c_length, CrystalMaterialData material, double nhit, 
                double nabs, double cry_tilt, double  cry_rcurv, double  cry_bend, double  cry_alayer, double  cry_xmax, 
                double cry_ymax, double cry_orient, double cry_miscut, double iProc, 
                double n_chan, double n_VR, double n_amorphous) {

    double const exenergy = CrystalMaterialData_get_excitation_energy(material);
    double const rho      = CrystalMaterialData_get_density(material);
    double const anuc     = CrystalMaterialData_get_A(material);
    double const zatom    = CrystalMaterialData_get_Z(material);
    double const emr      = CrystalMaterialData_get_nuclear_radius(material);
    double const dlri     = CrystalMaterialData_get_crystal_radiation_length(material);
    double const dlyi     = CrystalMaterialData_get_crystal_nuclear_length(material);
    double const ai       = CrystalMaterialData_get_crystal_plane_distance(material);
    double const eum      = CrystalMaterialData_get_crystal_potential(material);
    double const collnt   = CrystalMaterialData_get_nuclear_collision_length(material);
    double const hcut     = CrystalMaterialData_get_hcut(material);
    double const bnref    = CrystalMaterialData_get_nuclear_elastic_slope(material);
    double const csref0   = CrystalMaterialData_get_cross_section(material, 0);
    double const csref1   = CrystalMaterialData_get_cross_section(material, 1);
    double const csref5   = CrystalMaterialData_get_cross_section(material, 5);

    double s_temp     = 0.;
    double s_shift    = 0.;
    double s_rot      = 0.;
    double s_int      = 0.;
    double x_temp     = 0.;
    double x_shift    = 0.;
    double x_rot      = 0.;
    double x_int      = 0.;
    double xp_temp    = 0.;
    double xp_shift   = 0.;
    double xp_rot     = 0.;
    double xp_int     = 0.;
    double xp_tangent = 0.;
    double tilt_int   = 0.;
    double shift      = 0.;
    double delta      = 0.;
    double a_eq       = 0.;
    double b_eq       = 0.;
    double c_eq       = 0.;
    double cry_length  = c_length;

    s_imp  = 0.;
    iProc  = proc_out;

    double* crystal_result = (double*)malloc(21 * sizeof(double));

    double const cry_cBend  = cos(cry_bend);
    double const cry_sBend  = sin(cry_bend);

    // Transform in the crystal reference system
    // 1st transformation: shift of the center of the reference frame
    double const cry_cpTilt = cos(cry_tilt);
    double const cry_spTilt = sin(cry_tilt);
    double const cry_cnTilt = cry_cpTilt;
    double const cry_snTilt = -cry_spTilt;
    if (cry_tilt < 0) {
        s_shift = s;
        shift   = cry_rcurv*(1 - cry_cpTilt);

        if (cry_tilt < -cry_bend) {
            shift = cry_rcurv*(cry_cnTilt - cos(cry_bend - cry_tilt));
        }
        x_shift = x - shift;
    } else {
        s_shift = s;
        x_shift = x;
    }
  
    // 2nd transformation: rotation
    s_rot  = x_shift*cry_spTilt + s_shift*cry_cpTilt;
    x_rot  = x_shift*cry_cpTilt - s_shift*cry_spTilt;
    xp_rot = xp - cry_tilt;

    // 3rd transformation: drift to the new coordinate s=0
    xp = xp_rot;
    x  = x_rot - xp_rot*s_rot;
    z  = z - zp*s_rot;
    s  = 0.;
  
    // Check that particle hit the crystal
    if (x >= 0. && x < cry_xmax) {
        // MISCUT first step: P coordinates (center of curvature of crystalline planes)
        double s_P = (cry_rcurv-cry_xmax)*sin(-cry_miscut);
        double x_P = cry_xmax + (cry_rcurv-cry_xmax)*cos(-cry_miscut);

        double* result = interact(rng,part,x,xp,z,zp,p,cry_length,s_P,x_P,exenergy,rho,anuc,zatom,emr,dlri,dlyi,
                        ai,eum,collnt,hcut,csref0,csref1,csref5,bnref,cry_tilt,
                        cry_rcurv,cry_alayer,cry_xmax,cry_ymax,cry_orient,cry_miscut,cry_bend,cry_cBend,
                        cry_sBend,cry_cpTilt,cry_spTilt,cry_cnTilt,cry_snTilt,iProc);

        x = result[0];
        xp = result[1];
        z = result[2];
        zp = result[3];
        p = result[4];
        iProc = result[5];
        free(result);

        s   = cry_rcurv*cry_sBend;
        zlm = cry_rcurv*cry_sBend;

        if (iProc != proc_out) {
            isimp    = 1;
            nhit     = nhit + 1.;
            val_part_hit     = 1.;
            val_part_impact   = x0;
            val_part_indiv    = xp0;
        }

    } else {

        if (x < 0) { // Crystal can be hit from below
            xp_tangent = sqrt((-(2.*x)*cry_rcurv + pow(x,2.))/(pow(cry_rcurv,2.)));
        } else {             // Crystal can be hit from above
            xp_tangent = asin((cry_rcurv*(1. - cry_cBend) - x)/sqrt(((2.*cry_rcurv)*(cry_rcurv - x))*(1 - cry_cBend) + pow(x,2.)));
        }
        // If the hit is below, the angle must be greater or equal than the tangent,
        // or if the hit is above, the angle must be smaller or equal than the tangent
        if ((x < 0. && xp >= xp_tangent) || (x >= 0. && xp <= xp_tangent)) {

            // If it hits the crystal, calculate in which point and apply the transformation and drift to that point
            a_eq  = 1 + pow(xp,2.);
            b_eq  = (2.*xp)*(x - cry_rcurv);
            c_eq  = -(2.*x)*cry_rcurv + pow(x,2.);
            delta = pow(b_eq,2.) - 4*(a_eq*c_eq);
            s_int = (-b_eq - sqrt(delta))/(2*a_eq);
            s_imp = s_int;

            // MISCUT first step: P coordinates (center of curvature of crystalline planes)
            double s_P_tmp = (cry_rcurv-cry_xmax)*sin(-cry_miscut);
            double x_P_tmp = cry_xmax + (cry_rcurv-cry_xmax)*cos(-cry_miscut);

            if (s_int < cry_rcurv*cry_sBend) {
                // Transform to a new reference system: shift and rotate
                x_int  = xp*s_int + x;
                xp_int = xp;
                z      = z + zp*s_int;
                x      = 0.;
                s      = 0.;

                tilt_int = s_int/cry_rcurv;
                xp    = xp-tilt_int;

                // MISCUT first step (bis): transform P in new reference system
                // Translation
                s_P_tmp = s_P_tmp - s_int;
                x_P_tmp = x_P_tmp - x_int;
                // Rotation
                double s_P = s_P_tmp*cos(tilt_int) + x_P_tmp*sin(tilt_int);
                double x_P = -s_P_tmp*sin(tilt_int) + x_P_tmp*cos(tilt_int);

                double* result = interact(rng,part,x,xp,z,zp,p,cry_length-(tilt_int*cry_rcurv),s_P,x_P,exenergy,rho,anuc,
                                zatom,emr,dlri,dlyi,ai,eum,collnt,hcut,csref0,csref1,csref5,bnref,
                                cry_tilt,cry_rcurv,cry_alayer,cry_xmax,cry_ymax,cry_orient,cry_miscut,
                                cry_bend,cry_cBend,cry_sBend,cry_cpTilt,cry_spTilt,cry_cnTilt,cry_snTilt,iProc);

                x = result[0];
                xp = result[1];
                z = result[2];
                zp = result[3];
                p = result[4];
                iProc = result[5];
                free(result);

                s   = cry_rcurv*sin(cry_bend - tilt_int);
                zlm = cry_rcurv*sin(cry_bend - tilt_int);
                
                if (iProc != proc_out) {
                    x_rot    = x_int;
                    s_rot    = s_int;
                    xp_rot   = xp_int;
                    s_shift  =  s_rot*cry_cnTilt + x_rot*cry_snTilt;
                    x_shift  = -s_rot*cry_snTilt + x_rot*cry_cnTilt;
                    xp_shift = xp_rot + cry_tilt;

                    if (cry_tilt < 0.) {
                        x0  = x_shift + shift;
                        xp0 = xp_shift;
                    }
                    else {
                        x0  = x_shift;
                        xp0 = xp_shift;
                    }
                    isimp     = 1;
                    nhit      = nhit + 1.;
                    val_part_hit    = 1.;
                    val_part_impact = x0;
                    val_part_indiv  = xp0;
                    //
                }

                // un-rotate
                x_temp  = x;
                s_temp  = s;
                xp_temp = xp;
                s       =  s_temp*cos(-tilt_int) + x_temp*sin(-tilt_int);
                x    = -s_temp*sin(-tilt_int) + x_temp*cos(-tilt_int);
                xp   = xp_temp + tilt_int;

                // 2nd: shift back the 2 axis
                x = x + x_int;
                s = s + s_int;
            } else {
                s = cry_rcurv*sin(cry_length/cry_rcurv);
                x = x + s*xp;
                z = z + s*zp;
            }
        } else {
            s = cry_rcurv*sin(cry_length/cry_rcurv);
            x = x + s*xp;
            z = z + s*zp;
        }
    }

    // transform back from the crystal to the collimator reference system
    // 1st: un-rotate the coordinates
    x_rot  = x;
    s_rot  = s;
    xp_rot = xp;

    s_shift  =  s_rot*cry_cnTilt + x_rot*cry_snTilt;
    x_shift  = -s_rot*cry_snTilt + x_rot*cry_cnTilt;
    xp_shift = xp_rot + cry_tilt;

    // 2nd: shift back the reference frame
    if (cry_tilt < 0) {
        s  = s_shift;
        x  = x_shift + shift;
        xp = xp_shift;
    } else {
        x  = x_shift;
        s  = s_shift;
        xp = xp_shift;
    }
    //

    // 3rd: shift to new S=Length position
    x = xp*(c_length - s) + x;
    z = zp*(c_length - s) + z;
    s = c_length;

    nabs = 0;
  
    if (iProc == proc_AM) {
        n_amorphous = n_amorphous + 1;
    } else if (iProc == proc_VR) {
        n_VR = n_VR + 1;
    } else if (iProc == proc_CH) {
        n_chan = n_chan + 1;
    } else if (iProc == proc_absorbed) {
        nabs = 1;   // TODO: do we need to set part_abs_pos etc?
    } else if (iProc == proc_ch_absorbed) {
        nabs = 1;
    }

    crystal_result[0] = val_part_hit;
    crystal_result[1] = val_part_abs;
    crystal_result[2] = val_part_impact;
    crystal_result[3] = val_part_indiv;
    crystal_result[4] = nhit;
    crystal_result[5] = nabs;
    crystal_result[6] = s_imp;
    crystal_result[7] = isimp;
    crystal_result[8] = s;
    crystal_result[9] = zlm;
    crystal_result[10] = x;
    crystal_result[11] = xp;
    crystal_result[12] = x0;
    crystal_result[13] = xp0;
    crystal_result[14] = z;
    crystal_result[15] = zp;
    crystal_result[16] = p;
    crystal_result[17] = iProc;
    crystal_result[18] = n_chan;
    crystal_result[19] = n_VR;
    crystal_result[20] = n_amorphous;
    return crystal_result;
}

#endif /* XCOLL_EVEREST_CRYSTAL_INTERACT_H */
