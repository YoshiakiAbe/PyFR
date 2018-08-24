# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

#include <stdio.h>

<% tau = c['ldg-tau'] %>

<%pyfr:macro name='bc_impose_state' params='ul, nl, ur, urand, urande'>
    fpdtype_t pl = ${c['gamma'] - 1.0} * (ul[${nvars - 1}]
                 - (0.5 / ul[0]) * ${pyfr.dot('ul[{i}]', i=(1, ndims + 1))});
    fpdtype_t pt = (${c['pt']});
    pt = (pt > pl ? pt : pl);

    fpdtype_t udotu = ${2.0 * c['cpTt']} * (1.0
                    - pow(pt, ${(-c['Rdcp'])}) * pow(pl, ${c['Rdcp']}));
    fpdtype_t udotu_fluc = ${pyfr.dot('urand[{i}]', i=(0, ndims))};
    udotu = (udotu > 0 ? udotu : 0);

    ur[0] = ${1.0 / c['Rdcp']} * pl / (${c['cpTt']} - 0.5 * (udotu + udotu_fluc));
% for i, v in enumerate(c['vc']):
    ur[${i + 1}] = ${v} * ur[0] * sqrt(udotu) + ur[0] * urand[${i}];
% endfor
    ur[${nvars - 1}] = ${1.0 / (c['gamma'] - 1.0)} * pl + 0.5 * ur[0] * (udotu + udotu_fluc);
</%pyfr:macro>


<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t, bby, bbz, InitFlag, runi, runi_prev, uspr, uspr_prev, urand, urande, Amat'>
    int inif = InitFlag[0];
    int inz = floor((ploc[2] - ${c['zmin']}) / ${c['dzr']});
    int iny = floor((ploc[1] - ${c['ymin']}) / ${c['dyr']});
    int inzs[2] = {inz, min(inz + 1, ${c['mnflim_ez']} - 1)};
    int inys[2] = {iny, min(iny + 1, ${c['mnflim_ey']} - 1)};

    fpdtype_t cy0 = (${c['ymin']} + ${c['dyr']} * (iny + 1) - ploc[1]) / (${c['dyr']}*${c['dzr']});
    fpdtype_t cy1 = ${c['dyr']} / (${c['dyr']}*${c['dzr']})  - cy0;
    fpdtype_t cz0 = ${c['zmin']} + ${c['dzr']} * (inz + 1) - ploc[2];
    fpdtype_t cz1 = ${c['dzr']} - cz0;

    if(inif == 0) {
        fpdtype_t ual_sp_loc[2][2][3] = {{{0.0}}};	     
        for (int ly = 0; ly < 2; ly++) {
        	for (int lz = 0; lz < 2; lz++) {
                 for (int k = 0; k < 3; k++) {
                         for (int i = 0; i < ${c['Nfy']}*2 + 1; i++) {	
                                 fpdtype_t tmp = 0.0;	 
                                 #pragma unroll
                                 for (int j = 0; j < ${c['Nfz']}*2 + 1; j++) {
         		                        tmp += bbz[j] * runi_prev[k][(inys[ly] + i) * ${c['mnflim_ez']} + inzs[lz] + j];
         	                     }
                                 ual_sp_loc[ly][lz][k] += bby[i]*tmp;
                         }
                 }
            }   	
        }   

        for (int k = 0; k < 3; k++) {
           uspr_prev[k] = (cy0 * ual_sp_loc[0][0][k] + cy1 * ual_sp_loc[1][0][k]) * cz0 + (cy0 * ual_sp_loc[0][1][k] + cy1 * ual_sp_loc[1][1][k]) * cz1;
        }

        fpdtype_t uspr_loc = Amat[0] * uspr_prev[0];
        fpdtype_t vspr_loc = Amat[1] * uspr_prev[0] + Amat[2] * uspr_prev[1];
        fpdtype_t wspr_loc = Amat[3] * uspr_prev[2];

	uspr_prev[0] = uspr_loc;
	uspr_prev[1] = vspr_loc;
	uspr_prev[2] = wspr_loc;
    }

    fpdtype_t ual_sp_loc[2][2][3] = {{{0.0}}};	     
    for (int ly = 0; ly < 2; ly++) {
    	for (int lz = 0; lz < 2; lz++) {
             for (int k = 0; k < 3; k++) {
                     for (int i = 0; i < ${c['Nfy']}*2 + 1; i++) {	
                             fpdtype_t tmp = 0.0;	 
                             #pragma unroll
                             for (int j = 0; j < ${c['Nfz']}*2 + 1; j++) {
     		                        tmp += bbz[j] * runi[k][(inys[ly] + i) * ${c['mnflim_ez']} + inzs[lz] + j];
     	                     }
                             ual_sp_loc[ly][lz][k] += bby[i]*tmp;
                     }
             }
        }   	
    }   

    for (int k = 0; k < 3; k++) {
       uspr[k] = (cy0 * ual_sp_loc[0][0][k] + cy1 * ual_sp_loc[1][0][k]) * cz0 + (cy0 * ual_sp_loc[0][1][k] + cy1 * ual_sp_loc[1][1][k]) * cz1;
    }

    fpdtype_t uspr_loc = Amat[0] * uspr[0];
    fpdtype_t vspr_loc = Amat[1] * uspr[0] + Amat[2] * uspr[1];
    fpdtype_t wspr_loc = Amat[3] * uspr[2];

    urande[0] = ${c['Coft'][0]} * uspr_prev[0] + ${c['Coft'][1]} * uspr_loc;
    urande[1] = ${c['Coft'][0]} * uspr_prev[1] + ${c['Coft'][1]} * vspr_loc;
    urande[2] = ${c['Coft'][0]} * uspr_prev[2] + ${c['Coft'][1]} * wspr_loc;

    fpdtype_t usim = urande[0] * ${c['vc'][0]} - urande[1] * ${c['vc'][1]};
    fpdtype_t vsim = urande[0] * ${c['vc'][1]} + urande[1] * ${c['vc'][0]};

    urand[0] = usim;
    urand[1] = vsim;
    urand[2] = urande[2];

    uspr_prev[0] = uspr_loc;
    uspr_prev[1] = vspr_loc;
    uspr_prev[2] = wspr_loc;

    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur','urand', 'urande')};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl' externs='ploc, t, bby, bbz, InitFlag, runi, runi_prev, uspr, uspr_prev, urand, urande, Amat'>
    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'urand', 'urande')};
    ${pyfr.expand('bc_ldg_grad_state', 'ul', 'nl', 'gradul', 'gradur')};

    fpdtype_t fvr[${ndims}][${nvars}] = {{0}};
    ${pyfr.expand('viscous_flux_add', 'ur', 'gradur', 'fvr')};
    ${pyfr.expand('artificial_viscosity_add', 'gradur', 'fvr', 'artviscl')};

    // Perform the Riemann solve
    fpdtype_t ficomm[${nvars}], fvcomm;
    ${pyfr.expand('rsolve', 'ul', 'ur', 'nl', 'ficomm')};

% for i in range(nvars):
    fvcomm = ${' + '.join('nl[{j}]*fvr[{j}][{i}]'.format(i=i, j=j)
                          for j in range(ndims))};
% if tau != 0.0:
    fvcomm += ${tau}*(ul[${i}] - ur[${i}]);
% endif

    ul[${i}] = magnl*(ficomm[${i}] + fvcomm);
% endfor
</%pyfr:macro>


