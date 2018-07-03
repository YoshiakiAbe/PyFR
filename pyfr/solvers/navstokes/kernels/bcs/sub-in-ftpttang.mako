# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

#include <stdio.h>

<% tau = c['ldg-tau'] %>

<%pyfr:macro name='bc_impose_state' params='ul, nl, ur, urand'>
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


<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t, bby, bbz, runi, urand'>
    int inz = round((ploc[2] - ${c['zmin']}) / ${c['dzr']});
    int iny = round((ploc[1] - ${c['ymin']}) / ${c['dyr']});

    fpdtype_t ual_sp[2][3] = {{0.0}};

    for (int l = 0; l < 2; l++) {
            for (int k = 0; k < 3; k++) {
                    for (int i = 0; i < ${c['Nfy']}*2 + 1; i++) {	
                            fpdtype_t tmp = 0.0;
                            #pragma unroll
                            for (int j = 0; j < ${c['Nfz']}*2 + 1; j++) {
    		                        tmp += bbz[j] * runi[3 * l + k][(iny + i) * ${c['mnflim_ez']} + inz + j];
    	                    }
                            ual_sp[l][k] += bby[i]*tmp;
                    }
            }
    }

    fpdtype_t R11 = 0.05;
    fpdtype_t R21 = 0.05;
    fpdtype_t R22 = 0.05;
    fpdtype_t R33 = 0.05;

    fpdtype_t a11 = sqrt(R11);
    fpdtype_t a21 = R21 / sqrt(R11);
    fpdtype_t a22 = sqrt(R22 - (R21 / sqrt(R11)) * (R21 / sqrt(R11)));
    fpdtype_t a33 = sqrt(R33);

    a11 = 5.0;
    a21 = 0.0;
    a22 = 5.0;
    a33 = 5.0;

    urand[0] = 0.0;
    urand[1] = 0.0;
    urand[2] = 0.0;
% for i, v in enumerate(c['Coft']):
            urand[0] += ${v} * (a11 * ual_sp[${i}][0]);
            urand[1] += ${v} * (a21 * ual_sp[${i}][0] + a22 * ual_sp[${i}][1]);
            urand[2] += ${v} * (a33 * ual_sp[${i}][2]);
% endfor

    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'urand')};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl' externs='ploc, t, bby, bbz, runi, urand'>
    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'urand')};
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


