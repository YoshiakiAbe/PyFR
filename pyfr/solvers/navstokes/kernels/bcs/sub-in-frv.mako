# -*- coding: utf-8 -*-
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.common'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.artvisc'/>
<%include file='pyfr.solvers.euler.kernels.rsolvers.${rsolver}'/>
<%include file='pyfr.solvers.navstokes.kernels.flux'/>

#include <stdio.h>

<% tau = c['ldg-tau'] %>

<%pyfr:macro name='bc_impose_state' params='ul, nl, ur, ual_sp'>

    ur[0] = ${c['rho']};
% for i, v in enumerate('uvw'[:ndims]):
    ur[${i + 1}] = (${c['rho']}) * (${c[v]} + ual_sp[${i}]);
% endfor
    ur[${nvars - 1}] = ul[${nvars - 1}]
                     - 0.5*(1.0/ul[0])*${pyfr.dot('ul[{i}]', i=(1, ndims + 1))}
                     + 0.5*(1.0/ur[0])*${pyfr.dot('ur[{i}]', i=(1, ndims + 1))};

</%pyfr:macro>


<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t, bby, bbz, runi, testual'>
    int inz = round((ploc[2] - ${c['zmin']}) / ${c['dzr']});
    int iny = round((ploc[1] - ${c['ymin']}) / ${c['dyr']});

    fpdtype_t ual_sp[2][3] = {{0.0}};

    for (int l = 0; l < 2; l++) {
            for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < ${c['Nfz']}*2 + 1; j++) {
                            fpdtype_t tmp = 0.0;
                            #pragma unroll
                            for (int i = 0; i < ${c['Nfy']}*2 + 1; i++) {
    		                        tmp += bby[i] * runi[3 * l + k][(inz + j) * ${c['MNfy']} + iny + i];
    	                    }
                            ual_sp[l][k] += bbz[j]*tmp;
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

    a11 = 0.1;
    a21 = 0.0;
    a22 = 0.1;
    a33 = 0.1;

    testual[0] = 0.0;
    testual[1] = 0.0;
    testual[2] = 0.0;
% for i, v in enumerate(c['Coft']):
            testual[0] += ${v} * (a11 * ual_sp[${i}][0]);
            testual[1] += ${v} * (a21 * ual_sp[${i}][0] + a22 * ual_sp[${i}][1]);
            testual[2] += ${v} * (a33 * ual_sp[${i}][2]);
% endfor

    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'testual')};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_zero'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl' externs='ploc, t, bby, bbz, runi, testual'>
    // Viscous states
    fpdtype_t ur[${nvars}], gradur[${ndims}][${nvars}];
    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'testual')};
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


