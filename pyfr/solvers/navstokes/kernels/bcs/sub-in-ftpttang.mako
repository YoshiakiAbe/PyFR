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


<%pyfr:macro name='bc_ldg_state' params='ul, nl, ur' externs='ploc, t, urand, ufpts, Amat'>
    fpdtype_t a11 = Amat[0];
    fpdtype_t a21 = Amat[1];
    fpdtype_t a22 = Amat[2];
    fpdtype_t a33 = Amat[3];

    urand[0] = 0.0;
    urand[1] = 0.0;
    urand[2] = 0.0;

% for i, v in enumerate(c['Coft']):
            urand[0] += ${v} * (a11 * ufpts[${i} * 3]);
            urand[1] += ${v} * (a21 * ufpts[${i} * 3] + a22 * ufpts[${i} * 3 + 1]);
            urand[2] += ${v} * (a33 * ufpts[${i} * 3 + 2]);
% endfor

    ${pyfr.expand('bc_impose_state', 'ul', 'nl', 'ur', 'urand')};
</%pyfr:macro>

<%pyfr:alias name='bc_ldg_grad_state' func='bc_common_grad_copy'/>

<%pyfr:macro name='bc_common_flux_state' params='ul, gradul, artviscl, nl, magnl' externs='ploc, t, urand'>
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


