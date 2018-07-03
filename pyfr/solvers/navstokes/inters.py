# -*- coding: utf-8 -*-

import numpy as np

from pyfr.backends.base.kernels import ComputeMetaKernel
from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)


class NavierStokesIntInters(BaseAdvectionDiffusionIntInters):
    def __init__(self, be, lhs, rhs, elemap, cfg):
        super().__init__(be, lhs, rhs, elemap, cfg)

        # Pointwise template arguments
        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction')
        shock_capturing = self.cfg.get('solver', 'shock-capturing')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.intconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        if abs(self._tpl_c['ldg-beta']) == 0.5:
            self.kernels['copy_fpts'] = lambda: ComputeMetaKernel(
                [ele.kernels['_copy_fpts']() for ele in elemap.values()]
            )

        self.kernels['con_u'] = lambda: be.kernel(
            'intconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs,
            ulout=self._vect_lhs, urout=self._vect_rhs
        )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'intcflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesMPIInters(BaseAdvectionDiffusionMPIInters):
    def __init__(self, be, lhs, rhsrank, rallocs, elemap, cfg):
        super().__init__(be, lhs, rhsrank, rallocs, elemap, cfg)

        # Pointwise template arguments
        rsolver = cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = cfg.get('solver', 'viscosity-correction')
        shock_capturing = cfg.get('solver', 'shock-capturing')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.mpiconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        self.kernels['con_u'] = lambda: be.kernel(
            'mpiconu', tplargs=tplargs, dims=[self.ninterfpts],
            ulin=self._scal_lhs, urin=self._scal_rhs, ulout=self._vect_lhs
        )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'mpicflux', tplargs=tplargs, dims=[self.ninterfpts],
            ul=self._scal_lhs, ur=self._scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artviscl=self._artvisc_lhs, artviscr=self._artvisc_rhs,
            magnl=self._mag_pnorm_lhs, nl=self._norm_pnorm_lhs
        )


class NavierStokesBaseBCInters(BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        # Pointwise template arguments
        rsolver = cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = cfg.get('solver', 'shock-capturing', 'none')
        tplargs = dict(ndims=self.ndims, nvars=self.nvars, rsolver=rsolver,
                       visc_corr=visc_corr, shock_capturing=shock_capturing,
                       c=self._tpl_c, bctype=self.type,
                       bccfluxstate=self.cflux_state)

        be.pointwise.register('pyfr.solvers.navstokes.kernels.bcconu')
        be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')


        self.kernels['con_u'] = lambda: be.kernel(
            'bcconu', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ulin=self._scal_lhs,
            ulout=self._vect_lhs, nlin=self._norm_pnorm_lhs,
            **self._external_vals
         )
        self.kernels['comm_flux'] = lambda: be.kernel(
            'bccflux', tplargs=tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self._scal_lhs,
            gradul=self._vect_lhs, magnl=self._mag_pnorm_lhs,
            nl=self._norm_pnorm_lhs, artviscl=self._artvisc_lhs,
            **self._external_vals
         )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._tpl_c['cpTw'], = self._eval_opts(['cpTw'])
        self._tpl_c['v'] = self._eval_opts('uvw'[:self.ndims], default='0')


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )
        self._tpl_c.update(tplc)


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )
        self._tpl_c.update(tplc)


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )
        self._tpl_c.update(tplc)

        lagt = 0.1 # turbulent time scale
        self.drt = 0.001 # time step size for random seed
        #dr = {'y':0.005,'z':0.005} # uni grid size of inlet plane for random seed
        #L  = {'y':0.7,'z':0.7} # inlet plane size
        #cmin  = {'y':0.0,'z':0.0} # inlet plane min y / z
        #cmax  = {'y':2.0,'z':4.2} # inlet plane max y / z
        dr = {'y':0.005,'z':0.005} # uni grid size of inlet plane for random seed
        L  = {'y':0.7,'z':0.7} # inlet plane size
        cmin  = {'y':-0.587872982,'z':-1.14391935} # inlet plane min y / z
        cmax  = {'y':0.367873043,'z':1.14391935} # inlet plane max y / z

        MNf = 1
        for ind in ['y','z']: # y-z plane
            r1d = np.arange(cmin[ind], cmax[ind] + 0.5 * dr[ind], dr[ind])
            n = int(L[ind] / dr[ind])
            Nf = 2 * n # or >= 2
            Mf = int(np.round((cmax[ind] - cmin[ind]) / dr[ind])) + 1
            MNf = MNf * (Mf + 2 * Nf)

            self._tpl_c['Nf' + ind] = Nf
            self._tpl_c['Mf' + ind] = Mf
            self._tpl_c['d' + ind + 'r'] = dr[ind]
            self._tpl_c[ind + 'min'] = cmin[ind]
            self._tpl_c['MNf' + ind] = Nf * 2 + Mf
            bb = self._be.matrix((1, 2 * Nf + 1))
            self._set_external('bb' + ind, 'in broadcast fpdtype_t[{0}]'.format(2 * Nf + 1), value=bb)
    
            bbtl = [np.exp(- np.pi * np.abs(k - Nf) / n) for k in np.arange(2 * Nf + 1)]
            bbtls = np.sum([s * s for s in bbtl], axis=0)
            self.bbtmp = bbtmp = np.array([[s / bbtls for s in bbtl]])
            bb.set(bbtmp)        
            
        self.MNf = MNf
        self._tpl_c['MNf'] = MNf
        self.runi = runi = self._be.matrix((6, MNf))
        self._set_external('runi', 'in broadcast fpdtype_t[{0}][{1}]'.format(6, MNf), value=runi)

        urand = self._be.matrix((3, self.ninterfpts))
        self._set_external('urand', 'inout fpdtype_t[3]', value=urand)

        self._tpl_c['Coft'] = [np.exp(-0.5 * np.pi * self.drt / lagt), 
                               np.sqrt(1.0 - np.exp(-np.pi * self.drt / lagt))]


    def prepare(self, t):

        senum = int(np.round(t / self.drt)) + 1 # "+1" is to avoid 0 at t = 0
        np.random.seed(senum - 1)
        runin0 = np.random.uniform(-0.5, 0.5, (3, self.MNf))
        np.random.seed(senum)
        runin1 = np.random.uniform(-0.5, 0.5, (3, self.MNf))
        self.runi.set(np.vstack((runin0, runin1)))

        #senum = int(np.round(t / self.drt)) + 1 # "+1" is to avoid 0 at t = 0
        #np.random.seed(senum - 1) # need the previous t in subit...
        #runin0 = np.array([np.random.uniform(0., 1., self.MNf) - 0.5] * 3)
        #np.random.seed(senum)
        #runin1 = np.array([np.random.uniform(0., 1., self.MNf) - 0.5] * 3)
        #self.runi.set(np.vstack((runin0, runin1)))


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = None

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self._tpl_c['cpTt'], = self._eval_opts(['cpTt'])
        self._tpl_c.update(self._exp_opts(['pt'], lhs))
        self._tpl_c['Rdcp'] = (gamma - 1.0) / gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0] * np.pi / 180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self._tpl_c['vc'] = velcomps[:self.ndims]

        lagt = 0.01 # turbulent time scale
        self.drt = 0.0001 # time step size for random seed
        dr = {'y':0.01, 'z':0.01} # uni grid size of inlet plane for random seed
        L  = {'y':0.7, 'z':0.7} # inlet plane size
        cmin  = {'y':0.0, 'z':0.0} # inlet plane min y / z
        cmax  = {'y':2.0, 'z':4.2} # inlet plane max y / z

        # xyzfpts = (xyz, nface, fpts) => (xyz, 1d)
        xyzfpts = np.array([elemap[etype].get_ploc_for_inter(eidx,fidx) 
                            for etype, eidx, fidx, flags in lhs]).swapaxes(0,2).reshape(3, -1)
        xyzfpts_mins = {s:min(t) for s,t in zip(['x','y','z'], xyzfpts)}
        xyzfpts_maxs = {s:max(t) for s,t in zip(['x','y','z'], xyzfpts)}
        print(xyzfpts_mins['y'],xyzfpts_maxs['y'],xyzfpts_mins['z'],xyzfpts_maxs['z'])
        #quit()
        MNf, mnflim = 1, 1  # fixed constant
        self.Mf, self.Nf = {'y':0, 'z':0}, {'y':0, 'z':0}
        self.mfmin, self.mfmax = {'y':0, 'z':0}, {'y':0, 'z':0}
        self.mnflim_e = {'y':0, 'z':0}
        for ind in ['y', 'z']: # y-z plane
            r1d = np.arange(cmin[ind], cmax[ind] + 0.5 * dr[ind], dr[ind])
            n = int(L[ind] / dr[ind])
            Nf = 2 * n # or >= 2
            Mf = int(np.round((cmax[ind] - cmin[ind]) / dr[ind])) + 1
            MNf = MNf * (Mf + 2 * Nf)
            mfmin = int(np.round((xyzfpts_mins[ind] - cmin[ind]) / dr[ind]))
            mfmax = int(np.round((xyzfpts_maxs[ind] - cmin[ind]) / dr[ind]))
            mnflim = mnflim * ((mfmax - mfmin + 1) + 2 * Nf)

            self.Nf[ind] = self._tpl_c['Nf' + ind] = Nf
            self.Mf[ind] = self._tpl_c['Mf' + ind] = Mf
            self.mfmin[ind] = mfmin
            self.mfmax[ind] = mfmax
            self.mnflim_e[ind] = (mfmax - mfmin + 1) + 2 * Nf

            self._tpl_c['d' + ind + 'r'] = dr[ind]
            self._tpl_c[ind + 'min'] = xyzfpts_mins[ind] #cmin[ind]
            self._tpl_c['MNf' + ind] = Mf + 2 * Nf
            self._tpl_c['mnflim_e' + ind] = self.mnflim_e[ind]
            bb = self._be.matrix((1, 2 * Nf + 1))
            self._set_external('bb' + ind, 'in broadcast fpdtype_t[{0}]'.format(2 * Nf + 1), value=bb)
    
            bbtl = [np.exp(- np.pi * np.abs(k - Nf) / n) for k in np.arange(2 * Nf + 1)]
            bbtls = np.sum([s * s for s in bbtl], axis=0)
            self.bbtmp = bbtmp = np.array([[s / bbtls for s in bbtl]])
            bb.set(bbtmp)        
            
        self.MNf = self._tpl_c['MNf'] = MNf
        self.mnflim = mnflim
        self.runi = runi = self._be.matrix((6, self.mnflim))
        self._set_external('runi', 'in broadcast fpdtype_t[{0}][{1}]'.format(6, self.mnflim), value=runi)

        urand = self._be.matrix((3, self.ninterfpts))
        self._set_external('urand', 'inout fpdtype_t[3]', value=urand)

        self._tpl_c['Coft'] = [np.exp(-0.5 * np.pi * self.drt / lagt), 
                               np.sqrt(1.0 - np.exp(-np.pi * self.drt / lagt))]

    def prepare(self, t):

        #senum = int(np.round(t / self.drt)) + 1 # "+1" is to avoid 0 at t = 0
        #np.random.seed(senum - 1)
        #runin0 = np.random.uniform(-0.5, 0.5, (3, self.MNf))
        #np.random.seed(senum)
        #runin1 = np.random.uniform(-0.5, 0.5, (3, self.MNf))
        #print(runin0.shape)
        #self.runi.set(np.vstack((runin0, runin1)))
        
        MNfy, MNfz = self.Mf['y'] + 2 * self.Nf['y'], self.Mf['z'] + 2 * self.Nf['z']
        runins = []
        senum = int(np.round(t / self.drt)) + 1 # "+1" is to avoid 0 at t = 0
        for i, sn in enumerate([senum - 1, senum]):
            runin = np.zeros((self.mnflim, 3))
            for ly in range(self.mfmin['y'], self.mfmax['y'] + 2 * self.Nf['y'] + 1):
                np.random.seed((sn, ly))
                tmp = np.random.uniform(-0.5, 0.5, (MNfz, 3))
                ls = (ly - self.mfmin['y']) * self.mnflim_e['z'] + 0
                le = (ly - self.mfmin['y']) * self.mnflim_e['z'] + self.mnflim_e['z'] - 1
                runin[ls : le + 1] = tmp[self.mfmin['z'] : self.mfmax['z'] + 2 * self.Nf['z'] + 1]
            runins.append(runin.swapaxes(0, 1))
        self.runi.set(np.vstack((runins[0], runins[1])))


        #MNfy, MNfz = self.Mf['y'] + 2 * self.Nf['y'], self.Mf['z'] + 2 * self.Nf['z']
        #runin0, runin1 = np.zeros((self.MNf, 3)), np.zeros((self.MNf, 3))
        #for ly in range(0, MNfy):
        #    ls = ly * MNfz + 0
        #    le = ly * MNfz + MNfz - 1
        #    np.random.seed((senum - 1, ly))
        #    tmp = np.random.uniform(-0.5, 0.5, (MNfz, 3))
        #    runin0[ls : le + 1] = tmp[0 : MNfz]
        #    np.random.seed((senum, ly))
        #    tmp = np.random.uniform(-0.5, 0.5, (MNfz, 3))
        #    runin1[ls : le + 1] = tmp[0 : MNfz]
        #runin0 = runin0.swapaxes(0, 1)
        #runin1 = runin1.swapaxes(0, 1)
        #self.runi.set(np.vstack((runin0, runin1)))


        #MNfy, MNfz = self.Mf['y'] + 2 * self.Nf['y'], self.Mf['z'] + 2 * self.Nf['z']
        #runin0, runin1 = np.zeros((self.MNf, 3)), np.zeros((self.MNf, 3))
        #for ly in range(0, MNfy):
        #    for lz in range(0, MNfz):
        #        l = lz * MNfy + ly 
        #        np.random.seed((senum - 1, ly, lz))
        #        runin0[l] = np.random.uniform(-0.5, 0.5, 3)
        #        np.random.seed((senum, ly, lz))
        #        runin1[l] = np.random.uniform(-0.5, 0.5, 3)
        #runin0 = runin0.swapaxes(0, 1)
        #runin1 = runin1.swapaxes(0, 1)
        #self.runi.set(np.vstack((runin0, runin1)))

        #runin0 = [[0.0]*(self.Mf['y'] + 2 * self.Nf['y'])*(self.Mf['z'] + 2 * self.Nf['z'])]*3
        #for i in range(3):
        #    for ly in range(0, self.Mf['y']):
        #        for lz in range(0, self.Mf['z']):
        #            np.random.seed((senum - 1, ly, lz, i)) # need the previous t in subit...
        #            runin0[i][lz * (self.Mf['y'] + 2 * self.Nf['y']) + ly + 1] = np.random.uniform(0., 1., 1) - 0.5
        #runin1 = [[0.0]*(self.Mf['y'] + 2 * self.Nf['y'])*(self.Mf['z'] + 2 * self.Nf['z'])]*3
        #for i in range(3):
        #    for ly in range(0, self.Mf['y']):
        #        for lz in range(0, self.Mf['z']):
        #            np.random.seed((senum, ly, lz, i)) # need the previous t in subit...
        #            runin1[i][lz * (self.Mf['y'] + 2 * self.Nf['y']) + ly + 1] = np.random.uniform(0., 1., 1) - 0.5
        #self.runi.set(np.vstack((runin0, runin1)))

        #np.random.seed(senum-1)
        #runin0 = np.array([np.random.uniform(0., 1., self.MNf) - 0.5] * 3)
        #np.random.seed(senum)
        #runin1 = np.array([np.random.uniform(0., 1., self.MNf) - 0.5] * 3)
        #self.runi.set(np.vstack((runin0, runin1)))


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c.update(self._exp_opts(['p'], lhs))
