# -*- coding: utf-8 -*-
import time
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
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        tplc = self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )
        self._tpl_c.update(tplc)


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

        lagt = 0.1667 # turbulent time scale
        self.drt = 0.0001 # time step size for random seed
        #dr = {'y':0.00075,'z':0.00075} # uni grid size of inlet plane for random seed
        dr = {'y':0.001,'z':0.001} # uni grid size of inlet plane for random seed
        #dr = {'y':0.005,'z':0.005} # uni grid size of inlet plane for random seed
        #dr = {'y':0.01,'z':0.01} # uni grid size of inlet plane for random seed

        L  = {'y':0.1,'z':0.1} # correlation length
        #L  = {'y':0.65,'z':0.65} # correlation length for fitting ratio between inlet size/stencil size
        cmin  = {'y':-0.587872982,'z':-1.14391935} # inlet plane min y / z
        cmax  = {'y':0.367873043,'z':1.14391935} # inlet plane max y / z

        xyzfpts_2d = np.moveaxis([elemap[etype].get_ploc_for_inter(eidx,fidx) for etype, eidx, fidx, flags in lhs], -1, 0)
        xyzfpts = xyzfpts_2d.reshape(3, -1)
        xyzfpts_mins = {s:min(t) for s,t in zip(['x','y','z'], xyzfpts)}
        xyzfpts_maxs = {s:max(t) for s,t in zip(['x','y','z'], xyzfpts)}

        MNf, mnflim, mflim = 1, 1, 1  # fixed constant
        self.Mf, self.Nf = {'y':0, 'z':0}, {'y':0, 'z':0}
        self.mflim_e = {'y':0, 'z':0}
        self.mfmin, self.mfmax = {'y':0, 'z':0}, {'y':0, 'z':0}
        self.mnflim_e = {'y':0, 'z':0}
        self.bb1d = {'y':0, 'z':0}

        for ind in ['y', 'z']: # y-z plane
            r1d = np.arange(cmin[ind], cmax[ind] + 0.5 * dr[ind], dr[ind])
            n = int(L[ind] / dr[ind])
            Nf = 2 * n # or >= 2
            Mf = int(np.round((cmax[ind] - cmin[ind]) / dr[ind])) + 1
            MNf = MNf * (Mf + 2 * Nf)
            mfmin = int(np.round((xyzfpts_mins[ind] - cmin[ind]) / dr[ind]))
            mfmax = int(np.round((xyzfpts_maxs[ind] - cmin[ind]) / dr[ind]))
            mnflim = mnflim * ((mfmax - mfmin + 1) + 2 * Nf)
            mflim = mflim * (mfmax - mfmin + 1)

            self.Nf[ind] = Nf
            self.Mf[ind] = Mf
            self.mfmin[ind] = mfmin
            self.mfmax[ind] = mfmax
            self.mnflim_e[ind] = (mfmax - mfmin + 1) + 2 * Nf
            self.mflim = mflim
            self.mflim_e[ind] = mfmax - mfmin + 1

            self._tpl_c['d' + ind + 'r'] = dr[ind] 
            self._tpl_c[ind + 'min_inlet'] = cmin[ind]
            self._tpl_c[ind + 'max_inlet'] = cmax[ind]
            self._tpl_c[ind + 'min'] = xyzfpts_mins[ind] 
            self._tpl_c['mflim_e' + ind] = self.mflim_e[ind] 

            bbtl = [np.exp(- np.pi * np.abs(k - Nf) / n) for k in np.arange(2 * Nf + 1)]
            bbtls = np.sum([s * s for s in bbtl], axis=0)
            self.bb1d[ind] = np.array([s / bbtls for s in bbtl])

        self.MNf = MNf
        self.mnflim = mnflim
        self.mflim = mflim

        print('mflim,mymax,mzmax=',self.mflim,self.mflim_e['y'],self.mflim_e['z'])

        # Modeling Reynolds stress 
        # uu, uw, ww, vv = R11, R21, R22, R33 
        aarey = [5, 5, 5, 5] # sharpness
        bbrey = [0.271, 2.084, 2.084, 0.833] # flat
        ccrey = [0.0533, -0.0069, 0.0069, 0.0173] # max
        fact = 100.0 # <= to keep std.=1.0 (from model.py)
        fact = fact * 0.5 # amplify for ghost value
        Reys = [] 
        for i,a in enumerate(aarey):
            tmp = []
            for j,s in enumerate(xyzfpts.swapaxes(0, 1)):
                z = s[2]
                uu = ccrey[i] * (z**4 + bbrey[i]) * (np.tanh((z - cmin['z']) * a)  + np.tanh((cmax['z'] - z) * a) - 1.0)
                tmp.append(uu)
            Reys.append(np.array(tmp)[self._perm])
        Reys = np.array(Reys)
        aml = []
        r11sq = np.sqrt(np.abs(Reys[0])) + 1e-10
        aml.append(r11sq)
        aml.append(Reys[1] / r11sq)
        aml.append(np.sqrt(np.abs(Reys[2] - (Reys[1] / r11sq) * (Reys[1] / r11sq))))
        aml.append(np.sqrt(np.abs(Reys[3])))
        Amat = self._be.matrix((4, self.ninterfpts))
        self._set_external('Amat', 'fpdtype_t[4]', value=Amat)
        Amat.set(np.array(aml)*fact)
        
        # Cast random velocity to fpts; index info between uniform random grid and fpts
        #                             [f00   f01][cz[0]]
        # [f[y] f[z]] = [cy[0], cy[1]][         ][     ]       
        #                             [f10   f11][cz[1]]
        xsw = xyzfpts.swapaxes(0, 1)
        self.fptcast = (np.empty(self.ninterfpts, dtype=np.int32), np.empty(self.ninterfpts, dtype=np.int32))
        self.qpck = [(np.empty(self.ninterfpts, dtype=np.int32), np.empty(self.ninterfpts, dtype=np.int32)) for i in range(4)]
        fptcast_tmp = np.empty((2, self.ninterfpts), dtype=np.int32)
        qpck_tmp = [np.empty((2, self.ninterfpts), dtype=np.int32) for i in range(4)]
        self.cy = np.empty((2, self.ninterfpts))
        self.cz = np.empty((2, self.ninterfpts))
        for i,s in enumerate(xsw):
            iny = int((s[1] - xyzfpts_mins['y']) // dr['y'])
            inz = int((s[2] - xyzfpts_mins['z']) // dr['z'])
            inyp = min(iny + 1, self.mfmax['y'] - 1)
            inzp = min(inz + 1, self.mfmax['z'] - 1)
            fptcast_tmp[0][i], fptcast_tmp[1][i] = iny, inz
            self.cy[0][i] = xyzfpts_mins['y'] + dr['y'] * (iny + 1) - s[1]
            self.cz[0][i] = xyzfpts_mins['z'] + dr['z'] * (inz + 1) - s[2]
            self.cy[1][i] = dr['y'] - self.cy[0][i]
            self.cz[1][i] = dr['z'] - self.cz[0][i]
            qpck_tmp[0][0][i], qpck_tmp[0][1][i], qpck_tmp[1][0][i], qpck_tmp[1][1][i] = iny, inz, iny, inzp
            qpck_tmp[2][0][i], qpck_tmp[2][1][i], qpck_tmp[3][0][i], qpck_tmp[3][1][i] = inyp, inz, inyp, inzp
        self.cy = self.cy / (dr['y'] * dr['z'])
        self.cy[0], self.cy[1] = self.cy[0][self._perm], self.cy[1][self._perm]
        self.cz[0], self.cz[1] = self.cz[0][self._perm], self.cz[1][self._perm]
        # Permute
        fptcast_tmp[0], fptcast_tmp[1] = fptcast_tmp[0][self._perm], fptcast_tmp[1][self._perm]
        for i in range(4):
            qpck_tmp[i][0], qpck_tmp[i][1] = qpck_tmp[i][0][self._perm], qpck_tmp[i][1][self._perm]
        for i in range(self.ninterfpts):
            self.fptcast[0][i], self.fptcast[1][i] = (fptcast_tmp[0][i], fptcast_tmp[1][i])
            for j in range(4):
                self.qpck[j][0][i], self.qpck[j][1][i] = (qpck_tmp[j][0][i], qpck_tmp[j][1][i])

        # Allocate urand
        self.urand = urand = self._be.matrix((3, self.ninterfpts))
        self._set_external('urand', 'inout fpdtype_t[3]', value=urand)

        # Allocate Coft for temporal filtering
        self._tpl_c['Coft'] = [np.exp(-0.5 * np.pi * self.drt / lagt), 
                               np.sqrt(1.0 - np.exp(-np.pi * self.drt / lagt))]

        # bb compute
        bbzpy, bbzpz = np.pad(self.bb1d['y'], (self.mfmax['y'] - self.mfmin['y'], 0), 'constant'),\
                       np.pad(self.bb1d['z'], (self.mfmax['z'] - self.mfmin['z'], 0), 'constant')
        bbzp2d = np.outer(bbzpy, bbzpz)
        self.bbzp2df = np.fft.rfft2(bbzp2d)

        # Allocate ufpts
        self.ufpts = ufpts = self._be.matrix((6, self.ninterfpts))
        self._set_external('ufpts', 'fpdtype_t[6]', value=ufpts)
        self.ufpts_prev = np.zeros((3, self.ninterfpts))
        MNfz = self.Mf['z'] + 2 * self.Nf['z']
        lymax = self.mfmax['y'] + 2 * self.Nf['y'] + 1
        lyseed = np.hstack((np.arange(0, self.Mf['y']), np.arange(0, 2 * self.Nf['y'] + 1)))
        lzmax = self.mfmax['z'] + 2 * self.Nf['z'] + 1
        runin2ds = np.empty((lymax - self.mfmin['y'], lzmax - self.mfmin['z']))
        t = 0.0 # initial time
        senum = int(np.round(t / self.drt))
        for j in range(3):
            for i, ly in enumerate(range(self.mfmin['y'], lymax)):
                np.random.seed((senum, j, lyseed[ly]))
                tmp = np.random.uniform(-0.5, 0.5, MNfz)
                runin2ds[i, :] = tmp[self.mfmin['z'] : lzmax]
            runin2df = np.fft.rfft2(runin2ds)
            runin2df *= self.bbzp2df
            h2d_lim = np.fft.irfft2(runin2df)

            # Bilinear interpolation
            fmat = np.array([[h2d_lim[self.qpck[0]], h2d_lim[self.qpck[1]]],\
                             [h2d_lim[self.qpck[2]], h2d_lim[self.qpck[3]]]])
            self.ufpts_prev[j, :] = np.array([np.dot(np.dot(self.cy[:, i], fmat[:, :, i]), self.cz[:, i])\
                                     for i in range(self.ninterfpts)])

            # Round off
            #self.ufpts_prev[j, :] = h2d_lim[self.fptcast] 

    def prepare(self, t):
        tst = time.time()
        self.ta = ta = [0.0, 0.0, 0.0, 0.0, 0.0]

        senum = int(np.round(t / self.drt)) + 1
        MNfz = self.Mf['z'] + 2 * self.Nf['z']
        lymax = self.mfmax['y'] + 2 * self.Nf['y'] + 1
        lyseed = np.hstack((np.arange(0, self.Mf['y']), np.arange(0, 2 * self.Nf['y'] + 1)))
        lzmax = self.mfmax['z'] + 2 * self.Nf['z'] + 1

        runin2ds = np.empty((lymax - self.mfmin['y'], lzmax - self.mfmin['z']))
        ufpts = np.empty((3, self.ninterfpts))

        for j in range(3):
            tst_rand = time.time()
            for i, ly in enumerate(range(self.mfmin['y'], lymax)):
                np.random.seed((senum, j, lyseed[ly]))
                tmp = np.random.uniform(-0.5, 0.5, MNfz)
                runin2ds[i, :] = tmp[self.mfmin['z'] : lzmax]
            trand = time.time() - tst_rand

            tst_fft = time.time()
            runin2df = np.fft.rfft2(runin2ds)
            tfft = time.time() - tst_fft
            runin2df *= self.bbzp2df
            tmul = time.time() - tfft - tst_fft
            h2d_lim = np.fft.irfft2(runin2df)
            tifft = time.time() - tmul - tfft - tst_fft

            # Bilinear interpolation
            fmat = np.array([[h2d_lim[self.qpck[0]], h2d_lim[self.qpck[1]]],\
                             [h2d_lim[self.qpck[2]], h2d_lim[self.qpck[3]]]])
            ufpts[j, :] = np.array([np.dot(np.dot(self.cy[:, i], fmat[:, :, i]), self.cz[:, i])\
                                  for i in range(self.ninterfpts)])

            # Round off
            #ufpts[j, :] = h2d_lim[self.fptcast]
            tcast = time.time() - tifft - tmul - tfft - tst_fft
            ta[0]+=tfft
            ta[1]+=tifft
            ta[2]+=tcast
            ta[3]+=tmul
            ta[4]+=trand

        t1 = time.time() - tst
        self.ufpts.set(np.vstack((self.ufpts_prev, ufpts)))
        t2 = time.time() - t1 - tst
        self.ufpts_prev = ufpts        
        t4 = time.time() - tst
        print('total=',np.round(t4,5),'| rand=',np.round(self.ta[4],5), ' kernelset=',np.round(t2,5), \
          'fft/ifft=',np.round(t1,5), '(fft=', np.round(self.ta[0],5), ' ifft=',np.round(self.ta[1],5),\
          ' cast=',np.round(self.ta[2],5), ' mul=',np.round(self.ta[3],5),')')




class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg):
        super().__init__(be, lhs, elemap, cfgsect, cfg)

        self._tpl_c.update(self._exp_opts(['p'], lhs))
