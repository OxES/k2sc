import scipy.linalg as sla
from .detrender import *


class SegmentedDetrender(Detrender):
    def __init__(self,  flux, inputs, splits=[], mask=None, p0=None, tr_nrandom=100, tr_bspan=50, tr_nblocks=5):
        super(SegmentedDetrender, self).__init__(flux, inputs, mask, p0, tr_nrandom, tr_bspan, tr_nblocks)
        self.splits = splits


    def _get_mask(self, t1, t2, splits=None):
        """Creates a covariance matrix mask based on a list of splits in time."""
        splits = self.splits if splits is None else splits
        mask = zeros((t1.size,t2.size), np.bool)
        o1,o2 = 0, 0
        for split in splits:
            e1 = (where(t1 < split)[0]).max() + 1
            e2 = (where(t2 < split)[0]).max() + 1
            mask[o1:e1,o2:e2] = 1
            o1,o2 = e1,e2
        mask[o1:,o2:] = 1
        return mask


    def compute_cmat(self, pv, X1, X2, separate=False, add_wn=False, splits=None):
        splits = self.splits if splits is None else splits
        if pv is not None:
            self.set_pv(pv)

        K1 = self._k1.value(X1, X2)
        K2 = self._k2.value(X1, X2)
        if splits:
            t1 = X1[:,0].flatten()
            t2 = X2[:,0].flatten()
            K2 *= self._get_mask(t1, t2, splits)

        if separate:
            return K1, K2
        else:
            if add_wn:
                return K1 + K2 + self._pv[-1]**2 * identity(K1.shape[0])
            else:
                return K1 + K2


    def negll(self, pv=None, flux=None, inputs=None, splits=None):
        flux = flux if flux is not None else self.data.masked_normalised_flux
        inputs = inputs if inputs is not None else self.data.masked_inputs
        K = self.compute_cmat(pv, inputs, inputs, splits=splits, add_wn=True)
        L = sla.cho_factor(K)
        b = sla.cho_solve(L, flux)
        return log(diag(L[0])).sum() + 0.5 * dot(flux,b)


    def compute_gp(self, pv=None, inputs=None):
        if pv is not None:
            self.set_pv(pv)


    def predict(self, pv, flux=None, inputs=None, inputs_pred=None, mean_only=True, splits=None):
        flux = flux if flux is not None else self.data.masked_flux
        iptr = inputs if inputs is not None else self.data.masked_inputs
        ippr = inputs_pred if inputs_pred is not None else iptr

        K0 = self.compute_cmat(pv, iptr, iptr, add_wn=False, splits=splits)
        K  = K0 + self._pv[-1]**2 * identity(K0.shape[0])
        if inputs_pred is None:
            Ks  = K0.copy()
            Kss = K.copy()
        else:
            Ks  = self.compute_cmat(pv, ippr, ippr, add_wn=False, splits=splits)
            Kss = self.compute_cmat(pv, ippr, ippr, add_wn=True, splits=splits)

        L = sla.cho_factor(K)
        b = sla.cho_solve(L, flux)
        mu = dot(Ks, b)

        if mean_only:
            return mu
        else:
            b = sla.cho_solve(L, Ks.T)
            cov = Kss - dot(Ks, b)
            err = np.sqrt(diag(cov))
            return mu, err
