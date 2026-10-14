"""Independent covariance implementation of the harmonic-chain protocol.

hbar=1; V_ij=<{Delta R_i,Delta R_j}>/2; vacuum V=I/2.
All arrays use interleaved ordering (x_0,p_0,x_1,p_1,...).
Resource modes: L_0,...,L_(N-1),R_0,...,R_(N-1).
After insertion: observer,L_0,...,L_(N-1),R_0,...,R_(N-1).
The input replaces and discards L_insert; the observer is never evolved.
"""
from dataclasses import dataclass, asdict
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import cholesky, eigh, eigvalsh, expm
from scipy.signal import find_peaks
from scipy.special import xlogy


@dataclass(frozen=True)
class Parameters:
    n: int = 10
    mass_squared: float = 13.0
    spring: float = 5.0
    beta: float = 1.0
    eta: float = 1.0
    probe_r: float = 1.0
    probe_frequency: float = 1.0
    insert_site: int = 0

    def __post_init__(self):
        if self.n < 3:
            raise ValueError('Use N>=3 to avoid the two-site periodic-edge ambiguity.')
        if self.mass_squared <= 0 or self.spring < 0 or self.beta <= 0:
            raise ValueError('Require m^2>0, k>=0, beta>0.')
        if self.probe_r <= 0 or self.probe_frequency <= 0:
            raise ValueError('Require probe squeezing and frequency >0.')
        if not 0 <= self.insert_site < self.n:
            raise ValueError('Insertion site is outside the left ring.')


def omega(n):
    return np.kron(np.eye(n), np.array([[0., 1.], [-1., 0.]]))


def indices(modes):
    modes = np.asarray(list(modes), dtype=int)
    return np.column_stack((2*modes, 2*modes+1)).ravel()


def reduced(v, modes):
    q = indices(modes)
    return v[np.ix_(q, q)]


def symplectic_eigenvalues(v, tolerance=1e-8):
    """Hermitian spectrum of i L^T Omega L, where V=L L^T.

    Avoid taking arbitrary absolute eigenvalues of the nonsymmetric i Omega V.
    Tiny roundoff below 1/2 is clipped; a larger violation raises an error.
    """
    if not np.allclose(v, v.T, rtol=0, atol=1e-9):
        raise ValueError('Nonsymmetric covariance matrix.')
    n = v.shape[0]//2
    factor = cholesky((v+v.T)/2, lower=True)
    nu = eigvalsh(1j*(factor.T @ omega(n) @ factor))[n:]
    if nu.min() < .5-tolerance:
        raise ValueError(f'Unphysical covariance: minimum symplectic eigenvalue {nu.min()}')
    return np.maximum(nu, .5)


def entropy(v):
    """Von Neumann entropy in nats, with 0 log 0 handled continuously."""
    occupation = symplectic_eigenvalues(v)-.5
    return float(np.sum(xlogy(occupation+1, occupation+1)-xlogy(occupation, occupation)))


def mutual_information(v, a, b):
    a, b = list(a), list(b)
    if set(a) & set(b):
        raise ValueError('Mutual-information subsystems must be disjoint.')
    return entropy(reduced(v, a))+entropy(reduced(v, b))-entropy(reduced(v, a+b))


def congruence(s, v):
    result = s @ v @ s.T
    return (result+result.T)/2


def embed(s, modes, total_modes):
    result = np.eye(2*total_modes)
    q = indices(modes)
    result[np.ix_(q, q)] = s
    return result


def position_momentum_matrix(f, g):
    """Hamiltonian matrix for H=(x^T F x+p^T G p)/2, interleaved."""
    result = np.zeros((2*len(f), 2*len(f)))
    result[0::2, 0::2] = f
    result[1::2, 1::2] = g
    return result


class HarmonicProtocol:
    def __init__(self, parameters=Parameters()):
        self.p = parameters
        n = parameters.n
        # Build the ring Laplacian by summing bond energies, once per bond.
        laplacian = np.zeros((n, n))
        for j in range(n):
            bond = np.zeros(n)
            bond[j], bond[(j+1) % n] = 1., -1.
            laplacian += np.outer(bond, bond)
        self.f = parameters.mass_squared*np.eye(n)+parameters.spring*laplacian
        values, self.o = eigh(self.f)
        self.frequencies = np.sqrt(values)
        self.h_chain = position_momentum_matrix(self.f, np.eye(n))
        self.w0 = np.sqrt(parameters.mass_squared+2*parameters.spring)
        self.resource_left = list(range(n))
        self.resource_right = list(range(n, 2*n))
        self.left = list(range(1, n+1))
        self.right = list(range(n+1, 2*n+1))
        self.observer = [0]
        self.all_system = self.left+self.right
        self.h_coupling = self._coupling_hamiltonian()
        self.tfd = self._tfd()
        self.product = self.tfd.copy()
        l, r = indices(self.resource_left), indices(self.resource_right)
        self.product[np.ix_(l, r)] = 0.
        self.product[np.ix_(r, l)] = 0.
        self.resources = {'tfd': self.tfd, 'product': self.product}
        self.probe = self._probe()
        self.normalization = 2*entropy(self.probe[:2, :2])

    def _tfd(self):
        n, w, o, beta = self.p.n, self.frequencies, self.o, self.p.beta
        nu = .5/np.tanh(beta*w/2)
        # Stable csch expression, including low temperatures.
        z = np.exp(-beta*w/2)
        alpha = z/(1-z*z)
        dx = (o*(nu/w)) @ o.T
        dp = (o*(nu*w)) @ o.T
        cx = (o*(alpha/w)) @ o.T
        cp = (o*(alpha*w)) @ o.T
        v = np.zeros((4*n, 4*n))
        xl, pl = 2*np.arange(n), 2*np.arange(n)+1
        xr, pr = 2*np.arange(n, 2*n), 2*np.arange(n, 2*n)+1
        for rows, cols, block in [(xl,xl,dx),(xr,xr,dx),(pl,pl,dp),(pr,pr,dp),
                                  (xl,xr,cx),(xr,xl,cx),(pl,pr,-cp),(pr,pl,-cp)]:
            v[np.ix_(rows,cols)] = block
        return v

    def _probe(self):
        r = self.p.probe_r
        c, s = np.cosh(2*r)/2, np.sinh(2*r)/2
        v = np.array([[c,0,s,0],[0,c,0,-s],[s,0,c,0],[0,-s,0,c]])
        w = self.p.probe_frequency
        scale = np.diag([1/np.sqrt(w),np.sqrt(w)]*2)
        return congruence(scale, v)

    def _coupling_hamiltonian(self):
        """H_c = H_L+H_R + eta/4 sum_(j!=insert)
        [w0 x_Lj x_Rj + p_Lj p_Rj/w0]. Local terms are physical H,
        whereas the isolated backwards/forwards steps use beta H.
        """
        n = self.p.n
        h = np.zeros((4*n,4*n))
        h[:2*n,:2*n] = self.h_chain
        h[2*n:,2*n:] = self.h_chain
        for j in range(n):
            if j == self.p.insert_site:
                continue
            for q, coefficient in [(0,self.p.eta*self.w0/4),
                                    (1,self.p.eta/(4*self.w0))]:
                a, b = 2*j+q, 2*(n+j)+q
                h[a,b] = h[b,a] = coefficient
        return h

    def chain_evolution(self, t):
        """Analytic normal-mode propagator for modular evolution beta H."""
        w, o = self.frequencies, self.o
        c, s = np.cos(self.p.beta*w*t), np.sin(self.p.beta*w*t)
        result = np.zeros_like(self.h_chain)
        result[0::2,0::2] = result[1::2,1::2] = (o*c) @ o.T
        result[0::2,1::2] = (o*(s/w)) @ o.T
        result[1::2,0::2] = -(o*(s*w)) @ o.T
        return result

    def coupling_evolution(self, t):
        return expm(omega(2*self.p.n) @ self.h_coupling*t)

    def insert_probe(self, resource):
        """Trace the old insertion mode, tensor the probe, then relabel input."""
        total = 2*self.p.n+1
        v = np.zeros((2*total, 2*total))
        v[2:,2:] = resource
        target = 1+self.p.insert_site
        q = indices([target])
        v[q,:] = 0.
        v[:,q] = 0.
        probe_q = indices([0,target])
        v[np.ix_(probe_q,probe_q)] = self.probe
        return v

    def before_coupling(self, kind, evolution_time):
        n = self.p.n
        backward = embed(self.chain_evolution(-evolution_time), self.resource_left, 2*n)
        resource = congruence(backward, self.resources[kind])
        inserted = self.insert_probe(resource)
        forward = embed(self.chain_evolution(evolution_time), self.left, 2*n+1)
        return congruence(forward, inserted)

    def stages(self, kind, evolution_time, coupling_time):
        n = self.p.n
        backward = embed(self.chain_evolution(-evolution_time), self.resource_left, 2*n)
        after_backward = congruence(backward, self.resources[kind])
        inserted = self.insert_probe(after_backward)
        left_forward = congruence(embed(self.chain_evolution(evolution_time), self.left, 2*n+1), inserted)
        coupled = congruence(embed(self.coupling_evolution(coupling_time), self.all_system, 2*n+1), left_forward)
        final = congruence(embed(self.chain_evolution(evolution_time), self.right, 2*n+1), coupled)
        return {'inserted':inserted,'left_forward':left_forward,'coupled':coupled,'right_forward':final}

    def diagnostics(self, v):
        return {
            'I_obs_L': mutual_information(v, self.observer, self.left),
            'I_obs_R': mutual_information(v, self.observer, self.right),
            'I_obs_LR': mutual_information(v, self.observer, self.all_system),
        }

    def site_mutual_information(self, v):
        """I(observer:site) in nats, shape (2,N) with rows L and R.

        Each entry is computed from a two-mode reduced state. These entries
        are not additive contributions to the corresponding whole-side MI.
        """
        observer_entropy = entropy(reduced(v, self.observer))
        return np.array([[observer_entropy+entropy(reduced(v,[mode]))
                          -entropy(reduced(v,self.observer+[mode])) for mode in side]
                         for side in [self.left,self.right]])

    def resource_diagnostics(self, v):
        return {'S_L':entropy(reduced(v,self.resource_left)),
                'S_R':entropy(reduced(v,self.resource_right)),
                'S_LR':entropy(v),
                'I_L_R':mutual_information(v,self.resource_left,self.resource_right),
                'energy_L':float(np.trace(self.h_chain @ reduced(v,self.resource_left))/2),
                'energy_R':float(np.trace(self.h_chain @ reduced(v,self.resource_right))/2)}


def scan(protocol, values, varying, fixed_time):
    """Both resources always use exactly the same unitaries and times."""
    n = protocol.p.n
    before = ({kind:protocol.before_coupling(kind,fixed_time) for kind in protocol.resources}
              if varying=='coupling_time' else None)
    fixed_coupling = (embed(protocol.coupling_evolution(fixed_time),protocol.all_system,2*n+1)
                      if varying=='evolution_time' else None)
    rows = []
    for value in values:
        te, tc = ((fixed_time,value) if varying=='coupling_time' else (value,fixed_time))
        coupling = (embed(protocol.coupling_evolution(tc),protocol.all_system,2*n+1)
                    if fixed_coupling is None else fixed_coupling)
        right = embed(protocol.chain_evolution(te),protocol.right,2*n+1)
        for kind in protocol.resources:
            prepared = before[kind] if before is not None else protocol.before_coupling(kind,te)
            final = congruence(right,congruence(coupling,prepared))
            rows.append({'state':kind,'evolution_time':float(te),'coupling_time':float(tc),
                         **protocol.diagnostics(final)})
    return rows


def save_csv(path, rows):
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out, protocol, coupling_rows, evolution_rows, te, tc):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaussian-teleport-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(12,8),sharex='col',layout='constrained')
    colors = {'tfd':'#1966ab','product':'#d45a1b'}
    labels = {'tfd':'TFD','product':r'$\rho_L\otimes\rho_R$'}
    for col,(rows,xkey,title) in enumerate([
        (coupling_rows,'coupling_time',f'Coupling scan; evolution time = {te:g}'),
        (evolution_rows,'evolution_time',f'Evolution scan; coupling time = {tc:g}')]):
        for row,key in enumerate(['I_obs_L','I_obs_R']):
            ax = axes[row,col]
            for kind in protocol.resources:
                subset = [r for r in rows if r['state']==kind]
                ax.plot([r[xkey] for r in subset],[r[key] for r in subset],
                        color=colors[kind],ls='-' if kind=='tfd' else '--',lw=2,label=labels[kind])
            ax.axhline(protocol.normalization,color='0.5',lw=1,ls=':',label=r'$2S(\mathrm{obs})$')
            ax.set_ylabel(r'$I(\mathrm{obs}:L)$ [nats]' if row==0 else r'$I(\mathrm{obs}:R)$ [nats]')
            ax.grid(alpha=.18)
            ax.set_ylim(bottom=-.03,top=protocol.normalization*1.06)
            if row==0:
                ax.set_title(title)
                ax.legend(fontsize=9)
            else:
                ax.set_xlabel('Coupling time' if col==0 else 'Modular evolution time')
        axes[0,col].axvline(tc if col==0 else te,color='0.65',lw=.8)
        axes[1,col].axvline(tc if col==0 else te,color='0.65',lw=.8)
    p = protocol.p
    fig.suptitle(f'Harmonic rings: N={p.n} per side, m²={p.mass_squared:g}, k={p.spring:g}, '
                 f'β={p.beta:g}, η={p.eta:g}, probe r={p.probe_r:g}\n'
                 'Same marginals and dynamics; coupling excludes the insertion/output pair',fontsize=13)
    fig.savefig(out/'comparison.png',dpi=180)
    fig.savefig(out/'comparison.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    for ax,rows,xkey in zip(axes,[coupling_rows,evolution_rows],['coupling_time','evolution_time']):
        a,b = rows[::2],rows[1::2]
        for key,label,color in [('I_obs_L',r'$\Delta I(\mathrm{obs}:L)$','#1966ab'),
                                 ('I_obs_R',r'$\Delta I(\mathrm{obs}:R)$','#a33a69')]:
            ax.plot([r[xkey] for r in a],[u[key]-v[key] for u,v in zip(a,b)],
                    label=label,color=color,lw=1.7)
        ax.axhline(0,color='0.5',ls=':',lw=1)
        ax.set_xlabel('Coupling time' if xkey=='coupling_time' else 'Modular evolution time')
        ax.set_ylabel('TFD minus product [nats]')
        ax.grid(alpha=.18)
        ax.legend(fontsize=10)
    axes[0].set_title(f'Evolution time = {te:g}')
    axes[1].set_title(f'Coupling time = {tc:g}')
    fig.suptitle('Effect of the initial left–right correlations',fontsize=13)
    fig.savefig(out/'difference.png',dpi=180)
    fig.savefig(out/'difference.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',type=int,default=10)
    parser.add_argument('--mass-squared',type=float,default=13.)
    parser.add_argument('--spring',type=float,default=5.)
    parser.add_argument('--beta',type=float,default=1.)
    parser.add_argument('--eta',type=float,default=1.)
    parser.add_argument('--probe-r',type=float,default=1.)
    parser.add_argument('--probe-frequency',type=float,default=1.)
    parser.add_argument('--insert-site',type=int,default=0)
    parser.add_argument('--evolution-time',type=float,default=3.)
    parser.add_argument('--coupling-time',type=float,default=None,
                        help='Default: first resolved TFD right-side MI peak in coupling scan.')
    parser.add_argument('--coupling-max',type=float,default=16.)
    parser.add_argument('--coupling-points',type=int,default=321)
    parser.add_argument('--evolution-max',type=float,default=15.)
    parser.add_argument('--evolution-points',type=int,default=301)
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args = parser.parse_args()
    if args.coupling_points<3 or args.evolution_points<3 or min(args.coupling_max,args.evolution_max)<=0:
        parser.error('Scans require positive ranges and at least three points.')
    protocol = HarmonicProtocol(Parameters(**{key:getattr(args,key) for key in Parameters.__dataclass_fields__}))
    out = args.out
    out.mkdir(parents=True,exist_ok=True)
    coupling_rows = scan(protocol,np.linspace(0,args.coupling_max,args.coupling_points),
                         'coupling_time',args.evolution_time)
    tfd_rows = [r for r in coupling_rows if r['state']=='tfd']
    peaks, _ = find_peaks([r['I_obs_R'] for r in tfd_rows],prominence=.02*protocol.normalization)
    selected = int(peaks[0]) if len(peaks) else int(np.argmax([r['I_obs_R'] for r in tfd_rows]))
    tc = tfd_rows[selected]['coupling_time'] if args.coupling_time is None else args.coupling_time
    evolution_rows = scan(protocol,np.linspace(0,args.evolution_max,args.evolution_points),
                          'evolution_time',tc)
    stage_rows, covariances = [], {}
    for kind in protocol.resources:
        for stage,v in protocol.stages(kind,args.evolution_time,tc).items():
            stage_rows.append({'state':kind,'stage':stage,**protocol.diagnostics(v)})
            covariances[f'{kind}_{stage}'] = v
    all_rows = coupling_rows+evolution_rows+stage_rows
    max_drift = max(abs(r['I_obs_LR']-protocol.normalization) for r in all_rows)
    if max_drift > 1e-7:
        raise RuntimeError(f'Total observer-system MI drift exceeds tolerance: {max_drift}')
    delta = []
    for a,b in zip(evolution_rows[::2],evolution_rows[1::2]):
        delta.append({'evolution_time':a['evolution_time'],'TFD_minus_product_I_obs_R':a['I_obs_R']-b['I_obs_R']})
    summary = {
        'parameters':asdict(protocol.p),
        'conventions':{'ordering':'interleaved: obs,L0,...,L(N-1),R0,...,R(N-1)',
                       'entropy_units':'nats','vacuum_covariance':'I/2',
                       'H_coupling':'H_L+H_R+(eta/4) sum_(j!=insert) [w0*x_Lj*x_Rj+p_Lj*p_Rj/w0]',
                       'isolated_evolution':'modular beta H','local_terms_during_coupling':'physical H',
                       'probe':'TMS r; local diag(1/sqrt(probe_frequency),sqrt(probe_frequency))'},
        'scan':{'coupling_max':args.coupling_max,'coupling_points':args.coupling_points,
                'evolution_max':args.evolution_max,'evolution_points':args.evolution_points},
        'selected_evolution_time':args.evolution_time,'selected_coupling_time':tc,
        'coupling_selection':('user supplied' if args.coupling_time is not None else
                              'first TFD peak with prominence >= 2% of 2S(obs)' if len(peaks) else
                              'grid maximum fallback; no resolved interior peak'),
        'I_obs_LR_expected':protocol.normalization,'max_total_MI_drift':max_drift,
        'resources':{kind:protocol.resource_diagnostics(v) for kind,v in protocol.resources.items()},
        'selected_final':{r['state']:{k:r[k] for k in ['I_obs_L','I_obs_R','I_obs_LR']}
                          for r in stage_rows if r['stage']=='right_forward'},
        'evolution_scan_largest_TFD_advantage':max(delta,key=lambda r:r['TFD_minus_product_I_obs_R']),
        'evolution_scan_smallest_TFD_advantage':min(delta,key=lambda r:r['TFD_minus_product_I_obs_R']),
    }
    save_csv(out/'coupling_scan.csv',coupling_rows)
    save_csv(out/'evolution_scan.csv',evolution_rows)
    save_csv(out/'stages.csv',stage_rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(out/'covariances.npz',resource_tfd=protocol.tfd,resource_product=protocol.product,
                        probe=protocol.probe,**covariances)
    plot_results(out,protocol,coupling_rows,evolution_rows,args.evolution_time,tc)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
