"""Finite-energy TMS entanglement fidelity for the two harmonic resources.

The target is the original pure observer-input TMS. Compare it to the
observer-output state, either raw or with output D=sqrt(det X)*inv(X).
This shared decoder removes the deterministic symplectic part of X; it is
not asserted to optimize fidelity in the presence of anisotropic noise.
Both root and squared fidelities are recorded. No covariance tomography fit.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

from harmonic_protocol import (Parameters,HarmonicProtocol,omega,indices,reduced,
    congruence,embed,symplectic_eigenvalues,save_csv)


def pure_target_root_fidelity(target, output):
    """Zero-mean Gaussian root fidelity when target is pure, hbar=1.

    Froot=det(Vtarget+Voutput)^(-1/4); Fsq=Froot**2.
    Banchi et al., https://arxiv.org/html/1507.01941v1, Eq. (12) and pure limit.
    """
    if not np.allclose(symplectic_eigenvalues(target),.5,rtol=0,atol=1e-9):
        raise ValueError('The determinant shortcut requires a pure target.')
    symplectic_eigenvalues(output)
    sign, logdet = np.linalg.slogdet(target+output)
    if sign<=0:
        raise ValueError('Nonpositive covariance sum.')
    result = float(np.exp(-logdet/4))
    if result>1+1e-8:
        raise ValueError(f'Fidelity exceeds one: {result}')
    return min(result,1.)


def output_decoder(x):
    """One-mode symplectic D with D X=sqrt(det X) I, when det X>0.

    Singular and orientation-reversing channels have no such decoder.
    Return None and leave decoded fidelity undefined rather than inventing
    an inverse, using an anti-symplectic operation, or silently substituting I.
    """
    determinant = float(np.linalg.det(x))
    if determinant<=1e-12 or np.linalg.cond(x)>1e10:
        return None
    return np.sqrt(determinant)*np.linalg.inv(x)


def channel_on_probe(probe, x, y):
    a, c, b = probe[:2,:2],probe[:2,2:],probe[2:,2:]
    return np.block([[a,c @ x.T],[x @ c.T,x @ b @ x.T+y]])


def fidelities(probe, x, y):
    output = channel_on_probe(probe,x,y)
    raw = pure_target_root_fidelity(probe,output)
    d = output_decoder(x)
    decoded = (float('nan') if d is None else
               pure_target_root_fidelity(probe,congruence(block_diag(np.eye(2),d),output)))
    return raw,decoded


def evaluate(protocol, te, tc, coupling=None):
    """Extract X,Y directly from the post-insertion propagator and resource.

    R_out = X R_input + B R_environment, Y=B V_environment B^T.
    The environment is the backward-evolved resource with the insertion
    mode traced out. Its internal LR correlations must be retained for TFD.
    """
    p = protocol
    n = p.p.n
    coupling = p.coupling_evolution(tc) if coupling is None else coupling
    forward_l = embed(p.chain_evolution(te),p.resource_left,2*n)
    forward_r = embed(p.chain_evolution(te),p.resource_right,2*n)
    transfer = forward_r @ coupling @ forward_l
    backward = embed(p.chain_evolution(-te),p.resource_left,2*n)
    input_q = indices([p.p.insert_site])
    environment_q = np.array([i for i in range(4*n) if i not in input_q])
    backward_resources = {kind:congruence(backward,v) for kind,v in p.resources.items()}
    # Independent covariance construction for channel-output checks.
    extended = embed(transfer,p.all_system,2*n+1)
    finals = {kind:congruence(extended,p.insert_probe(v)) for kind,v in backward_resources.items()}
    rows, channel_data = [], {}
    max_error, min_cp = 0.,float('inf')
    for site in range(n):
        outq = indices([n+site])
        x = transfer[np.ix_(outq,input_q)]
        b = transfer[np.ix_(outq,environment_q)]
        d = output_decoder(x)
        channel_data[site] = {'X':x,'D':d}
        for kind,v in backward_resources.items():
            environment = v[np.ix_(environment_q,environment_q)]
            y = b @ environment @ b.T
            y = (y+y.T)/2
            channel_data[site][kind+'_Y'] = y
            output = channel_on_probe(p.probe,x,y)
            direct = reduced(finals[kind],[0,n+1+site])
            error = float(np.max(np.abs(output-direct)))
            max_error = max(max_error,error)
            cp_min = float(np.linalg.eigvalsh(y+.5j*(omega(1)-x @ omega(1) @ x.T)).min())
            min_cp = min(min_cp,cp_min)
            if error>1e-9 or cp_min < -1e-9:
                raise RuntimeError('Channel extraction or complete positivity check failed.')
            raw,decoded = fidelities(p.probe,x,y)
            rows.append({'state':kind,'evolution_time':float(te),'coupling_time':float(tc),'site':site,
                         'root_raw':raw,'squared_raw':raw**2,
                         'root_decoded':decoded,'squared_decoded':decoded**2,
                         'det_X':float(np.linalg.det(x)),'cond_X':float(np.linalg.cond(x)),
                         'decoder_defined':d is not None,'Y_xx':float(y[0,0]),
                         'Y_xp':float(y[0,1]),'Y_pp':float(y[1,1]),
                         'channel_covariance_error':error,'CP_min_eigenvalue':cp_min})
    return rows,channel_data,{'max_channel_covariance_error':max_error,'min_CP_eigenvalue':min_cp}


def finite_json(value):
    """Make unavailable decoded scores explicit JSON nulls, never NaNs."""
    if isinstance(value,np.ndarray): return finite_json(value.tolist())
    if isinstance(value,dict): return {str(k):finite_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [finite_json(v) for v in value]
    if isinstance(value,(np.floating,float)) and not np.isfinite(value): return None
    return value


def plot(out,p,evolution,coupling,snapshots,tc,te):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaussian-teleport-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors = {'tfd':'#1966ab','product':'#d45a1b'}
    labels = {'tfd':'TFD','product':r'$\rho_L\otimes\rho_R$'}
    receiver = p.p.insert_site
    def save(fig,name):
        fig.savefig(out/f'{name}.pdf')
        fig.savefig(out/f'{name}.png',dpi=180)
        plt.close(fig)
    fig,axes = plt.subplots(3,2,figsize=(12,10),sharex='col',layout='constrained')
    for col,(rows,key,title) in enumerate([(coupling,'coupling_time',f'Evolution time = {te:g}'),
                                          (evolution,'evolution_time',f'Coupling time = {tc:g}')]):
        selected = {kind:[r for r in rows if r['state']==kind and r['site']==receiver] for kind in p.resources}
        for kind in p.resources:
            for row,measure in enumerate(['root_raw','root_decoded']):
                data = selected[kind]
                axes[row,col].plot([r[key] for r in data],[r[measure] for r in data],
                                   color=colors[kind],ls='-' if kind=='tfd' else '--',label=labels[kind],lw=1.7)
                axes[row,col].set_ylim(0,1)
                axes[row,col].legend(fontsize=9)
        a,b = selected['tfd'],selected['product']
        for measure,label,color in [('root_raw','Raw','#6a6a6a'),('root_decoded','Decoded','#a33a69')]:
            axes[2,col].plot([r[key] for r in a],[u[measure]-v[measure] for u,v in zip(a,b)],
                             color=color,label=label,lw=1.5)
        axes[2,col].axhline(0,color='0.5',ls=':',lw=1)
        axes[2,col].legend(fontsize=9)
        axes[0,col].set_title(title)
        axes[2,col].set_xlabel('Coupling time' if col==0 else 'Modular evolution time')
        for ax in axes[:,col]: ax.grid(alpha=.18)
    axes[0,0].set_ylabel('Root fidelity: raw output')
    axes[1,0].set_ylabel('Root fidelity: common decoder')
    axes[2,0].set_ylabel('TFD − product root fidelity')
    fig.suptitle(f'Finite-energy TMS test at R{receiver}; probe r={p.p.probe_r:g}\n'
                 r'Root convention; output decoder $D=\sqrt{\det X}\,X^{-1}$; no decoder optimization')
    save(fig,'fidelity_comparison')

    fig,axes = plt.subplots(1,len(snapshots),figsize=(4.5*len(snapshots),3.7),sharey=True,layout='constrained')
    for ax,snapshot in zip(np.atleast_1d(axes),snapshots):
        for kind in p.resources:
            rows = [r for r in snapshot['rows'] if r['state']==kind]
            ax.plot([r['site'] for r in rows],[r['root_decoded'] for r in rows],
                    marker='o',color=colors[kind],ls='-' if kind=='tfd' else '--',label=labels[kind])
        ax.axvline(receiver,color='0.6',ls=':')
        ax.set_xlabel('Right-ring site')
        ax.set_title(snapshot['label']+f"\nt = {snapshot['evolution_time']:.4f}")
        ax.set_xticks(range(p.p.n));ax.grid(alpha=.18);ax.legend(fontsize=9)
    axes[0].set_ylabel('Root fidelity after common decoder')
    fig.suptitle('Site-resolved finite-energy entanglement fidelity\n'
                 'Gaps indicate an undefined decoder; raw fidelity remains available',fontsize=11)
    save(fig,'fidelity_sites')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).parent/'results'/'summary.json')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'fidelity_results')
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    p = HarmonicProtocol(Parameters(**source['parameters']))
    te,tc = source['selected_evolution_time'],source['selected_coupling_time']
    times = np.linspace(0,source['scan']['evolution_max'],source['scan']['evolution_points'])
    coupling_times = np.linspace(0,source['scan']['coupling_max'],source['scan']['coupling_points'])
    fixed_coupling = p.coupling_evolution(tc)
    evolution,coupling,checks = [],[],[]
    for t in times:
        rows,_,check = evaluate(p,t,tc,fixed_coupling)
        evolution.extend(rows);checks.append(check)
    for t in coupling_times:
        rows,_,check = evaluate(p,te,t)
        coupling.extend(rows);checks.append(check)
    receiver = p.p.insert_site
    selected = {k:[r for r in evolution if r['state']==k and r['site']==receiver] for k in p.resources}
    peak = max((r for r in selected['tfd'] if np.isfinite(r['root_decoded'])),key=lambda r:r['root_decoded'])
    specifications = [('Previous common time',te),('TFD decoded-fidelity grid peak',peak['evolution_time'])]
    snapshots = []
    arrays = {}
    for i,(label,t) in enumerate(specifications):
        rows,channels,_ = evaluate(p,t,tc,fixed_coupling)
        x = channels[receiver]['X']
        y_tfd,y_product = channels[receiver]['tfd_Y'],channels[receiver]['product_Y']
        snapshot = {'label':label,'evolution_time':t,'rows':rows,'receiver_channel':channels[receiver],
                    'receiver_delta_Y':y_tfd-y_product,
                    'relative_receiver_noise_difference':float(np.linalg.norm(y_tfd-y_product)/np.linalg.norm(y_product))}
        snapshots.append(snapshot)
        arrays[f'snapshot{i}_X'] = x
        for kind in p.resources: arrays[f'snapshot{i}_{kind}_Y'] = channels[receiver][kind+'_Y']
        if channels[receiver]['D'] is not None: arrays[f'snapshot{i}_D'] = channels[receiver]['D']
    deltas = [a['root_decoded']-b['root_decoded'] for a,b in zip(selected['tfd'],selected['product'])]
    summary = {'parameters':source['parameters'],'source_summary':str(args.source.resolve()),
               'target':'Original pure observer-input TMS, with fixed probe r and quadrature scaling.',
               'fidelity':'root = det(V_target+V_output)^(-1/4); squared = root^2; zero means',
               'reference':'https://arxiv.org/html/1507.01941v1',
               'decoder':'D=sqrt(det X) inv(X), identical for both resources; not fidelity optimized.',
               'undefined_decoder':'det X<=1e-12 or cond X>1e10: decoded scores unavailable (NaN in CSV, null in JSON).',
               'coupling_time':tc,'fixed_evolution_time':te,
               'grid':source['scan'],
               'receiver_evolution_peaks':{kind:max((r for r in data if np.isfinite(r['root_decoded'])),key=lambda r:r['root_decoded'])
                                            for kind,data in selected.items()},
               'receiver_decoded_difference_range':[float(np.nanmin(deltas)),float(np.nanmax(deltas))],
               'snapshots':snapshots,
               'checks':{'max_channel_covariance_error':max(c['max_channel_covariance_error'] for c in checks),
                         'min_CP_eigenvalue':min(c['min_CP_eigenvalue'] for c in checks),
                         'receiver_evolution_undefined_decoders':sum(not r['decoder_defined'] for r in selected['tfd'])}}
    args.out.mkdir(parents=True,exist_ok=True)
    save_csv(args.out/'evolution_scan.csv',evolution)
    save_csv(args.out/'coupling_scan.csv',coupling)
    (args.out/'summary.json').write_text(json.dumps(finite_json(summary),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(args.out/'receiver_channels.npz',**arrays)
    plot(args.out,p,evolution,coupling,snapshots,tc,te)
    print(json.dumps(finite_json({k:v for k,v in summary.items() if k!='snapshots'}),indent=2))
    for snapshot in snapshots:
        print(snapshot['label'],snapshot['evolution_time'])
        print(json.dumps(finite_json([r for r in snapshot['rows'] if r['site']==receiver]),indent=2))
        print('Delta Y:',snapshot['receiver_delta_Y'])


if __name__=='__main__':
    main()
