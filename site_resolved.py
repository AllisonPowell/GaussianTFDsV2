"""Site-resolved MI using the settings of an existing whole-side run.

Run: OPENBLAS_NUM_THREADS=1 python independent_protocol/site_resolved.py
Both preparations use the same times, including at plotted optimal snapshots.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

from harmonic_protocol import Parameters, HarmonicProtocol, congruence, embed, save_csv


def evolve_pair(protocol, te, coupling):
    right = embed(protocol.chain_evolution(te),protocol.right,2*protocol.p.n+1)
    answer = {}
    for kind in protocol.resources:
        coupled = congruence(coupling,protocol.before_coupling(kind,te))
        final = congruence(right,coupled)
        answer[kind] = (coupled,final)
    return answer


def refine_peak(protocol, kind, times, curve, coupling, selected_index=None):
    """Refine the largest sampled receiver peak in its adjacent grid interval.

    This does not claim a certified global maximum between all grid points.
    """
    j = int(np.argmax(curve)) if selected_index is None else int(selected_index)
    if j in [0,len(times)-1]:
        return float(times[j]),float(curve[j])
    def objective(te):
        final = evolve_pair(protocol,te,coupling)[kind][1]
        return -protocol.site_mutual_information(final)[1,protocol.p.insert_site]
    result = minimize_scalar(objective,bounds=(times[j-1],times[j+1]),method='bounded',
                             options={'xatol':1e-9})
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x),float(-result.fun)


def plot(out,p,times,site,pre,whole,snapshots,tc):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaussian-teleport-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors = {'tfd':'#1966ab','product':'#d45a1b'}
    names = {'tfd':'TFD','product':r'$\rho_L\otimes\rho_R$'}
    def save(fig,name):
        fig.savefig(out/f'{name}.png',dpi=180)
        fig.savefig(out/f'{name}.pdf')
        plt.close(fig)

    fig,axes = plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True,layout='constrained')
    # Shared absolute scale across resources AND sides; separate symmetric
    # difference scale across sides, with no per-panel renormalization.
    vmax = max(np.max(a) for a in site.values())
    difference = site['tfd']-site['product']
    dmax = max(np.max(np.abs(difference)),1e-12)
    for side in range(2):
        for col in range(3):
            ax = axes[side,col]
            values = site['tfd'][:,side,:] if col==0 else site['product'][:,side,:] if col==1 else difference[:,side,:]
            mesh = ax.pcolormesh(np.arange(p.p.n),times,values,shading='nearest',
                                cmap='magma' if col<2 else 'RdBu_r',
                                vmin=0 if col<2 else -dmax,vmax=vmax if col<2 else dmax,
                                rasterized=True)
            ax.axvline(p.p.insert_site,color='cyan' if col<2 else '0.2',ls=':',lw=1)
            ax.set_xticks(np.arange(p.p.n))
            ax.set_title(('Left' if side==0 else 'Right')+' — '+['TFD','Product','TFD − product'][col])
            if col==0: ax.set_ylabel('Modular evolution time')
            if side==1: ax.set_xlabel('Site index on ring')
            fig.colorbar(mesh,ax=ax,label='MI [nats]' if col<2 else 'ΔMI [nats]',shrink=.85)
    fig.suptitle(f'Site-resolved observer MI after the full protocol; coupling time = {tc:g}\n'
                 f'N={p.p.n}, m²={p.p.mass_squared:g}, k={p.p.spring:g}, β={p.p.beta:g}, '
                 f'η={p.p.eta:g}, probe r={p.p.probe_r:g}; dotted line: insertion/teleport site')
    save(fig,'site_heatmaps')

    fig,axes = plt.subplots(2,2,figsize=(12,7),sharex=True,layout='constrained')
    j = p.p.insert_site
    for kind in p.resources:
        style = '-' if kind=='tfd' else '--'
        for row,values in enumerate([site,pre]):
            axes[row,0].plot(times,values[kind][:,1,j],color=colors[kind],ls=style,lw=1.8,label=names[kind])
        axes[0,1].plot(times,whole[kind][:,1],color=colors[kind],ls=style,lw=1.8,label=names[kind])
    axes[1,1].plot(times,difference[:,1,j],color='#a33a69',lw=1.5,label='TFD − product')
    axes[1,1].axhline(0,color='0.5',ls=':',lw=1)
    axes[0,0].set_title(f'Teleport site R{j}: after final right evolution')
    axes[1,0].set_title(f'Same site R{j}: before final right evolution')
    axes[0,1].set_title('Entire right side: after full protocol')
    axes[1,1].set_title(f'Difference at R{j}: after full protocol')
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.set_ylabel('MI [nats]' if ax is not axes[1,1] else 'ΔMI [nats]')
        ax.legend(fontsize=9)
    for ax in axes[1,:]: ax.set_xlabel('Modular evolution time')
    fig.suptitle(f'Local recovery versus whole-side transfer; coupling time = {tc:g}')
    save(fig,'teleport_site')

    fig,axes = plt.subplots(2,len(snapshots),figsize=(4.5*len(snapshots),6),sharex=True,sharey='row',layout='constrained')
    for col,snapshot in enumerate(snapshots):
        for side in range(2):
            ax = axes[side,col]
            for kind in p.resources:
                ax.plot(np.arange(p.p.n),snapshot[kind]['site_MI'][side],
                        marker='o',ms=4,color=colors[kind],ls='-' if kind=='tfd' else '--',label=names[kind])
            ax.axvline(j,color='0.6',ls=':',lw=1)
            ax.set_xticks(np.arange(p.p.n))
            ax.grid(alpha=.18)
            ax.set_xlabel('Site index on ring')
            ax.set_ylabel('I(obs:'+('L' if side==0 else 'R')+'j) [nats]')
            ax.legend(fontsize=9)
            if side==0: ax.set_title(snapshot['label']+f"\nt = {snapshot['evolution_time']:.5f}")
    fig.suptitle('Spatial profiles at common times for both preparations')
    save(fig,'site_profiles')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(__file__).parent/'results'/'summary.json')
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'site_results')
    parser.add_argument('--evolution-points',type=int,default=None)
    args = parser.parse_args()
    source = json.loads(args.source.read_text())
    p = HarmonicProtocol(Parameters(**source['parameters']))
    tc = source['selected_coupling_time']
    points = args.evolution_points or source['scan']['evolution_points']
    if points<3: parser.error('Need at least three time points.')
    times = np.linspace(0,source['scan']['evolution_max'],points)
    coupling = embed(p.coupling_evolution(tc),p.all_system,2*p.p.n+1)
    site = {k:np.zeros((points,2,p.p.n)) for k in p.resources}
    pre = {k:np.zeros((points,2,p.p.n)) for k in p.resources}
    whole = {k:np.zeros((points,3)) for k in p.resources}
    rows = []
    max_cross_difference = 0.
    max_local_invariance_error = 0.
    for i,te in enumerate(times):
        pair = evolve_pair(p,te,coupling)
        max_cross_difference = max(max_cross_difference,float(np.max(np.abs(
            pair['tfd'][1][:2,2:]-pair['product'][1][:2,2:]))))
        for kind,(coupled,final) in pair.items():
            site[kind][i] = p.site_mutual_information(final)
            pre[kind][i] = p.site_mutual_information(coupled)
            d = p.diagnostics(final)
            whole[kind][i] = [d['I_obs_L'],d['I_obs_R'],d['I_obs_LR']]
            # Right-only evolution must leave EVERY left-site reduction intact.
            max_local_invariance_error = max(max_local_invariance_error,
                                            float(np.max(np.abs(site[kind][i,0]-pre[kind][i,0]))))
            if site[kind][i].min() < -1e-8:
                raise RuntimeError('Negative site MI beyond roundoff.')
            if np.any(site[kind][i]>whole[kind][i,:2,None]+1e-8):
                raise RuntimeError('Site MI exceeds the MI of its containing side.')
            for side in range(2):
                for j in range(p.p.n):
                    rows.append({'state':kind,'evolution_time':float(te),'coupling_time':tc,
                                 'side':['L','R'][side],'site':j,
                                 'I_obs_site':float(site[kind][i,side,j]),
                                 'I_obs_site_before_final_right':float(pre[kind][i,side,j]),
                                 'I_obs_whole_side':float(whole[kind][i,side])})
    j = p.p.insert_site
    peaks = {kind:refine_peak(p,kind,times,site[kind][:,1,j],coupling) for kind in p.resources}
    first_peaks = {}
    for kind in p.resources:
        found,_ = find_peaks(site[kind][:,1,j],prominence=.02*p.normalization)
        first_peaks[kind] = (refine_peak(p,kind,times,site[kind][:,1,j],coupling,found[0])
                             if len(found) else None)
    specifications = [('Previous common time',source['selected_evolution_time']),
                      ('Strongest TFD receiver peak',peaks['tfd'][0])]
    if first_peaks['tfd'] is not None:
        specifications.insert(1,('First TFD receiver peak',first_peaks['tfd'][0]))
    snapshots = []
    covariances = {}
    for number,(label,te) in enumerate(specifications):
        snapshot = {'label':label,'evolution_time':te}
        for kind,(coupled,v) in evolve_pair(p,te,coupling).items():
            local = p.site_mutual_information(v)
            snapshot[kind] = {'site_MI':local.tolist(),'whole_MI':p.diagnostics(v),
                              'teleport_site_MI':float(local[1,j]),
                              'teleport_site_MI_before_final_right':float(p.site_mutual_information(coupled)[1,j]),
                              'right_max_site':int(np.argmax(local[1]))}
            covariances[f'snapshot{number}_{kind}'] = v
        snapshots.append(snapshot)
    difference = site['tfd']-site['product']
    max_index = np.unravel_index(np.argmax(difference),difference.shape)
    min_index = np.unravel_index(np.argmin(difference),difference.shape)
    def extreme(index):
        i,side,jj = index
        return {'time':float(times[i]),'side':['L','R'][side],'site':int(jj),
                'TFD_minus_product':float(difference[index])}
    total_drift = max(float(np.max(np.abs(w[:,2]-p.normalization))) for w in whole.values())
    if total_drift>1e-7 or max_cross_difference>1e-9 or max_local_invariance_error>1e-8:
        raise RuntimeError('A physical invariance check failed.')
    summary = {'parameters':source['parameters'],'source_summary':str(args.source.resolve()),
               'coupling_time':tc,'evolution_max':float(times[-1]),'evolution_points':points,
               'units':'nats','I_obs_LR':p.normalization,
               'peak_prescription':'Largest sampled R_insert peak, refined within neighboring grid points; endpoints retained.',
               'receiver_peaks':{k:{'time':v[0],'MI':v[1]} for k,v in peaks.items()},
               'first_receiver_peak_prescription':'First interior sampled R_insert peak with prominence >= 2% of 2S(obs), locally refined.',
               'first_receiver_peaks':{k:None if v is None else {'time':v[0],'MI':v[1]} for k,v in first_peaks.items()},
               'snapshots':snapshots,'largest_site_difference':extreme(max_index),
               'smallest_site_difference':extreme(min_index),
               'checks':{'max_total_MI_drift':total_drift,
                         'max_observer_cross_covariance_difference':max_cross_difference,
                         'max_left_site_MI_change_under_final_right_evolution':max_local_invariance_error,
                         'site_MI_bounded_by_whole_side_MI':True}}
    args.out.mkdir(parents=True,exist_ok=True)
    save_csv(args.out/'site_scan.csv',rows)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(args.out/'site_data.npz',times=times,
                        **{f'{k}_final':v for k,v in site.items()},
                        **{f'{k}_before_final_right':v for k,v in pre.items()},
                        **{f'{k}_whole':v for k,v in whole.items()},**covariances)
    plot(args.out,p,times,site,pre,whole,snapshots,tc)
    print(json.dumps({k:v for k,v in summary.items() if k!='snapshots'},indent=2))
    for snapshot in snapshots:
        print(snapshot['label'],snapshot['evolution_time'],
              {k:snapshot[k]['teleport_site_MI'] for k in p.resources})


if __name__=='__main__':
    main()
