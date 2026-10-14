"""Physical invariants and alternative constructions, independent of scan plots."""
import unittest
import numpy as np
from scipy.linalg import expm, block_diag
from harmonic_protocol import (Parameters,HarmonicProtocol,omega,indices,reduced,
    entropy,mutual_information,congruence,embed,symplectic_eigenvalues)


class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.protocol = HarmonicProtocol(Parameters(n=4,beta=.7,probe_r=.8))

    def test_tfd_purity_and_product_marginals(self):
        p = self.protocol
        np.testing.assert_allclose(symplectic_eigenvalues(p.tfd),.5,atol=2e-12)
        np.testing.assert_allclose(p.tfd @ omega(8) @ p.tfd,omega(8)/4,atol=2e-12)
        for side in [p.resource_left,p.resource_right]:
            np.testing.assert_array_equal(reduced(p.tfd,side),reduced(p.product,side))
        self.assertAlmostEqual(mutual_information(p.product,p.resource_left,p.resource_right),0,places=10)
        self.assertAlmostEqual(entropy(p.product),2*entropy(reduced(p.tfd,p.resource_left)),places=10)

    def test_analytic_evolution_against_matrix_exponential(self):
        p = self.protocol
        for t in [-2.,0.,.73]:
            s = p.chain_evolution(t)
            expected = expm(omega(4) @ p.h_chain*p.p.beta*t)
            np.testing.assert_allclose(s,expected,atol=2e-12)
            np.testing.assert_allclose(s @ omega(4) @ s.T,omega(4),atol=2e-12)

    def test_thermal_stationarity_and_tfd_two_sided_stationarity(self):
        p = self.protocol
        s = p.chain_evolution(.83)
        local = reduced(p.tfd,p.resource_left)
        np.testing.assert_allclose(congruence(s,local),local,atol=2e-12)
        both = block_diag(p.chain_evolution(-.83),s)
        np.testing.assert_allclose(congruence(both,p.tfd),p.tfd,atol=2e-12)

    def test_discard_and_replace_against_explicit_swap(self):
        p = self.protocol
        resource = congruence(embed(p.chain_evolution(-.6),p.resource_left,8),p.tfd)
        # Explicit larger system: resource, observer, input. Swap input with L0,
        # then trace out the old L0, now occupying the input ancilla slot.
        full = block_diag(resource,p.probe)
        permutation = np.arange(20)
        a,b = indices([0]),indices([9])
        permutation[a],permutation[b] = permutation[b].copy(),permutation[a].copy()
        full = full[np.ix_(permutation,permutation)]
        explicit = reduced(full,[8]+list(range(8)))
        np.testing.assert_allclose(p.insert_probe(resource),explicit,atol=1e-14)

    def test_entropy_against_known_thermal_modes(self):
        p = self.protocol
        occupation = 1/np.expm1(p.p.beta*p.frequencies)
        expected = np.sum((occupation+1)*np.log1p(occupation)-occupation*np.log(occupation))
        self.assertAlmostEqual(entropy(reduced(p.tfd,p.resource_left)),expected,places=11)

    def test_no_signalling_without_coupling(self):
        p = HarmonicProtocol(Parameters(n=4,eta=0,probe_r=.8))
        for kind in p.resources:
            for t in [0.,.5,2.]:
                v = p.stages(kind,t,3.)['right_forward']
                d = p.diagnostics(v)
                self.assertAlmostEqual(d['I_obs_R'],0,places=9)
                self.assertAlmostEqual(d['I_obs_L'],p.normalization,places=9)

    def test_zero_evolution_and_zero_coupling_duration(self):
        p = self.protocol
        for kind in p.resources:
            for te,tc in [(0.,0.),(2.,0.)]:
                d = p.diagnostics(p.stages(kind,te,tc)['right_forward'])
                self.assertAlmostEqual(d['I_obs_R'],0,places=9)
                self.assertAlmostEqual(d['I_obs_L'],p.normalization,places=9)

    def test_no_springs_excluded_pair_cannot_transfer(self):
        p = HarmonicProtocol(Parameters(n=4,spring=0,eta=1))
        for kind in p.resources:
            d = p.diagnostics(p.stages(kind,1.3,4.1)['right_forward'])
            self.assertAlmostEqual(d['I_obs_R'],0,places=9)
            self.assertAlmostEqual(d['I_obs_L'],p.normalization,places=9)

    def test_stage_invariants_and_total_MI(self):
        p = self.protocol
        for kind in p.resources:
            stages = p.stages(kind,1.3,4.1)
            for stage,v in stages.items():
                d = p.diagnostics(v)
                self.assertAlmostEqual(d['I_obs_LR'],p.normalization,places=8)
                np.testing.assert_allclose(v[:2,:2],p.probe[:2,:2],atol=2e-12)
                if stage in ['inserted','left_forward']:
                    self.assertAlmostEqual(d['I_obs_R'],0,places=9)
                    self.assertAlmostEqual(d['I_obs_L'],p.normalization,places=9)
            a,b = p.diagnostics(stages['coupled']),p.diagnostics(stages['right_forward'])
            for key in a:
                self.assertAlmostEqual(a[key],b[key],places=9)

    def test_same_observer_cross_covariances_for_both_resources(self):
        p = self.protocol
        a = p.stages('tfd',1.3,4.1)['right_forward']
        b = p.stages('product',1.3,4.1)['right_forward']
        # Fixed linear dynamics imply the same signal transfer X, but different
        # environmental output noise. Since probe C is invertible, these cross
        # blocks directly determine X to the entire L+R output.
        np.testing.assert_allclose(a[:2,2:],b[:2,2:],atol=2e-12)
        self.assertGreater(np.linalg.norm(a[2:,2:]-b[2:,2:]),1e-3)

    def test_coupling_generator_and_excluded_pair(self):
        p = self.protocol
        h = p.h_coupling
        for q in [0,1]:
            self.assertEqual(h[q,8+q],0.)
        self.assertAlmostEqual(h[2,10],p.p.eta*p.w0/4)
        self.assertAlmostEqual(h[3,11],p.p.eta/(4*p.w0))
        s = p.coupling_evolution(1.7)
        np.testing.assert_allclose(s @ omega(8) @ s.T,omega(8),atol=2e-12)

    def test_unphysical_covariance_is_rejected(self):
        with self.assertRaises(ValueError):
            entropy(.4*np.eye(2))

    def test_site_MI_insertion_and_no_signalling(self):
        p = self.protocol
        for kind in p.resources:
            stages = p.stages(kind,1.3,4.1)
            expected = np.zeros((2,p.p.n))
            expected[0,p.p.insert_site] = p.normalization
            np.testing.assert_allclose(p.site_mutual_information(stages['inserted']),expected,atol=2e-11)
            np.testing.assert_allclose(p.site_mutual_information(stages['left_forward'])[1],0,atol=2e-11)

    def test_site_MI_ring_translation_and_reflection(self):
        p = self.protocol
        shifted = HarmonicProtocol(Parameters(n=4,beta=.7,probe_r=.8,insert_site=1))
        for kind in p.resources:
            a = p.site_mutual_information(p.stages(kind,1.3,4.1)['right_forward'])
            b = shifted.site_mutual_information(shifted.stages(kind,1.3,4.1)['right_forward'])
            np.testing.assert_allclose(b,np.roll(a,1,axis=1),atol=2e-11)
            np.testing.assert_allclose(a,a[:,(-np.arange(p.p.n)) % p.p.n],atol=2e-11)

    def test_two_mode_entropy_with_alternative_symplectic_invariants(self):
        p = self.protocol
        v = p.stages('tfd',1.3,4.1)['right_forward']
        for mode in p.left+p.right:
            two = reduced(v,[0,mode])
            delta = np.linalg.det(two[:2,:2])+np.linalg.det(two[2:,2:])+2*np.linalg.det(two[:2,2:])
            discriminant = max(delta*delta-4*np.linalg.det(two),0.)
            nu = np.sqrt((delta+np.array([-1,1])*np.sqrt(discriminant))/2)
            np.testing.assert_allclose(symplectic_eigenvalues(two),nu,atol=2e-12)


if __name__=='__main__':
    unittest.main(verbosity=2)
