import unittest
import numpy as np
from scipy.linalg import block_diag
from harmonic_protocol import Parameters,HarmonicProtocol,congruence
from fidelity_comparison import (pure_target_root_fidelity,output_decoder,
    channel_on_probe,fidelities,evaluate)


class FidelityChecks(unittest.TestCase):
    def setUp(self):
        self.p = HarmonicProtocol(Parameters(n=4,probe_r=.8))
        self.probe = self.p.probe

    def test_identity_and_noiseless_symplectic_correction(self):
        rotation = np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
        x = rotation @ np.diag([2.,.5])
        self.assertAlmostEqual(pure_target_root_fidelity(self.probe,self.probe),1.,places=12)
        np.testing.assert_allclose(output_decoder(x) @ x,np.eye(2),atol=1e-12)
        raw,decoded = fidelities(self.probe,x,np.zeros((2,2)))
        self.assertLess(raw,1.)
        self.assertAlmostEqual(decoded,1.,places=12)

    def test_pi_phase_flip_from_fock_overlap(self):
        output = congruence(block_diag(np.eye(2),-np.eye(2)),self.probe)
        # Sum_n (1-q) (-q)^n = 1/cosh(2r), the absolute pure-state overlap.
        expected = 1/np.cosh(2*self.p.p.probe_r)
        self.assertAlmostEqual(pure_target_root_fidelity(self.probe,output),expected,places=12)

    def test_loss_from_fock_overlap(self):
        q = np.tanh(self.p.p.probe_r)**2
        for tau in [0.,.2,.8,1.]:
            output = channel_on_probe(self.probe,np.sqrt(tau)*np.eye(2),(1-tau)*np.eye(2)/2)
            # Only the zero-lost-photon Kraus operator has a nonzero overlap.
            expected_root = (1-q)/(1-np.sqrt(tau)*q)
            self.assertAlmostEqual(pure_target_root_fidelity(self.probe,output),expected_root,places=12)

    def test_standard_teleportation_analytic_benchmark(self):
        for resource_r in [0.,.5,1.5]:
            noise = np.exp(-2*resource_r)
            output = channel_on_probe(self.probe,np.eye(2),noise*np.eye(2))
            expected = 1/np.sqrt(1+noise*np.cosh(2*self.p.p.probe_r))
            self.assertAlmostEqual(pure_target_root_fidelity(self.probe,output),expected,places=12)

    def test_singular_and_negative_determinant_decoders(self):
        self.assertIsNone(output_decoder(np.zeros((2,2))))
        self.assertIsNone(output_decoder(np.diag([1.,-1.])))
        self.assertIsNone(output_decoder(np.diag([1.,0.])))

    def test_mixed_target_rejected(self):
        with self.assertRaises(ValueError):
            pure_target_root_fidelity(self.probe+.1*np.eye(4),self.probe)

    def test_extracted_channel_against_protocol_and_probe_changes(self):
        a = self.p
        b = HarmonicProtocol(Parameters(n=4,probe_r=.4,probe_frequency=2.))
        _,ca,checks = evaluate(a,1.3,4.1)
        _,cb,_ = evaluate(b,1.3,4.1)
        self.assertLess(checks['max_channel_covariance_error'],1e-12)
        self.assertGreater(checks['min_CP_eigenvalue'],-1e-12)
        for j in range(a.p.n):
            for key in ['X','tfd_Y','product_Y']:
                np.testing.assert_allclose(ca[j][key],cb[j][key],atol=1e-12)


if __name__=='__main__':
    unittest.main(verbosity=2)
