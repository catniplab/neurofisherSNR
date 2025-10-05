import numpy as np
import pytest

from neurofisherSNR.optimize import initialize_C
from neurofisherSNR.snr import SNR_bound_instantaneous, SNR_bound_instantaneous_vectorized


class TestSNRBoundInstantaneous:
    """Test class for checking input and output dimensionality of SNR_bound_instantaneous function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducible tests
        np.random.seed(20250710)

        # Test parameters
        self.time_steps = 100
        self.d_latent = 3
        self.d_neurons = 10
        self.tgt_rate_per_bin = 0.1
        self.max_rate_per_bin = 1.0
        self.tgt_snr = 5.0

        # Create test data
        self.x = np.random.randn(self.time_steps, self.d_latent)
        self.C = initialize_C(
            self.d_latent, self.d_neurons, p_coh=0.5, p_sparse=0.0)
        self.b = np.random.randn(1, self.d_neurons)

    def test_snr_bound_instantaneous_simple(self):
        snr = SNR_bound_instantaneous(self.x, self.C.T, self.b)
        assert np.isfinite(snr), "SNR should be finite."
        assert snr > 0, "SNR should be positive for this input."

    def test_snr_bound_instantaneous_zero_signal(self):
        """Test SNR when x = 0 (zero signal), which should give -inf SNR."""
        # Create zero signal with d_latent = 1
        time_steps = 50
        d_latent = 1
        d_neurons = 5

        x_zero = np.zeros((time_steps, d_latent))
        C = initialize_C(d_latent, d_neurons, p_coh=0.5, p_sparse=0.0)
        b = np.random.randn(1, d_neurons)

        snr = SNR_bound_instantaneous(x_zero, C.T, b)
        assert np.isinf(
            snr) and snr < 0, f"SNR should be -inf for zero signal, got {snr}"

    def test_snr_bound_instantaneous_single_neuron_closed_form(self):
        """Test SNR for 1 latent and 1 neuron with known scalar solution

        For 1 latent and 1 neuron:
        - CT is a scalar (1x1 matrix)
        - Fisher information = CT^2 * lambda where lambda is mean firing rate
        - SNR = x_power / (1/FI) = x_power * CT^2 * lambda
        """
        time_steps = 100
        d_latent = 1
        x = np.random.randn(time_steps, d_latent)  # Standard normal
        CT = np.array([[2.0]])  # Single scalar loading
        b = np.array([[0.0]])   # Zero bias for simplicity

        # Compute SNR using the function
        snr_dB = SNR_bound_instantaneous(x, CT, b)

        from neurofisherSNR.utils import compute_firing_rate

        firing_rates = compute_firing_rate(x, CT, b)
        x_power = np.sum(np.mean(x**2, axis=0))

        # Vectorized version for 1x1 CT (scalar case)
        CC = CT[0, 0] ** 2 * firing_rates[:, 0] + 1e-6  # shape (n_timepoints,)
        invCC = 1.0 / CC
        invCC[invCC > 1e6] = 0.0
        invFI = np.mean(invCC)

        # Compute expected SNR using the actual formula
        expected_snr_linear = x_power / invFI
        expected_snr_dB = 10 * np.log10(expected_snr_linear)

        # Check that the computed SNR matches the expected value
        assert np.abs(snr_dB - expected_snr_dB) < 1e-10, (
            f"SNR mismatch: computed={snr_dB:.6f}, expected={expected_snr_dB:.6f}"
        )


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n_timepoints = 100
    d_latent = 5
    d_neurons = 50
    x = np.random.randn(n_timepoints, d_latent)
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    CT = np.random.randn(d_latent, d_neurons)
    b = np.random.randn(1, d_neurons)
    return x, CT, b

def test_snr_functions_equivalence(sample_data):
    """Test that the vectorized and original SNR functions produce nearly identical results."""
    x, CT, b = sample_data
    snr_original = SNR_bound_instantaneous(x, CT, b)
    snr_vectorized = SNR_bound_instantaneous_vectorized(x, CT, b)
    assert np.isclose(snr_original, snr_vectorized, atol=1e-9), \
        f"SNR values are not close. Original: {snr_original}, Vectorized: {snr_vectorized}"

def test_snr_edge_cases():
    """Test edge cases for the SNR functions."""
    # Test with single timepoint
    x = np.random.randn(1, 5)
    CT = np.random.randn(5, 10)
    b = np.random.randn(1, 10)
    assert np.isclose(SNR_bound_instantaneous(x, CT, b), SNR_bound_instantaneous_vectorized(x, CT, b), atol=1e-9)

    # Test with single neuron
    x = np.random.randn(100, 5)
    CT = np.random.randn(5, 1)
    b = np.random.randn(1, 1)
    assert np.isclose(SNR_bound_instantaneous(x, CT, b), SNR_bound_instantaneous_vectorized(x, CT, b), atol=1e-9)

    # Test with single latent dimension
    x = np.random.randn(100, 1)
    CT = np.random.randn(1, 10)
    b = np.random.randn(1, 10)
    assert np.isclose(SNR_bound_instantaneous(x, CT, b), SNR_bound_instantaneous_vectorized(x, CT, b), atol=1e-9)
