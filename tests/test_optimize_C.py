"""Unit tests for the optimize_C function in neurofisher.optimize module."""

import numpy as np
import pytest

from neurofisherSNR.optimize import initialize_C, optimize_C
from neurofisherSNR.snr import SNR_bound_instantaneous
from neurofisherSNR.utils import compute_firing_rate


class TestOptimizeCDimensions:
    """Test class for checking input and output dimensionality of optimize_C function."""

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
        self.C = initialize_C(self.d_latent, self.d_neurons, p_coh=0.5, p_sparse=0.0)
        self.b = np.random.randn(1, self.d_neurons)

    def test_optimize_C_input_dimensions(self):
        """Test that optimize_C correctly handles input dimensions."""
        # Test with correct dimensions
        scaled_C, updated_b, achieved_snr = optimize_C(
            x=self.x,
            C=self.C,
            b=self.b,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=self.max_rate_per_bin,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            verbose=False,
        )

        # Check input dimensions are preserved
        assert self.x.shape == (
            self.time_steps,
            self.d_latent,
        ), f"Expected x shape {(self.time_steps, self.d_latent)}, got {self.x.shape}"
        assert self.C.shape == (
            self.d_neurons,
            self.d_latent,
        ), f"Expected C shape {(self.d_neurons, self.d_latent)}, got {self.C.shape}"
        assert self.b.shape == (
            1,
            self.d_neurons,
        ), f"Expected b shape {(1, self.d_neurons)}, got {self.b.shape}"

    def test_optimize_C_output_dimensions(self):
        """Test that optimize_C returns outputs with correct dimensions."""
        scaled_C, updated_b, achieved_snr = optimize_C(
            x=self.x,
            C=self.C,
            b=self.b,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=self.max_rate_per_bin,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            verbose=False,
        )

        # Check output dimensions
        assert scaled_C.shape == (
            self.d_neurons,
            self.d_latent,
        ), f"Expected scaled_C shape {(self.d_neurons, self.d_latent)}, got {scaled_C.shape}"
        assert updated_b.shape == (
            1,
            self.d_neurons,
        ), f"Expected updated_b shape {(1, self.d_neurons)}, got {updated_b.shape}"
        assert np.isscalar(
            achieved_snr
        ), f"Expected achieved_snr to be scalar, got shape {np.array(achieved_snr).shape}"
        assert isinstance(
            achieved_snr, (int, float)
        ), f"Expected achieved_snr to be numeric, got type {type(achieved_snr)}"

    def test_optimize_C_different_dimensions(self):
        """Test optimize_C with different input dimensions."""
        # Test with different dimensions
        time_steps_2 = 200
        d_latent_2 = 5
        d_neurons_2 = 15

        x_2 = np.random.randn(time_steps_2, d_latent_2)
        C_2 = initialize_C(d_latent_2, d_neurons_2, p_coh=0.3, p_sparse=0.1)
        b_2 = np.random.randn(d_neurons_2)

        scaled_C_2, updated_b_2, achieved_snr_2 = optimize_C(
            x=x_2,
            C=C_2,
            b=b_2,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=self.max_rate_per_bin,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            verbose=False,
        )

        # Check output dimensions match input dimensions
        assert scaled_C_2.shape == (
            d_neurons_2,
            d_latent_2,
        ), f"Expected scaled_C_2 shape {(d_neurons_2, d_latent_2)}, got {scaled_C_2.shape}"
        assert updated_b_2.shape == (
            1,
            d_neurons_2,
        ), f"Expected updated_b_2 shape {(1, d_neurons_2)}, got {updated_b_2.shape}"
        assert np.isscalar(
            achieved_snr_2
        ), f"Expected achieved_snr_2 to be scalar, got shape {np.array(achieved_snr_2).shape}"

    def test_optimize_C_dimension_mismatch_errors(self):
        """Test that optimize_C raises appropriate errors for dimension mismatches."""
        # Test with mismatched C and b dimensions
        # Wrong number of neurons
        C_wrong = np.random.randn(self.d_latent, self.d_neurons + 1)
        b_wrong = np.random.randn(self.d_neurons)  # Correct size

        with pytest.raises(Exception):  # Should raise some kind of error
            optimize_C(
                x=self.x,
                C=C_wrong,
                b=b_wrong,
                tgt_rate_per_bin=self.tgt_rate_per_bin,
                max_rate_per_bin=self.max_rate_per_bin,
                tgt_snr=self.tgt_snr,
                snr_fn=SNR_bound_instantaneous,
                verbose=False,
            )

        # Test with mismatched x and C dimensions
        # Wrong latent dimension
        x_wrong = np.random.randn(self.time_steps, self.d_latent + 1)
        C_correct = np.random.randn(self.d_latent, self.d_neurons)  # Correct size

        with pytest.raises(Exception):  # Should raise some kind of error
            optimize_C(
                x=x_wrong,
                C=C_correct,
                b=self.b,
                tgt_rate_per_bin=self.tgt_rate_per_bin,
                max_rate_per_bin=self.max_rate_per_bin,
                tgt_snr=self.tgt_snr,
                snr_fn=SNR_bound_instantaneous,
                verbose=False,
            )

    def test_optimize_C_output_types(self):
        """Test that optimize_C returns outputs of correct types."""
        scaled_C, updated_b, achieved_snr = optimize_C(
            x=self.x,
            C=self.C,
            b=self.b,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=self.max_rate_per_bin,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            verbose=False,
        )

        # Check output types
        assert isinstance(
            scaled_C, np.ndarray
        ), f"Expected scaled_C to be numpy array, got {type(scaled_C)}"
        assert isinstance(
            updated_b, np.ndarray
        ), f"Expected updated_b to be numpy array, got {type(updated_b)}"
        assert isinstance(
            achieved_snr, (int, float)
        ), f"Expected achieved_snr to be numeric, got {type(achieved_snr)}"

        # Check data types
        assert scaled_C.dtype in [
            np.float32,
            np.float64,
        ], f"Expected scaled_C to be float, got {scaled_C.dtype}"
        assert updated_b.dtype in [
            np.float32,
            np.float64,
        ], f"Expected updated_b to be float, got {updated_b.dtype}"

    def test_optimize_C_output_values(self):
        """Test that optimize_C returns reasonable output values."""
        scaled_C, updated_b, achieved_snr = optimize_C(
            x=self.x,
            C=self.C,
            b=self.b,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=self.max_rate_per_bin,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            verbose=False,
        )

        # Check that outputs are finite
        assert np.all(np.isfinite(scaled_C)), "scaled_C contains non-finite values"
        assert np.all(np.isfinite(updated_b)), "updated_b contains non-finite values"
        assert np.isfinite(achieved_snr), "achieved_snr is not finite"

        # Check that SNR is reasonable (should be positive)
        assert achieved_snr > -np.inf, "achieved_snr should be greater than -inf"

        # Check that scaled_C has the same shape as input C
        assert (
            scaled_C.shape == self.C.shape
        ), f"scaled_C shape {scaled_C.shape} should match input C shape {self.C.shape}"

        # Check that updated_b has the same shape as input b
        assert (
            updated_b.shape == self.b.shape
        ), f"updated_b shape {updated_b.shape} should match input b shape {self.b.shape}"


    def test_optimize_C_priority_max_returns_consistent_rates_and_snr(self):
        """Returned C, b, and SNR should describe the same capped model."""
        scaled_C, updated_b, achieved_snr = optimize_C(
            x=self.x,
            C=self.C,
            b=self.b,
            tgt_rate_per_bin=self.tgt_rate_per_bin,
            max_rate_per_bin=0.25,
            tgt_snr=self.tgt_snr,
            snr_fn=SNR_bound_instantaneous,
            priority="max",
            min_gain=0.01,
            tol=1e-4,
            verbose=False,
        )

        rates = compute_firing_rate(self.x, scaled_C.T, updated_b)
        recomputed_snr = SNR_bound_instantaneous(self.x, scaled_C.T, updated_b)

        np.testing.assert_allclose(
            rates.mean(axis=0),
            self.tgt_rate_per_bin,
            rtol=1e-6,
            atol=1e-8,
        )
        assert float(rates.max()) <= 0.25 + 1e-8
        assert recomputed_snr == pytest.approx(achieved_snr, abs=1e-8)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
