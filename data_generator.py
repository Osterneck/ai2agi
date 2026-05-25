"""
============================================================
DTB — Digital Twin Brain  |  Component 4 of 5
ai2agi  ·  Ai70000, Ltd.  ·  Alex Osterneck, CLA, MSCS
============================================================
FILE I: data_generator.py
PURPOSE: Synthetic neural data generator.
         Generates biophysically-parameterized tensors that
         approximate real neuronal dynamics across:
           - Membrane potential (Hodgkin-Huxley-derived)
           - Synaptic currents (AMPA, NMDA, GABAa, GABAb)
           - Channel dynamics (open probability, decay)
           - Presynaptic spike timing
         Output tensor feeds into dtb_model.py (File II)
         and dtb_simulator.py (File III), which output
         to CV_Adapt (Component 5).

NOTE ON SYNTHETIC DATA:
         Synthetic data is the bootstrap substrate for DTB.
         In production true-AGI, real neuroimaging / EEG /
         MEG / fMRI data replaces or augments synthetic
         data. The tensor SHAPE and DTYPE contract is
         identical in both modes — CV_Adapt is agnostic
         to the data source.

TENSOR OUTPUT CONTRACT (to CV_Adapt):
         shape  : (n_samples, n_neurons, n_features)
         dtype  : tf.float32
         n_features breakdown (per neuron, per sample):
           [0]      C        — capacitance (F)
           [1]      gL       — leak conductance (S)
           [2]      V        — membrane potential (mV)
           [3]      EL       — leak equilibrium potential (mV)
           [4:8]    Isyn     — synaptic currents per NT (A)
           [8]      Iext     — external current (A)
           [9:13]   ji_u     — channel open probability
           [13:17]  Tau_u    — decay time constant (s)
           [17:21]  omega_u  — synaptic weights
           [21:24]  tkm      — presynaptic spike timing (ms)

MATHEMATICAL FOUNDATION (Genesis, pp. 35–40):
    1. Ci * dVi/dt = -gL,i*(Vi - EL) + Isum
    2. Isum = Σ ji,u*(Vi - Eu)*gi,u + Iext
    3. dji,u/dt = -1/Tu * ji,u + ωu * Σ(t - tkm)

NEUROTRANSMITTER INDEX MAP:
    [0] AMPA   — fast excitatory  (Eu = -70.0 mV)
    [1] NMDA   — slow excitatory  (Eu =   0.0 mV)
    [2] GABAa  — fast inhibitory  (Eu = -80.0 mV)
    [3] GABAb  — slow inhibitory  (Eu = -90.0 mV)
============================================================
"""

import numpy as np
import tensorflow as tf
from typing import Optional


# ── Biophysical parameter bounds ──────────────────────────
# All ranges drawn from Gerstner et al. (2014),
# Dayan & Abbott (2001), Hodgkin & Huxley (1952).

PARAM_BOUNDS = {
    "C":       (1e-9,  1e-6),    # Capacitance (F): 1 pF – 1 nF
    "gL":      (1e-9,  1e-6),    # Leak conductance (S)
    "V":       (-70.0, -50.0),   # Resting membrane potential (mV)
    "EL":      (-80.0, -60.0),   # Leak equilibrium potential (mV)
    "Isyn":    (0.0,   1e-9),    # Synaptic current (A), non-negative
    "Iext":    (-1e-9, 1e-9),    # External current (A), bipolar
    "ji_u":    (0.0,   1.0),     # Channel open probability [0,1]
    "Tau_u":   (1e-3,  1e-2),    # Decay time constant (s): 1–10 ms
    "omega_u": (0.0,   1.0),     # Synaptic weight, normalized
    "tkm":     (0.0,   100.0),   # Presynaptic spike timing (ms)
}

# Reversal potentials for each neurotransmitter (mV)
REVERSAL_POTENTIALS = {
    "AMPA":  -70.0,
    "NMDA":    0.0,
    "GABAa": -80.0,
    "GABAb": -90.0,
}

NT_LABELS = ["AMPA", "NMDA", "GABAa", "GABAb"]


class DataGenerator:
    """
    Synthetic biophysical neural data generator for DTB.

    Generates a single concatenated tensor representing
    the full neuronal state space across n_samples time
    points, n_neurons neurons, and n_neurotransmitters
    neurotransmitter channels.

    Parameters
    ----------
    n_samples : int
        Number of time-point samples to generate.
        Default: 100
    n_neurons : int
        Number of simulated neurons in the population.
        Default: 5 (scale to hundreds/thousands for
        full-brain simulation with GPU cluster)
    n_neurotransmitters : int
        Number of NT channels. Fixed at 4 (AMPA, NMDA,
        GABAa, GABAb) per Genesis specification.
        Default: 4
    n_presynaptic_neurons : int
        Number of presynaptic inputs per neuron.
        Default: 3
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Print parameter range verification on generate.
        Default: True
    """

    N_NEUROTRANSMITTERS_EXPECTED = 4

    def __init__(
        self,
        n_samples: int = 100,
        n_neurons: int = 5,
        n_neurotransmitters: int = 4,
        n_presynaptic_neurons: int = 3,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        if n_neurotransmitters != self.N_NEUROTRANSMITTERS_EXPECTED:
            raise ValueError(
                f"DTB requires exactly {self.N_NEUROTRANSMITTERS_EXPECTED} "
                f"neurotransmitters (AMPA, NMDA, GABAa, GABAb). "
                f"Got: {n_neurotransmitters}"
            )

        self.n_samples            = n_samples
        self.n_neurons            = n_neurons
        self.n_neurotransmitters  = n_neurotransmitters
        self.n_presynaptic_neurons= n_presynaptic_neurons
        self.verbose              = verbose

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

    # ── Private helpers ───────────────────────────────────

    def _uniform(self, low: float, high: float, shape: tuple) -> np.ndarray:
        return np.random.uniform(low, high, size=shape).astype(np.float32)

    def _verify(self, arrays: dict) -> None:
        """Print biophysical range verification for each parameter."""
        print("\n── DTB DataGenerator: Parameter Range Verification ──")
        fmt = {
            "C":       ("F",  ".2e"),
            "gL":      ("S",  ".2e"),
            "V":       ("mV", ".2f"),
            "EL":      ("mV", ".2f"),
            "Isyn":    ("A",  ".2e"),
            "Iext":    ("A",  ".2e"),
            "ji_u":    ("",   ".2%"),
            "Tau_u":   ("s",  ".2e"),
            "omega_u": ("",   ".4f"),
            "tkm":     ("ms", ".2f"),
        }
        for name, arr in arrays.items():
            unit, fstr = fmt[name]
            lo = format(float(np.min(arr)), fstr)
            hi = format(float(np.max(arr)), fstr)
            print(f"  {name:10s}: [{lo}, {hi}] {unit}")
        print("─" * 52 + "\n")

    # ── Public API ────────────────────────────────────────

    def generate_data(self) -> tf.Tensor:
        """
        Generate the full synthetic neural state tensor.

        Returns
        -------
        tf.Tensor
            shape  : (n_samples, n_neurons, n_features)
            dtype  : tf.float32
            n_features = 1+1+1+1+4+1+4+4+4+3 = 24
              (C, gL, V, EL, Isyn×4, Iext, ji_u×4,
               Tau_u×4, omega_u×4, tkm×3)
        """
        try:
            S, N, NT, PRE = (
                self.n_samples,
                self.n_neurons,
                self.n_neurotransmitters,
                self.n_presynaptic_neurons,
            )

            # ── Neuronal membrane parameters ──────────────
            C   = self._uniform(*PARAM_BOUNDS["C"],   (S, N, 1))
            gL  = self._uniform(*PARAM_BOUNDS["gL"],  (S, N, 1))
            V   = self._uniform(*PARAM_BOUNDS["V"],   (S, N, 1))
            EL  = self._uniform(*PARAM_BOUNDS["EL"],  (S, N, 1))

            # ── Synaptic / external currents ──────────────
            Isyn = self._uniform(*PARAM_BOUNDS["Isyn"], (S, N, NT))
            Iext = self._uniform(*PARAM_BOUNDS["Iext"], (S, N, 1))

            # ── Channel dynamics ──────────────────────────
            ji_u    = self._uniform(*PARAM_BOUNDS["ji_u"],    (S, N, NT))
            Tau_u   = self._uniform(*PARAM_BOUNDS["Tau_u"],   (S, N, NT))
            omega_u = self._uniform(*PARAM_BOUNDS["omega_u"], (S, N, NT))

            # ── Presynaptic spike timing ──────────────────
            tkm = self._uniform(*PARAM_BOUNDS["tkm"], (S, N, PRE))

            if self.verbose:
                self._verify({
                    "C": C, "gL": gL, "V": V, "EL": EL,
                    "Isyn": Isyn, "Iext": Iext,
                    "ji_u": ji_u, "Tau_u": Tau_u,
                    "omega_u": omega_u, "tkm": tkm,
                })

            # ── Concatenate → unified state tensor ────────
            # shape: (S, N, 24)
            input_data = np.concatenate(
                [C, gL, V, EL, Isyn, Iext, ji_u, Tau_u, omega_u, tkm],
                axis=-1,
            )

            return tf.convert_to_tensor(input_data, dtype=tf.float32)

        except Exception as e:
            raise RuntimeError(f"DataGenerator.generate_data() failed: {e}") from e

    def get_feature_labels(self) -> list[str]:
        """
        Returns ordered feature label list matching tensor axis-2 positions.
        Use for downstream logging, visualization, and CV_Adapt annotation.
        """
        nt = NT_LABELS
        return (
            ["C", "gL", "V", "EL"]
            + [f"Isyn_{t}" for t in nt]
            + ["Iext"]
            + [f"ji_u_{t}"    for t in nt]
            + [f"Tau_u_{t}"   for t in nt]
            + [f"omega_u_{t}" for t in nt]
            + [f"tkm_{i}"     for i in range(self.n_presynaptic_neurons)]
        )

    def get_tensor_spec(self) -> dict:
        """
        Returns the tensor shape/dtype contract for CV_Adapt handshake.
        """
        n_features = (
            4                          # C, gL, V, EL
            + self.n_neurotransmitters # Isyn per NT
            + 1                        # Iext
            + self.n_neurotransmitters # ji_u per NT
            + self.n_neurotransmitters # Tau_u per NT
            + self.n_neurotransmitters # omega_u per NT
            + self.n_presynaptic_neurons # tkm
        )
        return {
            "shape":        (self.n_samples, self.n_neurons, n_features),
            "dtype":        "tf.float32",
            "n_features":   n_features,
            "feature_labels": self.get_feature_labels(),
            "nt_map":       REVERSAL_POTENTIALS,
            "source":       "synthetic",  # "real" in production AGI
        }


# ── Module self-test ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DTB Component 4 — File I: DataGenerator self-test")
    print("=" * 60)

    gen = DataGenerator(
        n_samples=100,
        n_neurons=5,
        n_neurotransmitters=4,
        n_presynaptic_neurons=3,
        seed=42,
        verbose=True,
    )

    tensor = gen.generate_data()

    print(f"Output tensor shape : {tensor.shape}")
    print(f"Output tensor dtype : {tensor.dtype}")
    print(f"First sample, first neuron, all features:")
    print(f"  {tensor[0, 0, :].numpy().round(6)}")
    print()
    print("Tensor spec (CV_Adapt contract):")
    for k, v in gen.get_tensor_spec().items():
        print(f"  {k}: {v}")
    print()
    print("DataGenerator self-test PASSED.")
