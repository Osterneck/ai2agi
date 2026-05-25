"""
============================================================
DTB — Digital Twin Brain  |  Component 4 of 5
ai2agi  ·  Ai70000, Ltd.  ·  Alex Osterneck, CLA, MSCS
============================================================
FILE II: dtb_model.py
PURPOSE: TensorFlow-based DTB neural model.
         Implements the biophysical ODE system as a
         stateful TF computation graph. Each call to
         step() advances the simulation by one discrete
         time step dt (Euler integration).

EQUATIONS IMPLEMENTED (Genesis, pp. 35–59):

  Membrane potential update:
    dVi/dt = ( -gL_i*(Vi - EL) + Isum ) / Ci
    Vi_new = Vi + dt * dVi_dt

  Synaptic current sum:
    Isum = Σ_u [ ji_u * (Vi - Eu) * gi_u ] + Iext

  Channel dynamics:
    dji_u/dt = (-1/Tu)*ji_u + wu*Σ(Wij*spikes_j(t-tkm))
    ji_u_new = ji_u + dt * dji_u_dt
    [spike influence simplified to wu*exp(-dt) for
     computational tractability at current scale]

NEUROTRANSMITTER REVERSAL POTENTIALS (Eu):
    AMPA:  -70.0 mV  (fast excitatory)
    NMDA:    0.0 mV  (slow excitatory)
    GABAa: -80.0 mV  (fast inhibitory)
    GABAb: -90.0 mV  (slow inhibitory)

OUTPUT CONTRACT (to dtb_simulator.py / CV_Adapt):
    step() returns dict:
      'membrane_potential'  : np.ndarray (n_neurons,)
      'channel_states'      : np.ndarray (n_neurons, n_synapses)
      'synaptic_currents'   : np.ndarray (n_neurons,)
      'total_current'       : np.ndarray (n_neurons,)
      'dVi_dt'              : np.ndarray (n_neurons,)
      'dji_u_dt'            : np.ndarray (n_neurons, n_synapses)

    to_output_tensor() returns:
      tf.Tensor shape (n_neurons, 4) — [Vi, Isum, mean_ji, mean_gi]
      This is the raw DTB tensor input to CV_Adapt.
============================================================
"""

import numpy as np
import tensorflow as tf
from typing import Optional


# Reversal potentials for [AMPA, NMDA, GABAa, GABAb]
EU_REVERSAL = [-70.0, 0.0, -80.0, -90.0]


class DTBModel:
    """
    Stateful TensorFlow Digital Twin Brain model.

    Maintains the full neuronal state across simulation
    steps via tf.Variable. The state persists between
    calls to step() — this is intentional: the DTB is
    a continuous-time dynamical system.

    Parameters
    ----------
    num_neurons : int
        Simulated neuron count. Default: 100.
        Scale target for full DTB: 1e9 (billion).
        Current GPU-feasible range: 1e3 – 1e5.
    num_synapses : int
        Synapse types per neuron. Set to 4 to match
        NT count (AMPA, NMDA, GABAa, GABAb).
        Default: 4.
    dt : float
        Euler integration time step (seconds).
        Default: 0.01 (10 ms).
        For higher temporal resolution: 0.001 (1 ms).
    seed : int, optional
        TF random seed for reproducibility.
    """

    # ── Biophysical constants ─────────────────────────────
    # Fixed across all neurons in this population model.
    # In full-scale DTB, these become per-neuron tensors
    # drawn from Allen Brain Atlas morphology data.
    _Ci_DEFAULT   =  1.0    # Neuronal capacitance (normalized)
    _gL_DEFAULT   =  0.1    # Leak conductance (normalized)
    _EL_DEFAULT   = -65.0   # Leak equilibrium potential (mV)
    _Tu_DEFAULT   =  5.0    # Channel decay constant (ms)

    def __init__(
        self,
        num_neurons: int = 100,
        num_synapses: int = 4,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ):
        self.num_neurons  = num_neurons
        self.num_synapses = num_synapses
        self.dt           = dt
        self._step_count  = 0

        if seed is not None:
            tf.random.set_seed(seed)

        self._initialize_model()

    # ── Initialization ────────────────────────────────────

    def _initialize_model(self) -> None:
        """
        Initialize all TF Variables and Constants.

        Variables (learnable / updateable state):
          Vi      — membrane potential per neuron
          ji_u    — channel open probability
          gi_u    — synaptic conductance
          Iext    — external current
          wu      — synaptic weights

        Constants (fixed biophysical parameters):
          Ci, gL_i, EL, Eu, Tu
        """
        try:
            N, S = self.num_neurons, self.num_synapses

            # ── State variables ───────────────────────────
            self.Vi = tf.Variable(
                tf.random.uniform([N], minval=-70.0, maxval=-50.0,
                                  dtype=tf.float32),
                name="Vi",
                trainable=False,
            )
            self.ji_u = tf.Variable(
                tf.zeros([N, S], dtype=tf.float32),
                name="ji_u",
                trainable=False,
            )

            # ── Synaptic conductances (learnable) ─────────
            self.gi_u = tf.Variable(
                tf.random.uniform([N, S], dtype=tf.float32),
                name="gi_u",
                trainable=True,
            )

            # ── External drive (set per step if needed) ───
            self.Iext = tf.Variable(
                tf.zeros([N], dtype=tf.float32),
                name="Iext",
                trainable=False,
            )

            # ── Synaptic weights ──────────────────────────
            self.wu = tf.Variable(
                tf.random.uniform([S], dtype=tf.float32),
                name="wu",
                trainable=True,
            )

            # ── Biophysical constants ─────────────────────
            self.Ci   = tf.constant(self._Ci_DEFAULT,  dtype=tf.float32, name="Ci")
            self.gL_i = tf.constant(self._gL_DEFAULT,  dtype=tf.float32, name="gL_i")
            self.EL   = tf.constant(self._EL_DEFAULT,  dtype=tf.float32, name="EL")
            self.Eu   = tf.constant(EU_REVERSAL,        dtype=tf.float32, name="Eu")
            self.Tu   = tf.constant(self._Tu_DEFAULT,  dtype=tf.float32, name="Tu")

        except Exception as e:
            raise RuntimeError(f"DTBModel._initialize_model() failed: {e}") from e

    # ── Core simulation step ──────────────────────────────

    @tf.function
    def _compute_step(self):
        """
        Compiled TF graph for one Euler integration step.
        Returns all intermediate values for diagnostics.
        """
        # Driving force: (Vi - Eu) for each NT
        # Vi_expanded shape: (N, 1), Eu shape: (4,)
        # drive_force shape: (N, 4)
        Vi_expanded = tf.expand_dims(self.Vi, axis=1)
        drive_force = Vi_expanded - self.Eu

        # Synaptic current: Σ_u [ ji_u * (Vi - Eu) * gi_u ]
        # shape: (N, 4) → reduce over synapses → (N,)
        synaptic_currents = tf.reduce_sum(
            self.ji_u * drive_force * self.gi_u, axis=1
        )

        # Total current: Isum = synaptic + external
        Isum = synaptic_currents + self.Iext

        # Membrane potential update (Euler)
        dVi_dt = (-self.gL_i * (self.Vi - self.EL) + Isum) / self.Ci
        Vi_new = self.Vi + self.dt * dVi_dt

        # Channel dynamics update
        # Spike influence simplified: wu * exp(-dt)
        # Full form: wu * Σ(Wij * spikes_j(t-tkm))
        spike_influence = self.wu * tf.exp(-self.dt)
        dji_u_dt = (-1.0 / self.Tu) * self.ji_u + spike_influence
        ji_u_new = self.ji_u + self.dt * dji_u_dt

        return Vi_new, ji_u_new, dVi_dt, dji_u_dt, synaptic_currents, Isum

    def step(self, Iext_override: Optional[np.ndarray] = None) -> dict:
        """
        Advance DTB simulation by one time step (dt seconds).

        Parameters
        ----------
        Iext_override : np.ndarray, optional
            External current injection (A) per neuron.
            shape: (num_neurons,). If provided, overrides
            the current self.Iext state.
            Use for sensory input encoding from Component 4
            sensory simulation layer.

        Returns
        -------
        dict with keys:
            'step'               : int step number
            'membrane_potential' : np.ndarray (N,)    [mV]
            'channel_states'     : np.ndarray (N, S)
            'synaptic_currents'  : np.ndarray (N,)    [A]
            'total_current'      : np.ndarray (N,)    [A]
            'dVi_dt'             : np.ndarray (N,)
            'dji_u_dt'           : np.ndarray (N, S)
        """
        try:
            if Iext_override is not None:
                Iext_tf = tf.cast(
                    tf.convert_to_tensor(Iext_override), tf.float32
                )
                self.Iext.assign(Iext_tf)

            Vi_new, ji_u_new, dVi_dt, dji_u_dt, I_syn, Isum = (
                self._compute_step()
            )

            # Commit state updates
            self.Vi.assign(Vi_new)
            self.ji_u.assign(ji_u_new)
            self._step_count += 1

            return {
                "step":               self._step_count,
                "membrane_potential": Vi_new.numpy(),
                "channel_states":     ji_u_new.numpy(),
                "synaptic_currents":  I_syn.numpy(),
                "total_current":      Isum.numpy(),
                "dVi_dt":             dVi_dt.numpy(),
                "dji_u_dt":           dji_u_dt.numpy(),
            }

        except Exception as e:
            raise RuntimeError(
                f"DTBModel.step() failed at step {self._step_count}: {e}"
            ) from e

    # ── CV_Adapt output tensor ────────────────────────────

    def to_output_tensor(self) -> tf.Tensor:
        """
        Produce the standardized DTB output tensor for CV_Adapt.

        Returns a (num_neurons, 4) tf.float32 tensor encoding:
          col 0: membrane potential   Vi       (mV)
          col 1: total current        Isum     (A)
          col 2: mean channel state   mean(ji_u per neuron)
          col 3: mean conductance     mean(gi_u per neuron)

        CV_Adapt ingests this as T_DTB in:
          T_AGI = CV_Adapt(T_G, T_R, T_LLM, T_DTB)
        """
        Vi_expanded = tf.expand_dims(self.Vi, axis=1)
        drive_force = Vi_expanded - self.Eu
        I_syn = tf.reduce_sum(self.ji_u * drive_force * self.gi_u, axis=1)
        Isum  = I_syn + self.Iext

        mean_ji = tf.reduce_mean(self.ji_u, axis=1)
        mean_gi = tf.reduce_mean(self.gi_u, axis=1)

        return tf.stack([self.Vi, Isum, mean_ji, mean_gi], axis=1)

    # ── State inspection ──────────────────────────────────

    def get_state_summary(self) -> dict:
        """Returns scalar summary statistics of current state."""
        return {
            "step":           self._step_count,
            "Vi_mean_mV":     float(tf.reduce_mean(self.Vi).numpy()),
            "Vi_std_mV":      float(tf.math.reduce_std(self.Vi).numpy()),
            "Vi_min_mV":      float(tf.reduce_min(self.Vi).numpy()),
            "Vi_max_mV":      float(tf.reduce_max(self.Vi).numpy()),
            "ji_u_mean":      float(tf.reduce_mean(self.ji_u).numpy()),
            "Iext_mean_A":    float(tf.reduce_mean(self.Iext).numpy()),
            "wu_mean":        float(tf.reduce_mean(self.wu).numpy()),
        }

    def reset(self) -> None:
        """Reset all state variables to initial conditions."""
        self._initialize_model()
        self._step_count = 0

    def set_external_current(self, Iext: np.ndarray) -> None:
        """
        Inject external current (sensory or experimental).
        shape: (num_neurons,), dtype: float32-compatible.
        """
        self.Iext.assign(
            tf.cast(tf.convert_to_tensor(Iext), tf.float32)
        )

    def get_output_tensor_spec(self) -> dict:
        """CV_Adapt handshake spec for T_DTB."""
        return {
            "shape":   (self.num_neurons, 4),
            "dtype":   "tf.float32",
            "columns": [
                "membrane_potential_Vi_mV",
                "total_current_Isum_A",
                "mean_channel_open_probability",
                "mean_synaptic_conductance",
            ],
            "component": "DTB",
            "feeds_into": "CV_Adapt",
        }


# ── Module self-test ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("DTB Component 4 — File II: DTBModel self-test")
    print("=" * 60)

    model = DTBModel(num_neurons=100, num_synapses=4, dt=0.01, seed=42)

    print(f"Initialized: {model.num_neurons} neurons, "
          f"{model.num_synapses} synapses, dt={model.dt}s")
    print(f"Initial state: {model.get_state_summary()}")
    print()

    # Run 10 simulation steps
    for i in range(10):
        result = model.step()
        if i == 0 or i == 9:
            print(f"Step {result['step']:3d} | "
                  f"Vi[0:5]={result['membrane_potential'][:5].round(3)} mV | "
                  f"Isum[0:5]={result['total_current'][:5].round(6)} A")

    print()
    print(f"Post-10-step state: {model.get_state_summary()}")
    print()

    # CV_Adapt output tensor
    T_DTB = model.to_output_tensor()
    print(f"T_DTB shape  : {T_DTB.shape}")
    print(f"T_DTB dtype  : {T_DTB.dtype}")
    print(f"T_DTB[0:3]   :\n{T_DTB[:3].numpy().round(4)}")
    print()
    print("Output tensor spec (CV_Adapt contract):")
    for k, v in model.get_output_tensor_spec().items():
        print(f"  {k}: {v}")
    print()
    print("DTBModel self-test PASSED.")
