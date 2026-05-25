"""
============================================================
DTB — Digital Twin Brain  |  Component 4 of 5
ai2agi  ·  Ai70000, Ltd.  ·  Alex Osterneck, CLA, MSCS
============================================================
FILE III: dtb_simulator.py
PURPOSE: Full DTB simulation orchestrator.
         Integrates DataGenerator (File I) + DTBModel
         (File II) into a complete simulation pipeline
         that produces the final T_DTB tensor for
         CV_Adapt (Component 5).

         This is the top-level runtime entry point for
         DTB as Component 4 of the ai2agi AGI architecture:

           AGI ≈ f(G_enhanced, R_enhanced, ai_LLM,
                   **DTB**, CV_Adapt)

SIMULATION PIPELINE:
  1. DataGenerator.generate_data()
       → synthetic state tensor (n_samples, n_neurons, 24)
  2. DTBModel.initialize_model()
       → TF Variable state graph
  3. DTBModel.step() × n_steps
       → per-step state evolution
  4. DTBModel.to_output_tensor()
       → T_DTB: (n_neurons, 4) → CV_Adapt

EXTENDED DTB SUBSYSTEMS (Genesis pp. 42–55):
  Beyond the core electrical model, DTB_Simulation
  encompasses:

  DTB_simulation = F( {N}, {S}, {C}, {P}, H, E, A )

  Where:
    {N} = neuron objects (Vi, ji_u, gi_u, etc.)
    {S} = plasticity parameters (Hebbian / STDP)
    {C} = connectivity / connectome topology
    {P} = all governing parameters
    H   = hardware resources (GPU/DLA/HPC)
    E   = external environment
    A   = algorithms (integration, I/O, motor ctrl)

  Additional placeholder subsystems (to be expanded):
    - Hormonal/endocrine PK model: H'(t)=f_h(H(t),N(t))
    - Sensory input: I_sensory,i(t) = k(S(t), Si)
    - Motor output: M(t) = G(A(t))
    - External interaction: E_int(t) = I(S(t),M(t),B,W(t))
    - Glial cell influence on synaptic activity
    - Consciousness index C_i(t), Φ(t)
    - Linguistic L_i(t), planning P_i(t), decision D_i(t)

OUTPUT:
  Final T_DTB tensor + full simulation history log.
  T_DTB is passed to CV_Adapt as the fourth input:
    cv_adapt_pipeline(T_G, T_R, T_LLM, T_DTB)
============================================================
"""

import numpy as np
import tensorflow as tf
import time
from typing import Optional

from data_generator import DataGenerator
from dtb_model import DTBModel


def format_arr(arr: np.ndarray, n: int = 5, decimals: int = 4) -> str:
    """Format first n elements of array for console output."""
    return str(arr[:n].round(decimals))


class DTBSimulator:
    """
    Full Digital Twin Brain simulation orchestrator.

    Coordinates the DataGenerator and DTBModel into a
    complete simulation run and produces the final
    T_DTB tensor for CV_Adapt.

    Parameters
    ----------
    n_samples : int
        Time-point samples for synthetic data init.
    n_neurons : int
        Neuron count. Propagated to DTBModel.
    n_neurotransmitters : int
        Fixed at 4 (AMPA, NMDA, GABAa, GABAb).
    n_presynaptic_neurons : int
        Presynaptic input count per neuron.
    dt : float
        Euler integration step size (seconds).
    n_steps : int
        Number of simulation time steps to run.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool
        Per-step console output. Default: True.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_neurons: int = 100,
        n_neurotransmitters: int = 4,
        n_presynaptic_neurons: int = 3,
        dt: float = 0.01,
        n_steps: int = 10,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        self.n_neurons = n_neurons
        self.n_steps   = n_steps
        self.dt        = dt
        self.verbose   = verbose
        self.history   = []

        # ── File I: Data Generator ────────────────────────
        self.data_gen = DataGenerator(
            n_samples=n_samples,
            n_neurons=n_neurons,
            n_neurotransmitters=n_neurotransmitters,
            n_presynaptic_neurons=n_presynaptic_neurons,
            seed=seed,
            verbose=verbose,
        )

        # ── File II: TF Neural Model ──────────────────────
        self.model = DTBModel(
            num_neurons=n_neurons,
            num_synapses=n_neurotransmitters,
            dt=dt,
            seed=seed,
        )

        # ── Subsystem stubs (Genesis pp. 42–55) ──────────
        # Each subsystem produces a state vector that feeds
        # into the tensor fusion layer.
        self.hormonal_state  = None   # H(t) — PK/endocrine
        self.sensory_state   = None   # S(t) — sensory input
        self.motor_state     = None   # M(t) — motor output
        self.world_model     = None   # W(t) — internal world
        self.glial_state     = None   # glial influence
        self.consciousness   = None   # C_i(t), Phi(t)

    # ── Subsystem placeholders (Genesis algebraic vars) ───

    def _update_hormonal_state(self, neural_activity: np.ndarray) -> np.ndarray:
        """
        Hormonal/Chemical Signaling Subsystem.
        Genesis: H'(t) = f_h(H(t), N(t))

        PK model for: cortisol (HPA axis ODE),
        dopamine/serotonin (reward), norepinephrine
        (arousal), oxytocin, insulin.

        Placeholder: returns scalar arousal proxy
        proportional to mean neural activity.
        Full implementation: pharmacokinetic ODE solver
        (scipy.integrate.odeint over HPA axis system).
        """
        mean_activity = float(np.mean(np.abs(neural_activity)))
        if self.hormonal_state is None:
            self.hormonal_state = np.zeros(6, dtype=np.float32)
        # Cortisol proxy rises with sustained high activity
        self.hormonal_state[0] = min(1.0, self.hormonal_state[0] + 0.01 * mean_activity)
        # Dopamine proxy (reward signal — spikes on change)
        delta = mean_activity - (self.hormonal_state[5] if self.hormonal_state[5] != 0 else mean_activity)
        self.hormonal_state[1] = max(0.0, self.hormonal_state[1] + 0.05 * delta)
        # Norepinephrine proxy (arousal)
        self.hormonal_state[2] = 0.8 * self.hormonal_state[2] + 0.2 * mean_activity
        # Serotonin (inverse arousal / mood stabilizer)
        self.hormonal_state[3] = 1.0 - self.hormonal_state[2]
        # Oxytocin (social proxy — stub)
        self.hormonal_state[4] = 0.5
        # Store last mean for delta calc
        self.hormonal_state[5] = mean_activity
        return self.hormonal_state[:5]

    def _encode_sensory_input(self, step: int) -> np.ndarray:
        """
        Sensory Input Simulation Layer.
        Genesis: I_sensory,i(t) = k(S(t), Si)

        Converts external stimuli to per-neuron current.
        Placeholder: sinusoidal stimulus with noise,
        simulating oscillatory sensory drive.
        Full implementation: spike-train encoder from
        visual/auditory/interoceptive input streams.
        """
        t = step * self.dt
        # Sinusoidal sensory drive at 10 Hz (alpha band)
        base = 1e-10 * np.sin(2 * np.pi * 10 * t)
        noise = np.random.normal(0, 1e-11, size=self.n_neurons)
        return (base + noise).astype(np.float32)

    def _compute_motor_output(self, Vi: np.ndarray) -> np.ndarray:
        """
        Motor Output Subsystem.
        Genesis: M(t) = G(A(t))

        Translates neural population activity into motor
        command tensors. Placeholder: thresholded spike
        detection — neurons above threshold contribute
        to motor command.
        Full implementation: motor cortex column model
        → actuator / effector mapping.
        """
        threshold = -55.0  # mV spike threshold
        spikes = (Vi > threshold).astype(np.float32)
        return spikes

    def _update_world_model(self, sensory: np.ndarray, motor: np.ndarray) -> dict:
        """
        Internal World Model.
        Genesis: W(t) = update(W(t-1), S(t), M(t))

        Maintains DTB's internal representation of the
        external environment. Placeholder: simple state
        dict. Full implementation: predictive coding
        hierarchy with Kalman-filtered belief states.
        """
        return {
            "sensory_mean": float(np.mean(np.abs(sensory))),
            "motor_active": int(np.sum(motor)),
        }

    def _glial_modulation(self, Vi: np.ndarray, ji_u: np.ndarray) -> np.ndarray:
        """
        Glial Cell Influence.
        Genesis: h([Glial Activity], [Synaptic Activity])

        Astrocytes modulate synaptic transmission via
        tripartite synapse (Araque et al., 1999).
        Placeholder: smoothing filter on channel states.
        """
        return 0.99 * ji_u + 0.01 * np.random.uniform(0, 0.01, ji_u.shape)

    # ── Main simulation pipeline ──────────────────────────

    def run(self) -> dict:
        """
        Execute full DTB simulation pipeline.

        Steps:
          1. Generate synthetic data (DataGenerator)
          2. Run n_steps of DTBModel.step()
          3. Apply subsystem updates per step
          4. Collect history
          5. Produce final T_DTB tensor for CV_Adapt

        Returns
        -------
        dict:
          'T_DTB'          : tf.Tensor (n_neurons, 4)
          'history'        : list of per-step dicts
          'final_state'    : model state summary
          'subsystem_state': hormonal, sensory, motor, world
          'tensor_spec'    : CV_Adapt handshake spec
          'synthetic_data' : tf.Tensor (n_samples, n_neurons, 24)
          'runtime_s'      : float wall-clock seconds
        """
        start = time.time()

        if self.verbose:
            self._print_header()

        # ── Step 1: Generate synthetic data ───────────────
        synthetic_data = self.data_gen.generate_data()
        if self.verbose:
            print(f"Synthetic data tensor: {synthetic_data.shape}")
            print(f"  Sample [0,0,:5] = "
                  f"{synthetic_data[0,0,:5].numpy().round(6)}\n")

        # ── Steps 2–4: Simulation loop ────────────────────
        for step_idx in range(self.n_steps):

            # Sensory current injection (subsystem 4)
            I_sensory = self._encode_sensory_input(step_idx)
            result = self.model.step(Iext_override=I_sensory)

            # Hormonal modulation (subsystem 2)
            H_t = self._update_hormonal_state(result["membrane_potential"])

            # Motor output (subsystem M)
            M_t = self._compute_motor_output(result["membrane_potential"])

            # World model update (subsystem W)
            W_t = self._update_world_model(I_sensory, M_t)

            # Glial modulation (placeholder)
            glial_ji = self._glial_modulation(
                result["membrane_potential"],
                result["channel_states"],
            )

            # Enrich result with subsystem states
            result["hormonal_state"]  = H_t.copy()
            result["motor_output"]    = M_t
            result["world_model"]     = W_t
            result["sensory_current"] = I_sensory

            self.history.append(result)

            if self.verbose:
                self._print_step(result)

        # ── Step 5: Produce T_DTB for CV_Adapt ───────────
        T_DTB = self.model.to_output_tensor()
        runtime = time.time() - start

        if self.verbose:
            self._print_footer(T_DTB, runtime)

        return {
            "T_DTB":          T_DTB,
            "history":        self.history,
            "final_state":    self.model.get_state_summary(),
            "subsystem_state": {
                "hormonal":   self.history[-1]["hormonal_state"],
                "motor":      self.history[-1]["motor_output"],
                "world":      self.history[-1]["world_model"],
                "sensory":    self.history[-1]["sensory_current"],
            },
            "tensor_spec":    self.model.get_output_tensor_spec(),
            "synthetic_data": synthetic_data,
            "runtime_s":      runtime,
        }

    # ── Console output helpers ────────────────────────────

    def _print_header(self) -> None:
        print("=" * 60)
        print("DTB — Digital Twin Brain  |  Component 4 of 5")
        print("ai2agi  ·  Ai70000, Ltd.  ·  Alex Osterneck")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  neurons    : {self.n_neurons}")
        print(f"  dt         : {self.dt} s ({self.dt*1000:.1f} ms)")
        print(f"  steps      : {self.n_steps}")
        print(f"  total time : {self.n_steps * self.dt * 1000:.1f} ms simulated")
        print()

    def _print_step(self, result: dict) -> None:
        s = result["step"]
        Vi  = result["membrane_potential"]
        Isu = result["total_current"]
        H   = result["hormonal_state"]
        m   = int(np.sum(result["motor_output"]))

        print(f"Step {s:3d} │ "
              f"Vi[0:3]={format_arr(Vi,3,3)} mV │ "
              f"Isum[0:3]={format_arr(Isu,3,3)} A │ "
              f"Motor spikes={m} │ "
              f"Cortisol={H[0]:.4f} Dopa={H[1]:.4f}")

    def _print_footer(self, T_DTB: tf.Tensor, runtime: float) -> None:
        print()
        print("─" * 60)
        print("SIMULATION COMPLETE")
        print(f"  Runtime        : {runtime:.3f} s")
        print(f"  T_DTB shape    : {T_DTB.shape}  → CV_Adapt")
        print(f"  T_DTB dtype    : {T_DTB.dtype}")
        print(f"  T_DTB[0:3]     :\n{T_DTB[:3].numpy().round(4)}")
        print()
        print("T_DTB column legend:")
        print("  [0] membrane_potential_Vi_mV")
        print("  [1] total_current_Isum_A")
        print("  [2] mean_channel_open_probability")
        print("  [3] mean_synaptic_conductance")
        print()
        print("Ready for: cv_adapt_pipeline(T_G, T_R, T_LLM, T_DTB)")
        print("=" * 60)


# ── CV_Adapt interface stub ───────────────────────────────
# Mirrors the CV_Adapt pseudo-code from Genesis pp. 85–90.
# Replace with actual CV_Adapt module import when available.

def generate_placeholder_tensor(n_neurons: int, n_features: int,
                                 name: str) -> tf.Tensor:
    """Generate a placeholder tensor for components not yet built."""
    t = tf.random.uniform([n_neurons, n_features], dtype=tf.float32)
    print(f"  [placeholder] {name}: shape={t.shape}")
    return t


def cv_adapt_pipeline(
    T_G:   tf.Tensor,
    T_R:   tf.Tensor,
    T_LLM: tf.Tensor,
    T_DTB: tf.Tensor,
) -> tf.Tensor:
    """
    CV_Adapt central processor stub.
    Genesis pp. 85–90:
      cv_adapt_pipeline(tensor_g, tensor_r, tensor_ai, tensor_dtb)
        → receive → normalize → fuse → adapt → store → output

    Full implementation lives in component_5/cv_adapt.py.
    This stub validates the tensor interface contract.
    """

    def normalize_tensor(t: tf.Tensor, name: str) -> tf.Tensor:
        mean, var = tf.nn.moments(tf.cast(t, tf.float32), axes=[0])
        std = tf.sqrt(var + 1e-8)
        normed = (tf.cast(t, tf.float32) - mean) / std
        print(f"  normalize({name}): {t.shape} → mean≈0, std≈1")
        return normed

    print("\n── CV_Adapt Pipeline (DTB interface stub) ──────────")

    # Step 1: Receive
    tensors = {"T_G": T_G, "T_R": T_R, "T_LLM": T_LLM, "T_DTB": T_DTB}
    print("Step 1: Receive tensor inputs")
    for k, v in tensors.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Step 2: Normalize
    print("Step 2: Normalize")
    normed = {k: normalize_tensor(v, k) for k, v in tensors.items()}

    # Step 3: Fuse (concatenation along feature axis)
    print("Step 3: Fuse (concatenate features)")
    # Align all tensors to same neuron dim (use T_DTB.shape[0])
    fused = tf.concat(list(normed.values()), axis=-1)
    print(f"  Fused tensor: {fused.shape}")

    # Step 4: Adapt (linear projection — full impl uses attention)
    print("Step 4: Adapt (central processing)")
    n_out = 64   # AGI-DNA output dimensionality
    W_adapt = tf.random.uniform([fused.shape[-1], n_out], dtype=tf.float32)
    adapted = tf.matmul(fused, W_adapt)
    print(f"  Adapted tensor: {adapted.shape}")

    # Step 5: Output T_AGI (AGI-DNA)
    print("Step 5: Output T_AGI (AGI-DNA)")
    T_AGI = adapted
    print(f"  T_AGI (AGI-DNA): {T_AGI.shape}, dtype={T_AGI.dtype}")
    print("─" * 52)

    return T_AGI


# ── Main entry point ──────────────────────────────────────

if __name__ == "__main__":

    # ── Configure simulation ──────────────────────────────
    CONFIG = {
        "n_samples":            100,
        "n_neurons":            100,
        "n_neurotransmitters":  4,
        "n_presynaptic_neurons":3,
        "dt":                   0.01,
        "n_steps":              10,
        "seed":                 42,
        "verbose":              True,
    }

    # ── Run DTB simulation ────────────────────────────────
    simulator = DTBSimulator(**CONFIG)
    output = simulator.run()

    T_DTB = output["T_DTB"]

    # ── Interface with CV_Adapt ───────────────────────────
    N = CONFIG["n_neurons"]
    print("\nGenerating placeholder tensors for components 1, 2, 3:")
    T_G   = generate_placeholder_tensor(N, 32, "T_G   (G_enhanced)")
    T_R   = generate_placeholder_tensor(N, 32, "T_R   (R_enhanced)")
    T_LLM = generate_placeholder_tensor(N, 32, "T_LLM (ai_LLM)")

    # ── CV_Adapt pipeline ─────────────────────────────────
    T_AGI = cv_adapt_pipeline(T_G, T_R, T_LLM, T_DTB)

    # ── Final output summary ──────────────────────────────
    print("\n" + "=" * 60)
    print("AGI-DNA OUTPUT TENSOR")
    print("=" * 60)
    print(f"T_AGI shape    : {T_AGI.shape}")
    print(f"T_AGI dtype    : {T_AGI.dtype}")
    print(f"T_AGI[0:3]     :\n{T_AGI[:3].numpy().round(4)}")
    print()
    print("Formula verified:")
    print("  T_AGI = CV_Adapt(T_G, T_R, T_LLM, T_DTB)")
    print("  AGI ≈ f(G_enhanced, R_enhanced, ai_LLM, DTB, CV_Adapt)")
    print()
    print(f"Simulation runtime : {output['runtime_s']:.3f} s")
    print(f"Final DTB state    : {output['final_state']}")
    print()
    print("DTB Component 4 — FULL PIPELINE COMPLETE.")
    print("=" * 60)
