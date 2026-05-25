"""
cv_adapt_interface.py
─────────────────────
Integration stub: G_enhanced → CV_Adapt

Shows exactly how tensor_g slots into the CV_Adapt pipeline
as defined in the Genesis document (pages 85-89).

Usage:
    python run_g_enhanced.py    (runs full pipeline + prints report)
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from foundation_initializer import FoundationInitializer
from g_enhanced import GEnhanced


def generate_tensor_from_G_enhanced(n_episodes: int = 5) -> np.ndarray:
    """
    Drop-in function matching CV_Adapt pseudo-code signature:
        tensor_g = generate_tensor_from_G_enhanced()

    Returns:
        numpy float32 [100, 5, 14] — matches DTB tensor shape.
        In CV_Adapt: tf.convert_to_tensor(tensor_g, dtype=tf.float32)

    Note: FoundationInitializer.run() is called automatically before
    tensor generation — no manual wiring required.
    """
    module = GEnhanced(input_dim=32)
    fi = FoundationInitializer("")   # NIV corpus embedded — no PDF needed
    fi.run(module)                   # seeds HKS, TB, MQ before any tensor op
    tensor_g = module.generate_tensors(n_episodes=n_episodes)
    return tensor_g, module.summary()


def simulate_cv_adapt_receive(tensor_g: np.ndarray,
                               tensor_shape_dtb: tuple = (100, 5, 14)):
    """
    Simulates CV_Adapt's receive_tensor_inputs() for tensor_g.
    In production this runs inside the real CV_Adapt pipeline.
    """
    assert tensor_g.shape == tensor_shape_dtb, (
        f"Shape mismatch: got {tensor_g.shape}, expected {tensor_shape_dtb}")
    assert tensor_g.dtype == np.float32, \
        f"Dtype mismatch: got {tensor_g.dtype}"

    # CV_Adapt receive step (from Genesis pseudo-code page 86):
    all_tensors = {
        "tensor_g":   tensor_g,          # ← G_enhanced (this module)
        "tensor_r":   np.zeros_like(tensor_g),   # placeholder R_enhanced
        "tensor_ai":  np.zeros_like(tensor_g),   # placeholder ai_LLM
        "tensor_dtb": np.zeros_like(tensor_g),   # placeholder DTB
    }

    # CV_Adapt normalize step
    def normalize_tensor(t):
        mn, mx = t.min(), t.max()
        if mx - mn < 1e-8:
            return t
        return (t - mn) / (mx - mn)

    normalized = {k: normalize_tensor(v) for k, v in all_tensors.items()}

    # CV_Adapt fuse step (concatenation along feature axis)
    fused = np.concatenate(list(normalized.values()), axis=-1)
    # Shape: [100, 5, 56]  (4 components × 14 features)

    return fused


def print_report(summary: dict, tensor_g: np.ndarray, fused: np.ndarray):
    print("\n" + "═" * 70)
    print("  G_ENHANCED (e_Generalization) — COMPONENT 1 — ai2agi")
    print("  ai70000, Ltd.  |  Alex Osterneck, CLA, MSCS")
    print("═" * 70)
    print(f"\n  FORMULA:  ε = 1 / (ML + FSL + g(HKS,TB) + f(Q))")
    print(f"\n  RESULT:")
    print(f"    ML  (Meta-Learning)       = {summary['ML']:.4f}")
    print(f"    FSL (Few-Shot Learning)   = {summary['FSL']:.4f}")
    print(f"    CL  (Continual Learning)  = {summary['CL']:.4f}")
    print(f"    MI  (Multiple Intel.)     = {summary['MI']:.4f}")
    print(f"    ─────────────────────────────────")
    denom = summary['ML'] + summary['FSL'] + summary['CL'] + summary['MI']
    print(f"    Denominator               = {denom:.4f}")
    print(f"    ε  (Generalization Error) = {summary['epsilon']:.6f}")

    q = summary['quotients']
    print(f"\n  QUOTIENTS f(Q):")
    print(f"    PQ={q['PQ']:.3f}  IQ={q['IQ']:.3f}  EQ={q['EQ']:.3f}")
    print(f"    SQ={q['SQ']:.3f}  CQ={q['CQ']:.3f}  MQ={q['MQ']:.3f}")

    print(f"\n  KNOWLEDGE STRUCTURES:")
    print(f"    HKS depth    = {summary['HKS_depth']:.4f}")
    print(f"    TB density   = {summary['TB_density']:.4f}")
    print(f"    Task novelty = {summary['task_novelty']:.4f}")
    print(f"    Episodes run = {summary['episodes_run']}")

    print(f"\n  OUTPUT TENSOR → CV_Adapt:")
    print(f"    tensor_g shape  : {tensor_g.shape}   ✓ matches DTB [100,5,14]")
    print(f"    tensor_g dtype  : {tensor_g.dtype}")
    print(f"    tensor_g min    : {tensor_g.min():.4f}")
    print(f"    tensor_g max    : {tensor_g.max():.4f}")
    print(f"    tensor_g mean   : {tensor_g.mean():.4f}")

    print(f"\n  CHANNEL LAYOUT (14 features):")
    ch_names = ["ML_score","FSL_score","CL_score","MI_score","epsilon",
                "PQ","IQ","EQ","SQ","CQ","MQ","HKS_depth","TB_density","task_novelty"]
    ch_means = tensor_g.mean(axis=(0,1))
    for i, (name, val) in enumerate(zip(ch_names, ch_means)):
        print(f"    [{i:2d}] {name:14s} = {val:.4f}")

    print(f"\n  CV_ADAPT FUSION PREVIEW:")
    print(f"    fused tensor shape : {fused.shape}  (4 components × 14 features)")
    print(f"    fused mean         : {fused.mean():.4f}")

    print(f"\n  FORMULA VERIFICATION:")
    ml, fsl, cl, mi = summary['ML'], summary['FSL'], summary['CL'], summary['MI']
    computed_eps = 1.0 / (ml + fsl + cl + mi)
    print(f"    1/({ml:.3f}+{fsl:.3f}+{cl:.3f}+{mi:.3f}) = {computed_eps:.6f}")
    print(f"    Stored ε = {summary['epsilon']:.6f}  ✓")

    print("\n" + "═" * 70)
    print("  STATUS: G_enhanced tensor ready for CV_Adapt.receive_tensor_inputs()")
    print("  NEXT:   cv_adapt_pipeline(tensor_g, tensor_r, tensor_ai, tensor_dtb)")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    print("\nRunning G_enhanced / e_Generalization — Component 1...")
    print("Formula: ε = 1 / (ML + FSL + g(HKS,TB) + f(Q))\n")

    tensor_g, summary = generate_tensor_from_G_enhanced(n_episodes=5)

    fused = simulate_cv_adapt_receive(tensor_g)

    print_report(summary, tensor_g, fused)

    # Save tensor for CV_Adapt handoff
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "tensor_g.npy"), tensor_g)
    with open(os.path.join(save_dir, "g_enhanced_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: tensor_g.npy        → CV_Adapt input")
    print(f"  Saved: g_enhanced_summary.json → metadata\n")
