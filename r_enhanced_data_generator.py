## FILE I: r_enhanced_data_generator.py
## ai2agi by: ai70000, Ltd.
## Author: Alex Osterneck, CLA, MSCS
## Component_2: R_enhanced (e_Reasoning)
##
## overview: Generates synthetic input data representing the logical-reasoning
## state of an agent — symbolic propositions, neural embeddings, reinforcement
## signals, causal graphs, contextual state, memory traces, and math problem
## encodings. Mirrors the DataGenerator pattern from component_4 (DTB).
## Output: float32 TensorFlow tensor → input to R_EnhancedModel (File II)
## → final tensor_r fed to CV_Adapt (component_5) as one of four tensor inputs.
##
## Formula (from ai2agi research, Osterneck 2025):
##   ε_r = 1 / (SM(PN(CA(CR(RL(NS(input_data), reward(human_feedback)))))))
##
## Sequential pipeline stages:
##   NS  → Neuro-Symbolic Integration
##   RL  → Reinforcement Learning / RLHF
##   CR  → Causal Reasoning
##   CA  → Contextual Adaptation
##   PN  → Progressive Networks (anti-forgetting)
##   SM  → Society of Minds (math consensus)

import numpy as np
import tensorflow as tf


class REnhancedDataGenerator:
    """
    Generates synthetic multi-dimensional data encoding the six reasoning
    sub-systems of R_enhanced. Each sub-system has its own feature block.
    Real-world replacement: live symbolic KB, neural encoder output, RLHF
    reward signal, causal graph adjacency, context embeddings, task memory,
    and math problem state tensors.
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_symbols: int = 16,        # symbolic propositions per sample
        n_embedding_dim: int = 32,  # neural embedding dimension
        n_causal_nodes: int = 8,    # nodes in causal graph
        n_context_dim: int = 12,    # context state dimensions
        n_memory_traces: int = 6,   # progressive network memory slots
        n_math_ops: int = 10,       # math problem encoding length
    ):
        self.n_samples       = n_samples
        self.n_symbols       = n_symbols
        self.n_embedding_dim = n_embedding_dim
        self.n_causal_nodes  = n_causal_nodes
        self.n_context_dim   = n_context_dim
        self.n_memory_traces = n_memory_traces
        self.n_math_ops      = n_math_ops

    # ------------------------------------------------------------------
    # Sub-block generators (one per pipeline stage)
    # ------------------------------------------------------------------

    def _gen_ns_block(self):
        """
        Neuro-Symbolic block.
        Symbolic: binary truth-value propositions [0,1].
        Neural:   continuous embedding vector (normalized).
        Returns shape: (n_samples, n_symbols + n_embedding_dim)
        """
        symbolic_props = np.random.randint(0, 2, size=(
            self.n_samples, self.n_symbols)).astype(np.float32)
        neural_embeds  = np.random.uniform(-1.0, 1.0, size=(
            self.n_samples, self.n_embedding_dim)).astype(np.float32)
        # Normalize embeddings to unit sphere (mirrors real encoder output)
        norms = np.linalg.norm(neural_embeds, axis=1, keepdims=True) + 1e-8
        neural_embeds  = neural_embeds / norms
        return np.concatenate([symbolic_props, neural_embeds], axis=1)

    def _gen_rl_block(self):
        """
        Reinforcement Learning / RLHF block.
        reward:       scalar human-feedback reward ∈ [-1, 1].
        q_values:     action-value estimates for current state.
        policy_logits: policy distribution pre-softmax.
        Returns shape: (n_samples, 3)  [reward, max_q, policy_entropy]
        """
        reward        = np.random.uniform(-1.0,  1.0, size=(self.n_samples, 1))
        q_values      = np.random.uniform(-5.0,  5.0, size=(self.n_samples, 4))
        policy_logits = np.random.uniform(-3.0,  3.0, size=(self.n_samples, 4))
        policy_probs  = np.exp(policy_logits) / np.exp(policy_logits).sum(
            axis=1, keepdims=True)
        policy_entropy = -(policy_probs * np.log(policy_probs + 1e-8)).sum(
            axis=1, keepdims=True)
        max_q = q_values.max(axis=1, keepdims=True)
        return np.concatenate([reward, max_q, policy_entropy], axis=1).astype(
            np.float32)

    def _gen_cr_block(self):
        """
        Causal Reasoning block.
        Encodes a sparse causal adjacency matrix (flattened upper triangle).
        Values in [0,1] represent causal strength between nodes.
        Returns shape: (n_samples, n_causal_nodes*(n_causal_nodes-1)//2)
        """
        n = self.n_causal_nodes
        n_edges = n * (n - 1) // 2
        # Sparse: ~30% of edges are non-zero (realistic causal graph density)
        mask   = np.random.binomial(1, 0.3, size=(self.n_samples, n_edges))
        strength = np.random.uniform(0.0, 1.0, size=(self.n_samples, n_edges))
        return (mask * strength).astype(np.float32)

    def _gen_ca_block(self):
        """
        Contextual Adaptation block.
        Encodes environmental context state: task-type one-hot, urgency,
        domain familiarity, and a continuous context embedding.
        Returns shape: (n_samples, n_context_dim)
        """
        n_task_types = 4
        task_onehot  = np.eye(n_task_types)[np.random.randint(
            0, n_task_types, size=self.n_samples)].astype(np.float32)
        urgency      = np.random.uniform(0.0, 1.0, size=(self.n_samples, 1))
        familiarity  = np.random.uniform(0.0, 1.0, size=(self.n_samples, 1))
        ctx_embed    = np.random.uniform(-1.0, 1.0, size=(
            self.n_samples, self.n_context_dim - n_task_types - 2))
        return np.concatenate([task_onehot, urgency, familiarity, ctx_embed],
                              axis=1).astype(np.float32)

    def _gen_pn_block(self):
        """
        Progressive Networks (anti-forgetting) block.
        Encodes memory traces from prior task columns:
        activation strengths of lateral connections from previous task columns.
        Returns shape: (n_samples, n_memory_traces)
        """
        # Lateral connection weights from previous columns ∈ [0, 1]
        lateral_weights = np.random.uniform(0.0, 1.0, size=(
            self.n_samples, self.n_memory_traces)).astype(np.float32)
        # Older traces decay exponentially
        decay = np.exp(-np.arange(self.n_memory_traces) * 0.3).astype(
            np.float32)
        return lateral_weights * decay[np.newaxis, :]

    def _gen_sm_block(self):
        """
        Society of Minds (math consensus) block.
        Encodes multiple model proposals for a math sub-problem:
        each slot = one model's proposed answer (normalized) + confidence.
        Returns shape: (n_samples, n_math_ops * 2)  [answer, confidence] pairs
        """
        answers     = np.random.uniform(-10.0, 10.0, size=(
            self.n_samples, self.n_math_ops)).astype(np.float32)
        confidences = np.random.uniform(0.0,   1.0,  size=(
            self.n_samples, self.n_math_ops)).astype(np.float32)
        # Interleave: [a0, c0, a1, c1, ...]
        sm_block = np.stack([answers, confidences], axis=2).reshape(
            self.n_samples, self.n_math_ops * 2)
        return sm_block

    # ------------------------------------------------------------------
    # Main generator
    # ------------------------------------------------------------------

    def generate_data(self) -> tf.Tensor:
        """
        Concatenates all six sub-blocks into one flat feature tensor per sample.
        Output shape: (n_samples, total_features)
        total_features = ns + rl + cr + ca + pn + sm feature counts.
        """
        try:
            ns_block = self._gen_ns_block()   # (S, n_symbols + n_embedding_dim)
            rl_block = self._gen_rl_block()   # (S, 3)
            cr_block = self._gen_cr_block()   # (S, n_causal_nodes*(n_causal_nodes-1)//2)
            ca_block = self._gen_ca_block()   # (S, n_context_dim)
            pn_block = self._gen_pn_block()   # (S, n_memory_traces)
            sm_block = self._gen_sm_block()   # (S, n_math_ops * 2)

            print("\nR_enhanced Data Generator — Feature Block Verification:")
            print(f"  NS block (neuro-symbolic):      {ns_block.shape}  "
                  f"| range [{ns_block.min():.3f}, {ns_block.max():.3f}]")
            print(f"  RL block (reward/RLHF):         {rl_block.shape}  "
                  f"| range [{rl_block.min():.3f}, {rl_block.max():.3f}]")
            print(f"  CR block (causal graph):        {cr_block.shape}  "
                  f"| range [{cr_block.min():.3f}, {cr_block.max():.3f}]")
            print(f"  CA block (context adaptation):  {ca_block.shape}  "
                  f"| range [{ca_block.min():.3f}, {ca_block.max():.3f}]")
            print(f"  PN block (progressive nets):    {pn_block.shape}  "
                  f"| range [{pn_block.min():.3f}, {pn_block.max():.3f}]")
            print(f"  SM block (society of minds):    {sm_block.shape}  "
                  f"| range [{sm_block.min():.3f}, {sm_block.max():.3f}]")

            input_data = np.concatenate(
                [ns_block, rl_block, cr_block, ca_block, pn_block, sm_block],
                axis=1)

            print(f"\n  Combined input tensor shape:    {input_data.shape}")
            print(f"  Total features per sample:      {input_data.shape[1]}")

            return tf.convert_to_tensor(input_data, dtype=tf.float32)

        except Exception as e:
            raise Exception(f"REnhancedDataGenerator.generate_data() error: {e}")

    def get_feature_labels(self) -> list:
        """Returns ordered feature block labels matching generate_data() output."""
        n = self.n_causal_nodes
        return [
            f"NS_symbolic_{i}"   for i in range(self.n_symbols)
        ] + [
            f"NS_embed_{i}"      for i in range(self.n_embedding_dim)
        ] + [
            "RL_reward", "RL_max_q", "RL_policy_entropy"
        ] + [
            f"CR_edge_{i}"       for i in range(n * (n - 1) // 2)
        ] + [
            f"CA_context_{i}"    for i in range(self.n_context_dim)
        ] + [
            f"PN_lateral_{i}"    for i in range(self.n_memory_traces)
        ] + [
            f"SM_answer_{i//2}" if i % 2 == 0 else f"SM_conf_{i//2}"
            for i in range(self.n_math_ops * 2)
        ]


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("FILE I: R_enhanced Data Generator")
    print("Component_2 of ai2agi (ai70000, Ltd.)")
    print("=" * 60)

    gen    = REnhancedDataGenerator(n_samples=100)
    tensor = gen.generate_data()

    print(f"\nOutput tensor dtype:  {tensor.dtype}")
    print(f"Output tensor shape:  {tensor.shape}")
    print(f"\nFirst sample (first 10 features):")
    print(f"  {tensor[0, :10].numpy().round(4)}")
    print(f"\nFeature labels (first 10):")
    for label in gen.get_feature_labels()[:10]:
        print(f"  {label}")
    print("\nData generator verified. Ready for File II (R_EnhancedModel).")
