"""
G_enhanced / e_Generalization  —  Component 1 of ai2agi
ai70000, Ltd.  |  Alex Osterneck, CLA, MSCS
Research blueprint: "How we go from AI to AGI (ai2agi)", March 2025

Formula:
    ε = 1 / (ML + FSL + g(HKS, TB) + f(Q))

Output tensor shape: [100, 5, 14]  — matches DTB convention for CV_Adapt fusion.

Pipeline (matches pseudo-code in Genesis document):
    Step 1: meta_learning()              → ML scalar (trainable)
    Step 2: few_shot_learning()          → FSL scalar (trainable)
    Step 3: continual_learning()         → CL = g(HKS, TB) scalar (trainable)
    Step 4: symbolic_representation()   → TB component of CL
    Step 5: generate_tensors()           → unified G_enhanced tensor [100,5,14]

Sub-modules:
    MetaLearner        — MAML-style inner/outer loop over task episodes
    FewShotLearner     — Prototypical Network over support/query sets
    ContinualLearner   — EWC (Elastic Weight Consolidation) + dynamic knowledge graph (HKS)
    SymbolicMemory     — Time-Binding structure (Korzybski TB), NLP semantic accumulation
    MultipleIntelligences — Six quotients f(Q): PQ, IQ, EQ, SQ, CQ, MQ
    GEnhanced          — Ensemble combiner → G_enhanced output tensor

Tensor output contract toward CV_Adapt:
    Shape  : [n_samples=100, n_neurons=5, n_features=14]
    Dtype  : float32  (numpy; CV_Adapt converts via tf.convert_to_tensor)
    Channel layout (14 features per node):
        [0]   ML_score        — meta-learning capability score
        [1]   FSL_score       — few-shot learning capability score
        [2]   CL_score        — continual learning score g(HKS,TB)
        [3]   MI_score        — multiple intelligences score f(Q)
        [4]   epsilon         — e_Generalization error  1/(ML+FSL+CL+MI)
        [5]   PQ              — Physical Quotient
        [6]   IQ              — Intelligence Quotient
        [7]   EQ              — Emotional Quotient
        [8]   SQ              — Social Quotient
        [9]   CQ              — Creative Quotient
        [10]  MQ              — Moral Quotient
        [11]  HKS_depth       — knowledge graph depth (hierarchy)
        [12]  TB_density      — time-binding semantic accumulation density
        [13]  task_novelty    — episode novelty index (few-shot regime signal)
"""

import math
import time
import copy
import random
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [G_enhanced] %(levelname)s %(message)s")
log = logging.getLogger("G_enhanced")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (match DTB output convention)
# ─────────────────────────────────────────────────────────────────────────────
N_SAMPLES   = 100   # samples per tensor batch
N_NEURONS   = 5     # node dimension (matches DTB n_neurons)
N_FEATURES  = 14    # feature channels (defined in docstring above)
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskEpisode:
    """A meta-learning / few-shot episode: support set + query set."""
    support_x: torch.Tensor   # [n_support, input_dim]
    support_y: torch.Tensor   # [n_support]
    query_x:   torch.Tensor   # [n_query,   input_dim]
    query_y:   torch.Tensor   # [n_query]
    task_id:   int = 0
    domain:    str = "generic"


@dataclass
class CognitiveArchitecture:
    """Filtered cognitive architecture record (from Sukhobokov et al. 2024)."""
    name:          str
    has_consciousness:  bool = False
    has_subconscious:   bool = False
    has_worldview:      bool = False
    has_reflection:     bool = False
    agi_capable:        bool = False
    mi_quotients:       Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — MetaLearner  (MAML inner/outer loop)
# ─────────────────────────────────────────────────────────────────────────────
class BaseNetwork(nn.Module):
    """Shared backbone for meta-learning and few-shot learning."""
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, output_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MetaLearner(nn.Module):
    """
    MAML-style meta-learner (Model-Agnostic Meta-Learning, Finn et al. 2017).
    Produces ML scalar: higher = lower generalization error.

    Inner loop: task-specific gradient step on support set.
    Outer loop: meta-gradient update across tasks.
    """
    def __init__(self, input_dim: int = 32, inner_lr: float = 0.01,
                 outer_lr: float = 0.001, inner_steps: int = 3):
        super().__init__()
        self.backbone  = BaseNetwork(input_dim=input_dim)
        self.head      = nn.Linear(16, 1)
        self.inner_lr  = inner_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=outer_lr)
        self._ml_score: float = 0.5   # running capability score

    def inner_loop(self, episode: TaskEpisode,
                   fast_weights: Optional[List[torch.Tensor]] = None
                   ) -> Tuple[torch.Tensor, float]:
        """Adapt to a single task via inner-loop gradient steps."""
        # Use copy of params for inner loop (manual MAML without higher lib)
        params = [p.clone().detach().requires_grad_(True)
                  for p in self.parameters()]

        for _ in range(self.inner_steps):
            # Forward with support set
            feat = self._forward_with_params(episode.support_x, params)
            loss = F.mse_loss(feat.squeeze(-1),
                              episode.support_y.float())
            grads = torch.autograd.grad(loss, params,
                                        create_graph=False,
                                        allow_unused=True)
            params = [p - self.inner_lr * (g if g is not None else torch.zeros_like(p))
                      for p, g in zip(params, grads)]

        # Evaluate on query set
        with torch.no_grad():
            q_feat = self._forward_with_params(episode.query_x, params)
            q_loss = F.mse_loss(q_feat.squeeze(-1),
                                episode.query_y.float())
            task_score = float(torch.sigmoid(1.0 - q_loss).item())

        return q_loss, task_score

    def _forward_with_params(self, x: torch.Tensor,
                              params: List[torch.Tensor]) -> torch.Tensor:
        """Functional forward pass using given parameter list."""
        param_iter = iter(params)
        out = x
        for module in self.backbone.net:
            if isinstance(module, nn.Linear):
                w = next(param_iter)
                b = next(param_iter)
                out = F.linear(out, w, b)
            elif isinstance(module, nn.ReLU):
                out = F.relu(out)
        # head
        w_h = next(param_iter)
        b_h = next(param_iter)
        out = F.linear(out, w_h, b_h)
        return out

    def meta_update(self, episodes: List[TaskEpisode]) -> float:
        """Outer loop: accumulate query losses across tasks, update meta-params."""
        self.meta_optimizer.zero_grad()
        total_loss = torch.tensor(0.0, requires_grad=True)
        scores = []

        for ep in episodes:
            q_loss, score = self.inner_loop(ep)
            total_loss = total_loss + q_loss
            scores.append(score)

        total_loss.backward()
        self.meta_optimizer.step()

        self._ml_score = float(np.mean(scores))
        return self._ml_score

    @property
    def ml_score(self) -> float:
        return max(self._ml_score, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — FewShotLearner  (Prototypical Networks)
# ─────────────────────────────────────────────────────────────────────────────
class FewShotLearner(nn.Module):
    """
    Prototypical Network (Snell et al. 2017).
    Computes class prototypes from support set, classifies query by distance.
    Produces FSL scalar: higher = better few-shot generalization.
    """
    def __init__(self, input_dim: int = 32, embed_dim: int = 16):
        super().__init__()
        self.encoder = BaseNetwork(input_dim=input_dim,
                                   hidden_dim=64, output_dim=embed_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self._fsl_score: float = 0.5
        self._novelty:   float = 0.5

    def compute_prototypes(self, support_x: torch.Tensor,
                           support_y: torch.Tensor,
                           n_classes: int) -> torch.Tensor:
        """Mean embedding per class → prototype tensor [n_classes, embed_dim]."""
        embeddings = self.encoder(support_x)
        protos = []
        for c in range(n_classes):
            mask = (support_y == c)
            if mask.sum() == 0:
                protos.append(torch.zeros(embeddings.shape[-1]).to(DEVICE))
            else:
                protos.append(embeddings[mask].mean(0))
        return torch.stack(protos)

    def forward_episode(self, episode: TaskEpisode) -> Tuple[float, float]:
        """One prototypical episode → (fsl_score, task_novelty)."""
        n_classes = int(episode.support_y.max().item()) + 1
        protos   = self.compute_prototypes(episode.support_x,
                                           episode.support_y, n_classes)
        q_embed  = self.encoder(episode.query_x)

        # Squared Euclidean distance to each prototype
        dists = torch.cdist(q_embed.unsqueeze(0),
                            protos.unsqueeze(0)).squeeze(0)
        log_p = F.log_softmax(-dists, dim=1)
        loss  = F.nll_loss(log_p, episode.query_y.long())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds   = log_p.argmax(dim=1)
            acc     = (preds == episode.query_y.long()).float().mean().item()
            # novelty: mean min-distance to any prototype (high = more novel)
            novelty = dists.min(dim=1).values.mean().item()
            novelty = float(torch.sigmoid(torch.tensor(novelty / 10.0)).item())

        self._fsl_score = float(acc)
        self._novelty   = novelty
        return self._fsl_score, self._novelty

    @property
    def fsl_score(self) -> float:
        return max(self._fsl_score, 1e-6)

    @property
    def task_novelty(self) -> float:
        return self._novelty


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — ContinualLearner  (EWC + HKS knowledge graph)
#             implements g(HKS, TB)
# ─────────────────────────────────────────────────────────────────────────────
class HierarchicalKnowledgeStructure:
    """
    Dynamic knowledge graph (HKS).
    Nodes = concepts; edges = relationships.
    Implements hierarchical-structure per DFRE framework (Latapie & Kilic 2020).
    """
    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._task_count: int  = 0

    def ingest_task(self, task_id: int, domain: str,
                    concepts: List[str]) -> None:
        """Add task knowledge to graph, creating hierarchy."""
        domain_node = f"domain::{domain}"
        if not self.graph.has_node(domain_node):
            self.graph.add_node(domain_node, level=0, type="domain")

        task_node = f"task::{task_id}"
        self.graph.add_node(task_node, level=1, type="task", domain=domain)
        self.graph.add_edge(domain_node, task_node, weight=1.0)

        for c in concepts:
            concept_node = f"concept::{c}"
            if not self.graph.has_node(concept_node):
                self.graph.add_node(concept_node, level=2, type="concept")
            self.graph.add_edge(task_node, concept_node, weight=0.5)

        self._task_count += 1

    @property
    def depth(self) -> float:
        """Normalized hierarchy depth score [0,1]."""
        if len(self.graph) == 0:
            return 0.0
        try:
            levels = [d.get("level", 0)
                      for _, d in self.graph.nodes(data=True)]
            return float(min(max(levels) / 5.0, 1.0))
        except Exception:
            return 0.0

    @property
    def connectivity(self) -> float:
        """Graph density as HKS richness metric."""
        if len(self.graph) < 2:
            return 0.0
        return float(nx.density(self.graph))


class EWCNetwork(nn.Module):
    """Simple network with EWC (Elastic Weight Consolidation) for continual learning."""
    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.fisher: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 100.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_fisher(self, data_loader: List[Tuple[torch.Tensor, torch.Tensor]],
                       n_samples: int = 50) -> None:
        """Estimate Fisher information matrix for EWC penalty."""
        self.fisher = {n: torch.zeros_like(p)
                       for n, p in self.named_parameters()}
        self.optimal_params = {n: p.clone().detach()
                                for n, p in self.named_parameters()}

        self.eval()
        for i, (x, y) in enumerate(data_loader):
            if i >= n_samples:
                break
            self.zero_grad()
            out  = self(x)
            loss = F.mse_loss(out.squeeze(), y.float())
            loss.backward()
            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += p.grad.data.pow(2)

        for n in self.fisher:
            self.fisher[n] /= max(n_samples, 1)
        self.train()

    def ewc_loss(self) -> torch.Tensor:
        """EWC regularization term: penalizes deviation from previous optimal params."""
        if not self.fisher:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        for n, p in self.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] *
                               (p - self.optimal_params[n]).pow(2)).sum()
        return (self.ewc_lambda / 2.0) * loss


class ContinualLearner:
    """
    Continual learning via EWC + HKS.
    Implements  CL = g(HKS, TB)
    Prevents catastrophic forgetting while accumulating knowledge hierarchically.
    """
    def __init__(self, input_dim: int = 32):
        self.network    = EWCNetwork(input_dim=input_dim).to(DEVICE)
        self.optimizer  = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        self.hks        = HierarchicalKnowledgeStructure()
        self._cl_score: float = 0.5
        self._task_buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

    def learn_task(self, episode: TaskEpisode, n_epochs: int = 5) -> float:
        """Learn new task with EWC regularization."""
        # Update HKS
        concepts = [f"feat_{i}" for i in range(episode.support_x.shape[1])]
        self.hks.ingest_task(episode.task_id, episode.domain, concepts[:4])

        # Add to buffer
        self._task_buffer.extend(
            zip(episode.support_x, episode.support_y))

        # Train with EWC
        total_loss_val = 0.0
        for _ in range(n_epochs):
            self.optimizer.zero_grad()
            out  = self.network(episode.support_x)
            task_loss = F.mse_loss(out.squeeze(), episode.support_y.float())
            ewc_loss  = self.network.ewc_loss()
            loss = task_loss + ewc_loss
            loss.backward()
            self.optimizer.step()
            total_loss_val = loss.item()

        # Update Fisher after task
        if self._task_buffer:
            self.network.compute_fisher(self._task_buffer[-50:])

        # CL score: inverse of loss, bounded [0,1]
        self._cl_score = float(torch.sigmoid(
            torch.tensor(1.0 / (1.0 + total_loss_val))).item())
        return self._cl_score

    @property
    def cl_score(self) -> float:
        return max(self._cl_score, 1e-6)

    @property
    def hks_depth(self) -> float:
        return self.hks.depth

    @property
    def hks_connectivity(self) -> float:
        return self.hks.connectivity


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — SymbolicMemory  (Time-Binding / TB — Korzybski 1921)
# ─────────────────────────────────────────────────────────────────────────────
class SymbolicMemory:
    """
    Time-Binding structure (Korzybski, 'Manhood of Humanity', 1921).
    Humans transmit knowledge across time through language/symbols.
    Here: accumulates semantic representations of tasks across episodes,
    implemented as a growing embedding memory with density tracking.

    This is the TB component of  CL = g(HKS, TB).
    """
    def __init__(self, embed_dim: int = 16):
        self.embed_dim   = embed_dim
        self.memory:     List[np.ndarray] = []
        self.timestamps: List[float]      = []
        self.semantic_labels: List[str]   = []
        self._tb_density: float           = 0.0

        # Simple NLP-proxy: hash-based semantic embedding
        # (production: replace with sentence-transformers)
        self._embed_cache: Dict[str, np.ndarray] = {}

    def _semantic_embed(self, text: str) -> np.ndarray:
        """Deterministic pseudo-semantic embedding from text (NLP proxy)."""
        if text in self._embed_cache:
            return self._embed_cache[text]
        rng = np.random.RandomState(abs(hash(text)) % (2**31))
        embed = rng.randn(self.embed_dim).astype(np.float32)
        embed /= (np.linalg.norm(embed) + 1e-8)
        self._embed_cache[text] = embed
        return embed

    def bind(self, concept: str, episode_id: int) -> None:
        """
        Time-bind a concept: store its semantic embedding with timestamp.
        Models Korzybski's 'accumulation and transmission of knowledge.'
        """
        label = f"{concept}::ep{episode_id}"
        embed = self._semantic_embed(label)
        self.memory.append(embed)
        self.timestamps.append(time.time())
        self.semantic_labels.append(label)
        self._update_density()

    def _update_density(self) -> None:
        """TB density: mean cosine similarity across stored embeddings."""
        if len(self.memory) < 2:
            self._tb_density = 0.0
            return
        mat = np.stack(self.memory[-20:])   # last 20 for efficiency
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
        mat_n = mat / norms
        sim_mat = mat_n @ mat_n.T
        n = len(mat_n)
        # Mean off-diagonal (exclude self-similarity)
        mask = ~np.eye(n, dtype=bool)
        self._tb_density = float(sim_mat[mask].mean())

    @property
    def tb_density(self) -> float:
        """Normalized TB density [0,1]; higher = richer time-bound knowledge."""
        return max(0.0, min(1.0, (self._tb_density + 1.0) / 2.0))

    @property
    def accumulation_score(self) -> float:
        """Score based on breadth of accumulated symbols."""
        return float(min(len(self.memory) / 200.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 5 — MultipleIntelligences  (f(Q) — Cichocki & Kuleshov 2020)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Quotients:
    """
    Six intelligence quotients from Cichocki & Kuleshov (2020).
    PQ=Physical, IQ=Intelligence, EQ=Emotional, SQ=Social, CQ=Creative, MQ=Moral.
    All normalized to [0,1].
    """
    PQ: float = 0.5   # Physical / sensorimotor
    IQ: float = 0.5   # Rational / cognitive
    EQ: float = 0.5   # Emotional
    SQ: float = 0.5   # Social
    CQ: float = 0.5   # Creative
    MQ: float = 0.5   # Moral / ethical

    def to_array(self) -> np.ndarray:
        return np.array([self.PQ, self.IQ, self.EQ,
                         self.SQ, self.CQ, self.MQ], dtype=np.float32)

    @classmethod
    def from_performance(cls, task_metrics: Dict[str, float]) -> "Quotients":
        """
        Infer quotient estimates from task performance metrics.
        Production: replace with dedicated assessors per quotient type.
        """
        acc    = task_metrics.get("accuracy",  0.5)
        novel  = task_metrics.get("novelty",   0.5)
        loss   = task_metrics.get("loss",      0.5)
        cl     = task_metrics.get("cl_score",  0.5)
        return cls(
            PQ=float(np.clip(acc * 0.8 + novel * 0.2, 0, 1)),
            IQ=float(np.clip(acc * 0.6 + (1.0 - loss) * 0.4, 0, 1)),
            EQ=float(np.clip(cl  * 0.7 + novel * 0.3, 0, 1)),
            SQ=float(np.clip(cl  * 0.5 + acc   * 0.5, 0, 1)),
            CQ=float(np.clip(novel * 0.8 + acc * 0.2, 0, 1)),
            MQ=float(np.clip(cl  * 0.6 + (1.0 - loss) * 0.4, 0, 1)),
        )


class MultipleIntelligences:
    """
    MI = f(Q)
    Weighted aggregation of six quotients into a single MI score.
    Higher MI → lower generalization error.
    """
    WEIGHTS = np.array([0.10, 0.25, 0.15, 0.15, 0.20, 0.15], dtype=np.float32)
    # PQ, IQ, EQ, SQ, CQ, MQ

    def __init__(self):
        self._quotients = Quotients()
        self._mi_score: float = 0.5

    def update(self, task_metrics: Dict[str, float]) -> float:
        self._quotients  = Quotients.from_performance(task_metrics)
        q_arr            = self._quotients.to_array()
        self._mi_score   = float(np.dot(q_arr, self.WEIGHTS))
        return self._mi_score

    @property
    def mi_score(self) -> float:
        return max(self._mi_score, 1e-6)

    @property
    def quotients(self) -> Quotients:
        return self._quotients


# ─────────────────────────────────────────────────────────────────────────────
# EPISODE GENERATOR  (synthetic task distribution)
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeGenerator:
    """
    Generates synthetic task episodes for the G_enhanced pipeline.
    In production: replace with real multi-domain datasets.
    """
    DOMAINS = ["language", "vision", "reasoning", "motor", "social", "symbolic"]

    def __init__(self, input_dim: int = 32, n_classes: int = 2,
                 n_support: int = 10, n_query: int = 20):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query   = n_query
        self._episode_count = 0

    def sample(self) -> TaskEpisode:
        """Sample one episode from a random synthetic task distribution."""
        domain = random.choice(self.DOMAINS)
        # Different tasks shift the data distribution (novelty simulation)
        shift = np.random.randn(self.input_dim).astype(np.float32) * 0.5
        scale = np.random.uniform(0.8, 1.5)

        def make_batch(n):
            x = (torch.randn(n, self.input_dim) + torch.tensor(shift)) * scale
            y = torch.randint(0, self.n_classes, (n,))
            return x.to(DEVICE), y.to(DEVICE)

        sx, sy = make_batch(self.n_support)
        qx, qy = make_batch(self.n_query)
        ep = TaskEpisode(support_x=sx, support_y=sy,
                         query_x=qx,   query_y=qy,
                         task_id=self._episode_count,
                         domain=domain)
        self._episode_count += 1
        return ep


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — GEnhanced  (the unified G_enhanced module)
# ─────────────────────────────────────────────────────────────────────────────
class GEnhanced:
    """
    G_enhanced / e_Generalization  — Component 1, ai2agi architecture.

    Implements:
        ε = 1 / (ML + FSL + g(HKS, TB) + f(Q))

    Output:
        generate_tensors() → numpy array [100, 5, 14] matching DTB shape.
        Ready for tf.convert_to_tensor() in CV_Adapt.
    """

    def __init__(self, input_dim: int = 32):
        log.info("Initializing G_enhanced (e_Generalization) module...")
        self.input_dim = input_dim

        # Five sub-systems
        self.meta_learner    = MetaLearner(input_dim=input_dim).to(DEVICE)
        self.fsl_learner     = FewShotLearner(input_dim=input_dim).to(DEVICE)
        self.cont_learner    = ContinualLearner(input_dim=input_dim)
        self.sym_memory      = SymbolicMemory()
        self.multi_intel     = MultipleIntelligences()

        self.episode_gen     = EpisodeGenerator(input_dim=input_dim)

        # State tracking
        self._last_epsilon:    float = 1.0
        self._last_ml:         float = 0.0
        self._last_fsl:        float = 0.0
        self._last_cl:         float = 0.0
        self._last_mi:         float = 0.0
        self._episode_history: List[Dict] = []

        log.info(f"  Device: {DEVICE}")
        log.info(f"  Output tensor shape: [{N_SAMPLES}, {N_NEURONS}, {N_FEATURES}]")
        log.info("G_enhanced initialized.")

    # ──────────────────────────────────────────────────────────────────
    # STEP 1: meta_learning()
    # ──────────────────────────────────────────────────────────────────
    def meta_learning(self, n_tasks: int = 4) -> float:
        """
        MAML outer loop over n_tasks episodes.
        Returns ML capability score.
        """
        episodes = [self.episode_gen.sample() for _ in range(n_tasks)]
        ml_score = self.meta_learner.meta_update(episodes)
        self._last_ml = ml_score
        log.info(f"  meta_learning()  ML_score = {ml_score:.4f}")
        return ml_score

    # ──────────────────────────────────────────────────────────────────
    # STEP 2: few_shot_learning()
    # ──────────────────────────────────────────────────────────────────
    def few_shot_learning(self, episode: Optional[TaskEpisode] = None) -> float:
        """
        Prototypical Network episode.
        Returns FSL accuracy score.
        """
        if episode is None:
            episode = self.episode_gen.sample()
        fsl_score, novelty = self.fsl_learner.forward_episode(episode)
        self._last_fsl = fsl_score
        log.info(f"  few_shot_learning()  FSL_score = {fsl_score:.4f}  "
                 f"novelty = {novelty:.4f}")
        return fsl_score

    # ──────────────────────────────────────────────────────────────────
    # STEP 3: continual_learning()  =  g(HKS, TB)
    # ──────────────────────────────────────────────────────────────────
    def continual_learning(self, episode: Optional[TaskEpisode] = None) -> float:
        """
        EWC-protected task learning + HKS graph update.
        Returns CL score = g(HKS, TB).
        """
        if episode is None:
            episode = self.episode_gen.sample()
        cl_score = self.cont_learner.learn_task(episode)
        self._last_cl = cl_score
        log.info(f"  continual_learning()  CL_score = {cl_score:.4f}  "
                 f"HKS_depth = {self.cont_learner.hks_depth:.4f}")
        return cl_score

    # ──────────────────────────────────────────────────────────────────
    # STEP 4: symbolic_representation()  (Time-Binding TB)
    # ──────────────────────────────────────────────────────────────────
    def symbolic_representation(self, episode_id: int = 0,
                                 domain: str = "generic") -> float:
        """
        Korzybski Time-Binding: bind domain concepts into symbolic memory.
        Updates TB component of CL.
        Returns TB accumulation score.
        """
        concepts = [domain, f"task_{episode_id}", "generalization",
                    "meta", "few_shot", "continual"]
        for c in concepts:
            self.sym_memory.bind(c, episode_id)
        tb_score = self.sym_memory.accumulation_score
        log.info(f"  symbolic_representation()  TB_density = "
                 f"{self.sym_memory.tb_density:.4f}  TB_accum = {tb_score:.4f}")
        return tb_score

    # ──────────────────────────────────────────────────────────────────
    # STEP 5: generate_tensors()  →  G_enhanced output tensor
    # ──────────────────────────────────────────────────────────────────
    def generate_tensors(self, n_episodes: int = 5) -> np.ndarray:
        """
        Full G_enhanced pipeline.
        Runs n_episodes, collects sub-system scores, computes ε,
        then builds output tensor of shape [N_SAMPLES, N_NEURONS, N_FEATURES].

        Returns:
            numpy float32 array [100, 5, 14]
            Ready for CV_Adapt via tf.convert_to_tensor(tensor_g, dtype=tf.float32)
        """
        # ── FOUNDATION REQUIREMENT ─────────────────────────────────────
        # FoundationInitializer.run() must be called before generate_tensors().
        # Raises FoundationNotInitializedError if not initialized.
        # See: foundation_initializer.py — ai2agi, Ai70000 Ltd.
        from foundation_initializer import verify_foundation
        verify_foundation(self)
        # ──────────────────────────────────────────────────────────────

        log.info("=" * 60)
        log.info("generate_tensors() — G_enhanced pipeline starting")
        log.info("=" * 60)

        # ── Run n_episodes of learning ──────────────────────────────
        for ep_idx in range(n_episodes):
            log.info(f"\n── Episode {ep_idx + 1}/{n_episodes} ──")
            episode = self.episode_gen.sample()

            ml_score  = self.meta_learning(n_tasks=3)
            fsl_score = self.few_shot_learning(episode)
            cl_score  = self.continual_learning(episode)
            tb_score  = self.symbolic_representation(episode.task_id,
                                                      episode.domain)

            task_metrics = {
                "accuracy": fsl_score,
                "novelty":  self.fsl_learner.task_novelty,
                "loss":     1.0 - cl_score,
                "cl_score": cl_score,
            }
            mi_score = self.multi_intel.update(task_metrics)

            # ── Compute ε = 1/(ML + FSL + CL + MI) ─────────────────
            denom   = ml_score + fsl_score + cl_score + mi_score
            epsilon = 1.0 / max(denom, 1e-6)

            self._last_epsilon = epsilon
            self._last_ml      = ml_score
            self._last_fsl     = fsl_score
            self._last_cl      = cl_score
            self._last_mi      = mi_score

            self._episode_history.append({
                "ep": ep_idx, "ML": ml_score, "FSL": fsl_score,
                "CL": cl_score, "MI": mi_score, "epsilon": epsilon,
                "domain": episode.domain,
            })

            log.info(f"  ε = 1/({ml_score:.3f}+{fsl_score:.3f}+"
                     f"{cl_score:.3f}+{mi_score:.3f}) = {epsilon:.4f}")

        # ── Build output tensor  [100, 5, 14] ───────────────────────
        log.info("\nBuilding G_enhanced output tensor [100, 5, 14]...")
        tensor = self._build_output_tensor()

        log.info(f"G_enhanced tensor generated.")
        log.info(f"  Shape  : {tensor.shape}")
        log.info(f"  Dtype  : {tensor.dtype}")
        log.info(f"  ε (final): {self._last_epsilon:.6f}")
        log.info(f"  ML={self._last_ml:.4f}  FSL={self._last_fsl:.4f}  "
                 f"CL={self._last_cl:.4f}  MI={self._last_mi:.4f}")
        log.info("=" * 60)

        return tensor

    def _build_output_tensor(self) -> np.ndarray:
        """
        Construct [N_SAMPLES=100, N_NEURONS=5, N_FEATURES=14] tensor.

        Each sample is a slightly perturbed version of the current state,
        modeling the stochastic nature of generalization across instances.
        Each of the 5 neuron-nodes carries the same 14-channel feature vector
        with small Gaussian noise to model within-sample variance — matching
        how DTB distributes neuron-level data across its 5-neuron dimension.
        """
        q   = self.multi_intel.quotients
        rng = np.random.default_rng(seed=int(time.time()) % 100000)

        # Base feature vector (14 channels as per contract)
        base = np.array([
            self._last_ml,                       # [0]  ML_score
            self._last_fsl,                      # [1]  FSL_score
            self._last_cl,                       # [2]  CL_score
            self._last_mi,                       # [3]  MI_score
            self._last_epsilon,                  # [4]  epsilon
            q.PQ,                                # [5]  PQ
            q.IQ,                                # [6]  IQ
            q.EQ,                                # [7]  EQ
            q.SQ,                                # [8]  SQ
            q.CQ,                                # [9]  CQ
            q.MQ,                                # [10] MQ
            self.cont_learner.hks_depth,         # [11] HKS_depth
            self.sym_memory.tb_density,          # [12] TB_density
            self.fsl_learner.task_novelty,       # [13] task_novelty
        ], dtype=np.float32)

        # Replicate to [N_SAMPLES, N_NEURONS, N_FEATURES] with biological noise
        tensor = np.tile(base, (N_SAMPLES, N_NEURONS, 1))
        noise  = rng.normal(0, 0.02, tensor.shape).astype(np.float32)
        tensor = np.clip(tensor + noise, 0.0, 1.0)

        # epsilon channel: re-derived from noisy ML/FSL/CL/MI for each sample
        ml_n  = tensor[:, :, 0]
        fsl_n = tensor[:, :, 1]
        cl_n  = tensor[:, :, 2]
        mi_n  = tensor[:, :, 3]
        tensor[:, :, 4] = 1.0 / np.maximum(ml_n + fsl_n + cl_n + mi_n, 1e-6)
        # Re-clip epsilon (can exceed 1 for weak learners)
        tensor[:, :, 4] = np.clip(tensor[:, :, 4], 0.0, 2.0)

        return tensor.astype(np.float32)

    def summary(self) -> Dict:
        """Return current state summary for logging / CV_Adapt metadata."""
        q = self.multi_intel.quotients
        return {
            "component":  "G_enhanced",
            "formula":    "ε = 1/(ML + FSL + g(HKS,TB) + f(Q))",
            "epsilon":     self._last_epsilon,
            "ML":          self._last_ml,
            "FSL":         self._last_fsl,
            "CL":          self._last_cl,
            "MI":          self._last_mi,
            "quotients":   {"PQ": q.PQ, "IQ": q.IQ, "EQ": q.EQ,
                            "SQ": q.SQ, "CQ": q.CQ, "MQ": q.MQ},
            "HKS_depth":   self.cont_learner.hks_depth,
            "TB_density":  self.sym_memory.tb_density,
            "task_novelty": self.fsl_learner.task_novelty,
            "tensor_shape": [N_SAMPLES, N_NEURONS, N_FEATURES],
            "tensor_dtype": "float32",
            "cv_adapt_key": "tensor_g",
            "episodes_run": len(self._episode_history),
        }
