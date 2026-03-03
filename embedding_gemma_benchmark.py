#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# EmbeddingGemma-300m Benchmark (raw OpenVINO Python API)
# Tests INT4 vs FP32 models on CPU and GPU, measuring:
#   - TTFT  (Time To First Token / first-inference latency)
#   - TPOT  (Time Per Output Token / per-embedding latency)
#   - Cosine-similarity accuracy between INT4 and FP32 embeddings
#   - Ranking accuracy (Recall@1) across multiple test scenarios
#
# Usage:
#   python embedding_gemma_benchmark.py <INT4_MODEL_DIR> <FP32_MODEL_DIR> [warmup] [iters] [dataset.tsv]

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import openvino_genai as ov_genai


# ──────────────────────── helpers ────────────────────────

@dataclass
class TimingStats:
    first_latency_ms: float = 0.0   # TTFT  - first call latency
    mean_latency_ms: float = 0.0    # TPOT  - average per-call latency
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    iterations: int = 0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(dot / norm)


def percentile(sorted_arr: np.ndarray, p: float) -> float:
    """Linear-interpolation percentile matching the C++ implementation."""
    if sorted_arr.size == 0:
        return 0.0
    idx = p / 100.0 * (sorted_arr.size - 1)
    lo = int(idx)
    hi = min(lo + 1, sorted_arr.size - 1)
    frac = idx - lo
    return float(sorted_arr[lo] * (1.0 - frac) + sorted_arr[hi] * frac)


# ──────────────── mean pooling + L2 normalize ────────────────

def mean_pool_and_normalize(
    embeddings: np.ndarray,      # [batch, seq_len, hidden]
    attention_mask: np.ndarray,  # [batch, seq_len]
) -> np.ndarray:
    """Mean pooling over token dim respecting attention_mask, then L2 normalize."""
    # Expand mask for broadcasting: [batch, seq_len, 1]
    mask = attention_mask[:, :, np.newaxis].astype(np.float64)
    # Masked sum
    summed = np.sum(embeddings.astype(np.float64) * mask, axis=1)          # [batch, hidden]
    counts = np.maximum(mask.sum(axis=1), 1.0)                             # [batch, 1]
    pooled = summed / counts                                               # [batch, hidden]
    # L2 normalize
    norms = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-12
    return (pooled / norms).astype(np.float32)


# ──────────────── embedding inference ────────────────

@dataclass
class EmbeddingOutput:
    query_embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    doc_embeddings: List[np.ndarray] = field(default_factory=list)


def has_input(compiled_model: ov.CompiledModel, name: str) -> bool:
    """Check if the compiled model has a named input."""
    for inp in compiled_model.inputs:
        if name in inp.get_names():
            return True
    return False


def embed_texts(
    request: ov.InferRequest,
    tokenizer: ov_genai.Tokenizer,
    model_has_position_ids: bool,
    texts: List[str],
) -> np.ndarray:
    """Run inference on a list of texts and return pooled+normalized embeddings."""
    encoded = tokenizer.encode(texts)

    input_ids = encoded.input_ids.data
    attention_mask = encoded.attention_mask.data

    request.set_tensor("input_ids", encoded.input_ids)
    request.set_tensor("attention_mask", encoded.attention_mask)

    if model_has_position_ids:
        batch, seq_len = input_ids.shape
        pos_ids = np.tile(np.arange(seq_len, dtype=np.int64), (batch, 1))
        request.set_tensor("position_ids", ov.Tensor(pos_ids))

    request.infer()

    output = request.get_output_tensor().data  # [batch, seq_len, hidden]
    return mean_pool_and_normalize(output, attention_mask)


# ──────────────── test scenario ────────────────

@dataclass
class TestScenario:
    query: str = ""
    documents: List[str] = field(default_factory=list)
    expected_top_idx: int = -1  # index of best-matching doc (-1 = unknown)


def get_builtin_scenarios() -> List[TestScenario]:
    """Built-in test scenarios covering diverse domains."""
    return [
        # 0 - Astronomy
        TestScenario(
            query="Which planet is known as the Red Planet?",
            documents=[
                "Venus is often called Earth's twin because of its similar size and proximity.",
                "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                "Jupiter, the largest planet in our solar system, has a prominent red spot.",
                "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
            ],
            expected_top_idx=1,
        ),
        # 1 - History
        TestScenario(
            query="Who was the first person to walk on the moon?",
            documents=[
                "Yuri Gagarin was the first human to journey into outer space in 1961.",
                "Neil Armstrong became the first person to walk on the moon on July 20, 1969.",
                "Buzz Aldrin was the second person to walk on the lunar surface during Apollo 11.",
                "John Glenn was the first American to orbit the Earth in 1962.",
            ],
            expected_top_idx=1,
        ),
        # 2 - Computer Science
        TestScenario(
            query="What is a neural network?",
            documents=[
                "A database is an organized collection of structured information stored electronically.",
                "A compiler translates source code written in a programming language into machine code.",
                "A neural network is a computing system inspired by biological neural networks in the brain, consisting of interconnected nodes that process information.",
                "An operating system manages computer hardware and provides services for applications.",
            ],
            expected_top_idx=2,
        ),
        # 3 - Biology
        TestScenario(
            query="How do antibiotics work?",
            documents=[
                "Vaccines stimulate the immune system to produce antibodies against specific pathogens.",
                "Antibiotics work by killing bacteria or preventing them from reproducing, either by disrupting cell wall synthesis, protein production, or DNA replication.",
                "Vitamins are organic compounds that are essential nutrients required in small quantities.",
                "Hormones are chemical messengers that travel through the bloodstream to tissues and organs.",
            ],
            expected_top_idx=1,
        ),
        # 4 - Geography
        TestScenario(
            query="What is the longest river in the world?",
            documents=[
                "Mount Everest, standing at 8,849 meters, is the highest peak above sea level.",
                "The Sahara Desert is the largest hot desert in the world, covering most of North Africa.",
                "The Nile River, stretching approximately 6,650 kilometers through northeastern Africa, is considered the longest river in the world.",
                "The Pacific Ocean is the largest and deepest ocean, covering more than 30 percent of Earth's surface.",
            ],
            expected_top_idx=2,
        ),
        # 5 - Physics
        TestScenario(
            query="What is dark matter?",
            documents=[
                "Antimatter is made up of particles with the same mass but opposite charge as normal matter.",
                "Dark energy is a hypothetical form of energy that permeates all of space and accelerates expansion.",
                "Gravity is a fundamental force that attracts objects with mass toward each other.",
                "Dark matter is an invisible form of matter that does not emit, absorb, or reflect light, detected only through its gravitational effects on visible matter.",
            ],
            expected_top_idx=3,
        ),
        # 6 - Music
        TestScenario(
            query="Who composed the Ninth Symphony?",
            documents=[
                "Mozart composed over 600 works including symphonies, operas, and chamber music.",
                "Bach is known for his intricate compositions including the Brandenburg Concertos.",
                "Ludwig van Beethoven composed his famous Ninth Symphony, which includes the Ode to Joy choral finale.",
                "Tchaikovsky composed several famous ballets including Swan Lake and The Nutcracker.",
            ],
            expected_top_idx=2,
        ),
        # 7 - Medicine
        TestScenario(
            query="What are the symptoms of a heart attack?",
            documents=[
                "A stroke occurs when blood supply to part of the brain is interrupted or reduced.",
                "Symptoms of a heart attack include chest pain or pressure, shortness of breath, pain radiating to the arm or jaw, nausea, and cold sweat.",
                "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
                "Appendicitis typically starts with pain around the navel that shifts to the lower right abdomen.",
            ],
            expected_top_idx=1,
        ),
        # 8 - Technology
        TestScenario(
            query="How does 5G technology differ from 4G?",
            documents=[
                "Wi-Fi 6 offers improved speed, latency, and capacity compared to previous Wi-Fi generations.",
                "Bluetooth 5.0 provides longer range and faster data transfer than Bluetooth 4.2.",
                "Fiber optic cables transmit data as pulses of light through glass or plastic strands.",
                "5G technology offers significantly faster data speeds, lower latency, and greater capacity than 4G by using higher frequency bands and advanced antenna technology.",
            ],
            expected_top_idx=3,
        ),
        # 9 - Environment
        TestScenario(
            query="What causes ocean acidification?",
            documents=[
                "Coral bleaching occurs when corals expel their symbiotic algae due to stress from warm water.",
                "Overfishing depletes fish stocks faster than they can replenish through natural reproduction.",
                "Ocean acidification is caused by the absorption of excess carbon dioxide from the atmosphere, which reacts with seawater to form carbonic acid, lowering the ocean's pH.",
                "Plastic pollution in the oceans harms marine wildlife through ingestion and entanglement.",
            ],
            expected_top_idx=2,
        ),
    ]


def load_tsv_dataset(tsv_path: str) -> List[TestScenario]:
    """Load scenarios from a TSV file.

    Format (tab-separated, first row is header):
        query \\t expected_idx \\t doc_0 \\t doc_1 \\t doc_2 \\t ...
    """
    scenarios: List[TestScenario] = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 4:
                continue
            query = row[0]
            expected_idx = int(row[1])
            docs = [d for d in row[2:] if d.strip()]
            scenarios.append(TestScenario(query=query, documents=docs, expected_top_idx=expected_idx))
    return scenarios


# ──────────────── run all scenarios on one model config ────────────────

@dataclass
class ScenarioResult:
    embeddings: EmbeddingOutput = field(default_factory=EmbeddingOutput)
    predicted_top_idx: int = -1
    top_score: float = 0.0
    correct: bool = False


@dataclass
class ConfigResult:
    label: str = ""
    stats: TimingStats = field(default_factory=TimingStats)
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    correct_count: int = 0
    total_scenarios: int = 0
    success: bool = False


def run_all_scenarios(
    model_dir: str,
    device: str,
    query_prompt: str,
    doc_prompt: str,
    scenarios: List[TestScenario],
    warmup_iters: int,
    bench_iters: int,
) -> ConfigResult:
    cr = ConfigResult(total_scenarios=len(scenarios))

    print(f"  Loading model from: {model_dir} on {device} ...", end="", flush=True)
    t0 = time.perf_counter()

    core = ov.Core()
    model = core.read_model(os.path.join(model_dir, "openvino_model.xml"))
    compiled = core.compile_model(model, device)
    request = compiled.create_infer_request()
    tokenizer = ov_genai.Tokenizer(model_dir)
    model_has_pos_ids = has_input(compiled, "position_ids")

    load_ms = (time.perf_counter() - t0) * 1000
    print(f" loaded in {load_ms:.1f} ms")

    # Warmup with first scenario
    wq = query_prompt + scenarios[0].query
    wd = [doc_prompt + d for d in scenarios[0].documents]
    for _ in range(warmup_iters):
        embed_texts(request, tokenizer, model_has_pos_ids, [wq])
        embed_texts(request, tokenizer, model_has_pos_ids, wd)

    # Benchmark
    all_latencies: List[float] = []
    first_latency = 0.0
    last_iter_results: List[ScenarioResult] = []

    for iteration in range(bench_iters):
        iter_results: List[ScenarioResult] = []
        for si, sc in enumerate(scenarios):
            pq = query_prompt + sc.query
            pd = [doc_prompt + d for d in sc.documents]

            qs = time.perf_counter()
            q_emb = embed_texts(request, tokenizer, model_has_pos_ids, [pq])
            qe = time.perf_counter()
            all_latencies.append((qe - qs) * 1000)
            if iteration == 0 and si == 0:
                first_latency = all_latencies[-1]

            ds = time.perf_counter()
            d_emb = embed_texts(request, tokenizer, model_has_pos_ids, pd)
            de = time.perf_counter()
            all_latencies.append((de - ds) * 1000)

            # Score documents
            scores = [cosine_similarity(q_emb[0], d_emb[di]) for di in range(len(d_emb))]
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            correct = best_idx == sc.expected_top_idx

            sr = ScenarioResult(
                embeddings=EmbeddingOutput(
                    query_embedding=q_emb[0],
                    doc_embeddings=[d_emb[i] for i in range(len(d_emb))],
                ),
                predicted_top_idx=best_idx,
                top_score=best_score,
                correct=correct,
            )
            iter_results.append(sr)

        last_iter_results = iter_results

    # Use last iteration results
    cr.scenario_results = last_iter_results
    cr.correct_count = sum(1 for sr in last_iter_results if sr.correct)

    # Compute stats
    lat = np.array(all_latencies)
    lat_sorted = np.sort(lat)
    cr.stats = TimingStats(
        first_latency_ms=first_latency,
        mean_latency_ms=float(lat.mean()),
        min_latency_ms=float(lat_sorted[0]),
        max_latency_ms=float(lat_sorted[-1]),
        p50_latency_ms=percentile(lat_sorted, 50.0),
        p99_latency_ms=percentile(lat_sorted, 99.0),
        iterations=len(all_latencies),
    )
    cr.success = True
    return cr


# ──────────────── pretty print ────────────────

def print_stats(label: str, s: TimingStats) -> None:
    print(f"\n  -- {label} --")
    print(f"  Iterations : {s.iterations}")
    print(f"  TTFT       : {s.first_latency_ms:.2f} ms")
    print(f"  TPOT (avg) : {s.mean_latency_ms:.2f} ms")
    print(f"  Latency min: {s.min_latency_ms:.2f} ms")
    print(f"  Latency max: {s.max_latency_ms:.2f} ms")
    print(f"  Latency p50: {s.p50_latency_ms:.2f} ms")
    print(f"  Latency p99: {s.p99_latency_ms:.2f} ms")


# ──────────────── main ────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="EmbeddingGemma-300m Benchmark (raw OpenVINO Python API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python embedding_gemma_benchmark.py models/int4 models/fp32
  python embedding_gemma_benchmark.py models/int4 models/fp32 --warmup 3 --iters 10
  python embedding_gemma_benchmark.py models/int4 models/fp32 --dataset test_dataset.tsv
""",
    )
    parser.add_argument("int4_dir", help="Path to INT4 model directory")
    parser.add_argument("fp32_dir", help="Path to FP32 model directory")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument("--iters", type=int, default=10, help="Benchmark iterations (default: 10)")
    parser.add_argument("--dataset", type=str, default="", help="TSV dataset file (optional)")
    args = parser.parse_args()

    # SentenceTransformer-style prompts for EmbeddingGemma
    query_prompt = "task: search result | query: "
    doc_prompt = "title: none | text: "

    # Load test scenarios
    if args.dataset:
        scenarios = load_tsv_dataset(args.dataset)
        dataset_source = args.dataset
    else:
        scenarios = get_builtin_scenarios()
        dataset_source = "built-in (10 scenarios)"

    print("=" * 64)
    print("  EmbeddingGemma-300m Benchmark  (raw OV Python API)")
    print("=" * 64)
    print(f"  INT4 model  : {args.int4_dir}")
    print(f"  FP32 model  : {args.fp32_dir}")
    print(f"  Warmup      : {args.warmup} iterations")
    print(f"  Benchmark   : {args.iters} iterations")
    print(f"  Scenarios   : {len(scenarios)}")
    print(f"  Dataset     : {dataset_source}")
    print("=" * 64)

    devices = ["CPU", "GPU"]
    all_results: List[ConfigResult] = []

    for device in devices:
        for precision, model_dir in [("FP32", args.fp32_dir), ("INT4", args.int4_dir)]:
            label = f"{precision} / {device}"
            print(f"\n{'=' * 64}")
            print(f"  Testing: {label}  ({len(scenarios)} scenarios x {args.iters} iters)")
            print("=" * 64)

            cr = ConfigResult(label=label)
            try:
                cr = run_all_scenarios(
                    model_dir, device, query_prompt, doc_prompt,
                    scenarios, args.warmup, args.iters,
                )
                cr.label = label

                print_stats(label, cr.stats)

                # Per-scenario ranking results
                print(f"\n  -- Recall@1: {label} --")
                for si, (sc, sr) in enumerate(zip(scenarios, cr.scenario_results)):
                    status = "OK" if sr.correct else "MISS"
                    q_short = sc.query[:50]
                    print(
                        f"  [{si:2d}] {status:>4s}  predicted={sr.predicted_top_idx}"
                        f" expected={sc.expected_top_idx}"
                        f"  score={sr.top_score:.4f}"
                        f'  Q: "{q_short}"'
                    )
                pct = 100.0 * cr.correct_count / cr.total_scenarios if cr.total_scenarios else 0.0
                print(f"  Recall@1 = {cr.correct_count} / {cr.total_scenarios} ({pct:.1f}%)")

            except Exception as ex:
                print(f"  [SKIP] {label}: {ex}")

            all_results.append(cr)

    # -- Cross-precision accuracy comparison --
    print(f"\n{'=' * 64}")
    print("  Embedding Accuracy: INT4 vs FP32  (per-scenario cosine sim)")
    print("=" * 64)

    for device in devices:
        fp32_cr: Optional[ConfigResult] = None
        int4_cr: Optional[ConfigResult] = None
        for cr in all_results:
            if cr.label == f"FP32 / {device}" and cr.success:
                fp32_cr = cr
            if cr.label == f"INT4 / {device}" and cr.success:
                int4_cr = cr
        if not fp32_cr or not int4_cr:
            continue

        print(f"\n  -- INT4 vs FP32 on {device} --")
        sum_q_cos = 0.0
        sum_d_cos = 0.0
        n_scenarios = min(len(fp32_cr.scenario_results), len(int4_cr.scenario_results))
        for si in range(n_scenarios):
            fp32_sr = fp32_cr.scenario_results[si]
            int4_sr = int4_cr.scenario_results[si]

            q_cos = cosine_similarity(fp32_sr.embeddings.query_embedding, int4_sr.embeddings.query_embedding)
            sum_q_cos += q_cos

            nd = min(len(fp32_sr.embeddings.doc_embeddings), len(int4_sr.embeddings.doc_embeddings))
            scenario_doc_cos = 0.0
            for di in range(nd):
                scenario_doc_cos += cosine_similarity(
                    fp32_sr.embeddings.doc_embeddings[di],
                    int4_sr.embeddings.doc_embeddings[di],
                )
            if nd > 0:
                scenario_doc_cos /= nd
            sum_d_cos += scenario_doc_cos

            print(f"  [{si:2d}] query_cos={q_cos:.6f}  doc_cos={scenario_doc_cos:.6f}")

        if n_scenarios > 0:
            print(f"  Mean query cosine sim : {sum_q_cos / n_scenarios:.6f}")
            print(f"  Mean doc cosine sim   : {sum_d_cos / n_scenarios:.6f}")

    # -- Summary Table --
    print(f"\n{'=' * 64}")
    print("  Performance & Accuracy Summary")
    print("=" * 64)
    print(
        f"{'Config':<18s}{'TTFT(ms)':<12s}{'TPOT(ms)':<12s}"
        f"{'p50(ms)':<12s}{'p99(ms)':<12s}{'Recall@1':<14s}"
    )
    print("-" * 80)

    for cr in all_results:
        if not cr.success:
            print(f"{cr.label:<18s}  SKIPPED")
            continue
        pct = 100.0 * cr.correct_count / cr.total_scenarios if cr.total_scenarios else 0.0
        recall_str = f"{cr.correct_count}/{cr.total_scenarios} ({pct:.0f}%)"
        print(
            f"{cr.label:<18s}{cr.stats.first_latency_ms:<12.2f}{cr.stats.mean_latency_ms:<12.2f}"
            f"{cr.stats.p50_latency_ms:<12.2f}{cr.stats.p99_latency_ms:<12.2f}{recall_str:<14s}"
        )

    print("-" * 80)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
