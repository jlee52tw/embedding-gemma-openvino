// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// GenAI TextEmbeddingPipeline Benchmark for EmbeddingGemma-300m
// Uses ov::genai::TextEmbeddingPipeline API (high-level GenAI SDK)
// Tests INT4 vs FP32 models on CPU and GPU, measuring:
//   - TTFT  (Time To First Token / first-inference latency)
//   - TPOT  (Time Per Output Token / per-embedding latency)
//   - Ranking accuracy (Recall@1) across multiple test scenarios
//
// NOTE: TextEmbeddingPipeline currently does NOT set position_ids,
//       which is required by EmbeddingGemma. This test will report
//       SKIP for models that need position_ids. Once the GenAI SDK
//       adds position_ids support, this test will work automatically.
//
// Usage:
//   genai_embedding_test <MODEL_DIR_1> [MODEL_DIR_2] [device] [warmup] [iters] [dataset.tsv]
//
// Examples:
//   genai_embedding_test models/fp32
//   genai_embedding_test models/int4 models/fp32 CPU 2 5
//   genai_embedding_test models/int4 models/fp32 GPU 2 5 test_dataset.tsv

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include "openvino/genai/rag/text_embedding_pipeline.hpp"

// ──────────────────────── helpers ────────────────────────

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct TimingStats {
    double first_latency_ms = 0.0;
    double mean_latency_ms  = 0.0;
    double min_latency_ms   = 0.0;
    double max_latency_ms   = 0.0;
    double p50_latency_ms   = 0.0;
    double p99_latency_ms   = 0.0;
    size_t iterations       = 0;
};

static double cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na  += static_cast<double>(a[i]) * a[i];
        nb  += static_cast<double>(b[i]) * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

static double percentile(std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    double idx = p / 100.0 * (sorted_v.size() - 1);
    size_t lo  = static_cast<size_t>(idx);
    size_t hi  = std::min(lo + 1, sorted_v.size() - 1);
    double frac = idx - lo;
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac;
}

// ──────────────── test scenario ────────────────

struct TestScenario {
    std::string query;
    std::vector<std::string> documents;
    int expected_top_idx;  // index of the best-matching document (-1 = unknown)
};

static std::vector<TestScenario> get_builtin_scenarios() {
    return {
        // 0 - Astronomy
        {"Which planet is known as the Red Planet?",
         {"Venus is often called Earth's twin because of its similar size and proximity.",
          "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
          "Jupiter, the largest planet in our solar system, has a prominent red spot.",
          "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."},
         1},

        // 1 - History
        {"Who was the first person to walk on the moon?",
         {"Yuri Gagarin was the first human to journey into outer space in 1961.",
          "Neil Armstrong became the first person to walk on the moon on July 20, 1969.",
          "Buzz Aldrin was the second person to walk on the lunar surface during Apollo 11.",
          "John Glenn was the first American to orbit the Earth in 1962."},
         1},

        // 2 - Computer Science
        {"What is a neural network?",
         {"A database is an organized collection of structured information stored electronically.",
          "A compiler translates source code written in a programming language into machine code.",
          "A neural network is a computing system inspired by biological neural networks in the brain, consisting of interconnected nodes that process information.",
          "An operating system manages computer hardware and provides services for applications."},
         2},

        // 3 - Biology
        {"How do antibiotics work?",
         {"Vaccines stimulate the immune system to produce antibodies against specific pathogens.",
          "Antibiotics work by killing bacteria or preventing them from reproducing, either by disrupting cell wall synthesis, protein production, or DNA replication.",
          "Vitamins are organic compounds that are essential nutrients required in small quantities.",
          "Hormones are chemical messengers that travel through the bloodstream to tissues and organs."},
         1},

        // 4 - Geography
        {"What is the longest river in the world?",
         {"Mount Everest, standing at 8,849 meters, is the highest peak above sea level.",
          "The Sahara Desert is the largest hot desert in the world, covering most of North Africa.",
          "The Nile River, stretching approximately 6,650 kilometers through northeastern Africa, is considered the longest river in the world.",
          "The Pacific Ocean is the largest and deepest ocean, covering more than 30 percent of Earth's surface."},
         2},

        // 5 - Physics
        {"What is dark matter?",
         {"Antimatter is made up of particles with the same mass but opposite charge as normal matter.",
          "Dark energy is a hypothetical form of energy that permeates all of space and accelerates expansion.",
          "Gravity is a fundamental force that attracts objects with mass toward each other.",
          "Dark matter is an invisible form of matter that does not emit, absorb, or reflect light, detected only through its gravitational effects on visible matter."},
         3},

        // 6 - Music
        {"Who composed the Ninth Symphony?",
         {"Mozart composed over 600 works including symphonies, operas, and chamber music.",
          "Bach is known for his intricate compositions including the Brandenburg Concertos.",
          "Ludwig van Beethoven composed his famous Ninth Symphony, which includes the Ode to Joy choral finale.",
          "Tchaikovsky composed several famous ballets including Swan Lake and The Nutcracker."},
         2},

        // 7 - Medicine
        {"What are the symptoms of a heart attack?",
         {"A stroke occurs when blood supply to part of the brain is interrupted or reduced.",
          "Symptoms of a heart attack include chest pain or pressure, shortness of breath, pain radiating to the arm or jaw, nausea, and cold sweat.",
          "Pneumonia is an infection that inflames the air sacs in one or both lungs.",
          "Appendicitis typically starts with pain around the navel that shifts to the lower right abdomen."},
         1},

        // 8 - Technology
        {"How does 5G technology differ from 4G?",
         {"Wi-Fi 6 offers improved speed, latency, and capacity compared to previous Wi-Fi generations.",
          "Bluetooth 5.0 provides longer range and faster data transfer than Bluetooth 4.2.",
          "Fiber optic cables transmit data as pulses of light through glass or plastic strands.",
          "5G technology offers significantly faster data speeds, lower latency, and greater capacity than 4G by using higher frequency bands and advanced antenna technology."},
         3},

        // 9 - Environment
        {"What causes ocean acidification?",
         {"Coral bleaching occurs when corals expel their symbiotic algae due to stress from warm water.",
          "Overfishing depletes fish stocks faster than they can replenish through natural reproduction.",
          "Ocean acidification is caused by the absorption of excess carbon dioxide from the atmosphere, which reacts with seawater to form carbonic acid, lowering the ocean's pH.",
          "Plastic pollution in the oceans harms marine wildlife through ingestion and entanglement."},
         2},
    };
}

// Load scenarios from a TSV file (tab-separated):
//   query \t expected_idx \t doc_0 \t doc_1 \t doc_2 \t doc_3 [...]
// First line is a header row and is skipped.
static std::vector<TestScenario> load_tsv_dataset(const std::filesystem::path& tsv_path) {
    std::vector<TestScenario> scenarios;
    std::ifstream fin(tsv_path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open dataset file: " + tsv_path.string());
    }

    std::string line;
    if (!std::getline(fin, line)) return scenarios;  // skip header

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, '\t')) {
            fields.push_back(field);
        }
        if (fields.size() < 4) continue;

        TestScenario sc;
        sc.query = fields[0];
        sc.expected_top_idx = std::stoi(fields[1]);
        for (size_t i = 2; i < fields.size(); ++i) {
            if (!fields[i].empty())
                sc.documents.push_back(fields[i]);
        }
        scenarios.push_back(std::move(sc));
    }
    return scenarios;
}

// ──────────────── pretty print ────────────────

static void print_stats(const std::string& label, const TimingStats& s) {
    std::cout << "\n  -- " << label << " --\n"
              << "  Iterations : " << s.iterations << "\n"
              << "  TTFT       : " << std::fixed << std::setprecision(2) << s.first_latency_ms << " ms\n"
              << "  TPOT (avg) : " << s.mean_latency_ms << " ms\n"
              << "  Latency min: " << s.min_latency_ms << " ms\n"
              << "  Latency max: " << s.max_latency_ms << " ms\n"
              << "  Latency p50: " << s.p50_latency_ms << " ms\n"
              << "  Latency p99: " << s.p99_latency_ms << " ms\n";
}

// ──────────────── embedding output ────────────────

struct EmbeddingOutput {
    std::vector<float> query_embedding;
    std::vector<std::vector<float>> doc_embeddings;
};

struct ScenarioResult {
    EmbeddingOutput embeddings;
    int predicted_top_idx = -1;
    double top_score = 0.0;
    bool correct = false;
};

struct ConfigResult {
    std::string label;
    TimingStats stats;
    std::vector<ScenarioResult> scenario_results;
    size_t correct_count = 0;
    size_t total_scenarios = 0;
    bool success = false;
};

// ──────────────── GenAI TextEmbeddingPipeline runner ────────────────

static ConfigResult run_all_scenarios(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const std::string& query_instruction,
    const std::string& embed_instruction,
    const std::vector<TestScenario>& scenarios,
    size_t warmup_iters,
    size_t bench_iters)
{
    ConfigResult cr;
    cr.total_scenarios = scenarios.size();

    std::cout << "  Loading GenAI TextEmbeddingPipeline from: " << model_dir
              << " on " << device << " ..." << std::flush;
    auto t0 = Clock::now();

    ov::genai::TextEmbeddingPipeline::Config config;
    config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
    config.normalize = true;
    config.query_instruction = query_instruction;
    config.embed_instruction = embed_instruction;

    ov::genai::TextEmbeddingPipeline pipeline(model_dir, device, config);

    auto t1 = Clock::now();
    double load_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();
    std::cout << " loaded in " << std::fixed << std::setprecision(1) << load_ms << " ms" << std::endl;

    // Helper: extract vector<float> from EmbeddingResult variant
    auto get_float_vec = [](const ov::genai::EmbeddingResult& r) -> std::vector<float> {
        return std::get<std::vector<float>>(r);
    };
    auto get_float_vecs = [](const ov::genai::EmbeddingResults& r) -> std::vector<std::vector<float>> {
        return std::get<std::vector<std::vector<float>>>(r);
    };

    // Warmup
    for (size_t i = 0; i < warmup_iters; ++i) {
        pipeline.embed_query(scenarios[0].query);
        pipeline.embed_documents(scenarios[0].documents);
    }

    // Benchmark
    std::vector<double> all_latencies;
    double first_latency = 0.0;

    for (size_t iter = 0; iter < bench_iters; ++iter) {
        for (size_t si = 0; si < scenarios.size(); ++si) {
            const auto& sc = scenarios[si];

            auto qs = Clock::now();
            auto q_result = get_float_vec(pipeline.embed_query(sc.query));
            auto qe = Clock::now();
            all_latencies.push_back(std::chrono::duration_cast<Ms>(qe - qs).count());
            if (iter == 0 && si == 0) first_latency = all_latencies.back();

            auto ds = Clock::now();
            auto d_result = get_float_vecs(pipeline.embed_documents(sc.documents));
            auto de = Clock::now();
            all_latencies.push_back(std::chrono::duration_cast<Ms>(de - ds).count());

            // Keep last iteration results
            if (iter == bench_iters - 1) {
                ScenarioResult sr;
                sr.embeddings.query_embedding = q_result;
                sr.embeddings.doc_embeddings = d_result;

                double best = -2.0;
                int best_idx = -1;
                for (size_t di = 0; di < d_result.size(); ++di) {
                    double cs = cosine_similarity(q_result, d_result[di]);
                    if (cs > best) { best = cs; best_idx = static_cast<int>(di); }
                }
                sr.predicted_top_idx = best_idx;
                sr.top_score = best;
                sr.correct = (best_idx == sc.expected_top_idx);
                if (sr.correct) cr.correct_count++;
                cr.scenario_results.push_back(std::move(sr));
            }
        }
    }

    // Compute stats
    cr.stats.iterations = all_latencies.size();
    cr.stats.first_latency_ms = first_latency;
    cr.stats.mean_latency_ms = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / all_latencies.size();
    std::sort(all_latencies.begin(), all_latencies.end());
    cr.stats.min_latency_ms = all_latencies.front();
    cr.stats.max_latency_ms = all_latencies.back();
    cr.stats.p50_latency_ms = percentile(all_latencies, 50.0);
    cr.stats.p99_latency_ms = percentile(all_latencies, 99.0);
    cr.success = true;

    return cr;
}

// ──────────────── main ────────────────

int main(int argc, char* argv[]) try {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <MODEL_DIR_1> [MODEL_DIR_2] [device] [warmup] [iters] [dataset.tsv]\n"
                  << "\nExamples:\n"
                  << "  " << argv[0] << " models/fp32\n"
                  << "  " << argv[0] << " models/int4 models/fp32 CPU 2 5\n"
                  << "  " << argv[0] << " models/int4 models/fp32 GPU 2 5 test_dataset.tsv\n";
        return EXIT_FAILURE;
    }

    // Parse arguments
    std::vector<std::pair<std::string, std::filesystem::path>> models;
    std::string device = "CPU";
    size_t warmup = 2;
    size_t iters  = 5;
    std::string dataset_path;

    // Collect model dirs (arguments that are valid directories)
    int arg_idx = 1;
    while (arg_idx < argc && std::filesystem::is_directory(argv[arg_idx])) {
        std::filesystem::path p = argv[arg_idx];
        // Auto-detect precision from path name
        std::string dirname = p.filename().string();
        std::string precision = "MODEL";
        std::string lower_dir;
        for (char c : dirname) lower_dir += static_cast<char>(std::tolower(c));
        if (lower_dir.find("int4") != std::string::npos || lower_dir.find("int8") != std::string::npos)
            precision = "INT4";
        else if (lower_dir.find("fp32") != std::string::npos || lower_dir.find("float32") != std::string::npos)
            precision = "FP32";
        else if (lower_dir.find("fp16") != std::string::npos || lower_dir.find("float16") != std::string::npos)
            precision = "FP16";
        models.push_back({precision, p});
        arg_idx++;
    }

    // Remaining args: device, warmup, iters, dataset
    if (arg_idx < argc) device = argv[arg_idx++];
    if (arg_idx < argc) warmup = std::stoul(argv[arg_idx++]);
    if (arg_idx < argc) iters  = std::stoul(argv[arg_idx++]);
    if (arg_idx < argc) dataset_path = argv[arg_idx++];

    // SentenceTransformer-style prompts for EmbeddingGemma
    const std::string query_prompt = "task: search result | query: ";
    const std::string doc_prompt   = "title: none | text: ";

    // Load test scenarios
    std::vector<TestScenario> scenarios;
    std::string dataset_source;
    if (!dataset_path.empty()) {
        scenarios = load_tsv_dataset(dataset_path);
        dataset_source = dataset_path;
    } else {
        scenarios = get_builtin_scenarios();
        dataset_source = "built-in (10 scenarios)";
    }

    std::cout << "================================================================\n"
              << "  GenAI TextEmbeddingPipeline Benchmark\n"
              << "================================================================\n"
              << "  Models      : " << models.size() << "\n";
    for (const auto& [prec, path] : models) {
        std::cout << "    " << prec << " : " << path << "\n";
    }
    std::cout << "  Device      : " << device << "\n"
              << "  Warmup      : " << warmup << " iterations\n"
              << "  Benchmark   : " << iters  << " iterations\n"
              << "  Scenarios   : " << scenarios.size() << "\n"
              << "  Dataset     : " << dataset_source << "\n"
              << "================================================================\n";

    std::vector<ConfigResult> all_results;

    for (const auto& [precision, model_dir] : models) {
        std::string label = precision + " / " + device;
        std::cout << "\n================================================================\n"
                  << "  Testing: " << label
                  << "  (" << scenarios.size() << " scenarios x " << iters << " iters)\n"
                  << "================================================================\n";

        ConfigResult cr;
        cr.label = label;
        try {
            cr = run_all_scenarios(model_dir, device, query_prompt, doc_prompt,
                                   scenarios, warmup, iters);
            cr.label = label;

            print_stats(label, cr.stats);

            // Print per-scenario ranking results
            std::cout << "\n  -- Recall@1: " << label << " --\n";
            for (size_t si = 0; si < scenarios.size(); ++si) {
                const auto& sc = scenarios[si];
                const auto& sr = cr.scenario_results[si];
                std::string status = sr.correct ? "OK" : "MISS";
                std::cout << "  [" << std::setw(2) << si << "] " << std::setw(4) << status
                          << "  predicted=" << sr.predicted_top_idx
                          << " expected=" << sc.expected_top_idx
                          << "  score=" << std::fixed << std::setprecision(4) << sr.top_score
                          << "  Q: \"" << sc.query.substr(0, 50) << "\"\n";
            }
            std::cout << "  Recall@1 = " << cr.correct_count << " / " << cr.total_scenarios
                      << " (" << std::fixed << std::setprecision(1)
                      << (100.0 * cr.correct_count / cr.total_scenarios) << "%)\n";
        } catch (const std::exception& ex) {
            std::cerr << "  [SKIP] " << label << ": " << ex.what() << "\n";
        }
        all_results.push_back(std::move(cr));
    }

    // -- Cross-model accuracy comparison --
    if (all_results.size() >= 2) {
        std::cout << "\n================================================================\n"
                  << "  Embedding Accuracy: Cross-Model Comparison (cosine sim)\n"
                  << "================================================================\n";

        auto& ref = all_results[0];
        for (size_t mi = 1; mi < all_results.size(); ++mi) {
            auto& test = all_results[mi];
            if (!ref.success || !test.success) continue;

            std::cout << "\n  -- " << test.label << " vs " << ref.label << " --\n";
            double sum_q_cos = 0.0, sum_d_cos = 0.0;
            size_t n = std::min(ref.scenario_results.size(), test.scenario_results.size());
            for (size_t si = 0; si < n; ++si) {
                double q_cos = cosine_similarity(
                    ref.scenario_results[si].embeddings.query_embedding,
                    test.scenario_results[si].embeddings.query_embedding);
                sum_q_cos += q_cos;

                size_t nd = std::min(ref.scenario_results[si].embeddings.doc_embeddings.size(),
                                     test.scenario_results[si].embeddings.doc_embeddings.size());
                double doc_cos = 0.0;
                for (size_t di = 0; di < nd; ++di) {
                    doc_cos += cosine_similarity(
                        ref.scenario_results[si].embeddings.doc_embeddings[di],
                        test.scenario_results[si].embeddings.doc_embeddings[di]);
                }
                if (nd > 0) doc_cos /= nd;
                sum_d_cos += doc_cos;

                std::cout << "  [" << std::setw(2) << si << "] query_cos=" << std::fixed << std::setprecision(6) << q_cos
                          << "  doc_cos=" << doc_cos << "\n";
            }
            if (n > 0) {
                std::cout << "  Mean query cosine sim : " << (sum_q_cos / n) << "\n"
                          << "  Mean doc cosine sim   : " << (sum_d_cos / n) << "\n";
            }
        }
    }

    // -- Summary Table --
    std::cout << "\n================================================================\n"
              << "  Performance & Accuracy Summary\n"
              << "================================================================\n"
              << std::left
              << std::setw(18) << "Config"
              << std::setw(12) << "TTFT(ms)"
              << std::setw(12) << "TPOT(ms)"
              << std::setw(12) << "p50(ms)"
              << std::setw(12) << "p99(ms)"
              << std::setw(14) << "Recall@1"
              << "\n"
              << std::string(80, '-') << "\n";

    for (const auto& cr : all_results) {
        if (!cr.success) {
            std::cout << std::setw(18) << cr.label << "  SKIPPED\n";
            continue;
        }
        std::ostringstream recall_str;
        recall_str << cr.correct_count << "/" << cr.total_scenarios
                   << " (" << std::fixed << std::setprecision(0)
                   << (100.0 * cr.correct_count / cr.total_scenarios) << "%)";
        std::cout << std::setw(18) << cr.label
                  << std::setw(12) << std::fixed << std::setprecision(2) << cr.stats.first_latency_ms
                  << std::setw(12) << cr.stats.mean_latency_ms
                  << std::setw(12) << cr.stats.p50_latency_ms
                  << std::setw(12) << cr.stats.p99_latency_ms
                  << std::setw(14) << recall_str.str()
                  << "\n";
    }
    std::cout << std::string(80, '-') << "\n";

    std::cout << "\nDone.\n";
    return EXIT_SUCCESS;

} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
