// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// EmbeddingGemma-300m Benchmark (raw OpenVINO C++ API)
// Tests INT4 vs FP32 models on CPU and GPU, measuring:
//   - TTFT  (Time To First Token / first-inference latency)
//   - TPOT  (Time Per Output Token / per-embedding latency)
//   - Cosine-similarity accuracy between INT4 and FP32 embeddings
//   - Ranking accuracy (Recall@1) across multiple test scenarios
//
// Usage:
//   embedding_gemma_benchmark <INT4_MODEL_DIR> <FP32_MODEL_DIR> [warmup] [iters] [dataset.tsv]

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
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/genai/tokenizer.hpp"

// ──────────────────────── helpers ────────────────────────

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

struct TimingStats {
    double first_latency_ms = 0.0;   // TTFT  - first call latency
    double mean_latency_ms  = 0.0;   // TPOT  - average per-call latency
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

// ──────────────── mean pooling + L2 normalize ────────────────

// Mean pooling over token dim, respecting attention_mask
// token_embeddings: [batch, seq_len, hidden], attention_mask: [batch, seq_len]
static std::vector<std::vector<float>> mean_pool_and_normalize(
    const float* embeddings, const int64_t* attn_mask,
    size_t batch, size_t seq_len, size_t hidden)
{
    std::vector<std::vector<float>> result(batch, std::vector<float>(hidden, 0.0f));
    for (size_t b = 0; b < batch; ++b) {
        double count = 0.0;
        for (size_t s = 0; s < seq_len; ++s) {
            int64_t mask = attn_mask[b * seq_len + s];
            if (mask == 0) continue;
            count += 1.0;
            for (size_t h = 0; h < hidden; ++h) {
                result[b][h] += embeddings[b * seq_len * hidden + s * hidden + h];
            }
        }
        if (count < 1.0) count = 1.0;
        // mean
        for (size_t h = 0; h < hidden; ++h) {
            result[b][h] /= static_cast<float>(count);
        }
        // L2 normalize
        double norm = 0.0;
        for (size_t h = 0; h < hidden; ++h) {
            norm += static_cast<double>(result[b][h]) * result[b][h];
        }
        norm = std::sqrt(norm) + 1e-12;
        for (size_t h = 0; h < hidden; ++h) {
            result[b][h] /= static_cast<float>(norm);
        }
    }
    return result;
}

// ──────────────── embedding inference ────────────────

struct EmbeddingOutput {
    std::vector<float> query_embedding;
    std::vector<std::vector<float>> doc_embeddings;
};

// Check if model has a named input
static bool has_input(const ov::CompiledModel& cm, const std::string& name) {
    for (const auto& inp : cm.inputs()) {
        if (inp.get_any_name() == name) return true;
        for (const auto& n : inp.get_names()) {
            if (n == name) return true;
        }
    }
    return false;
}

// Run inference on a list of texts and return pooled+normalized embeddings
static std::vector<std::vector<float>> embed_texts(
    ov::InferRequest& request,
    ov::genai::Tokenizer& tokenizer,
    bool model_has_position_ids,
    const std::vector<std::string>& texts)
{
    ov::genai::TokenizedInputs encoded = tokenizer.encode(texts);

    request.set_tensor("input_ids", encoded.input_ids);
    request.set_tensor("attention_mask", encoded.attention_mask);

    // Create position_ids if needed
    if (model_has_position_ids) {
        auto shape = encoded.input_ids.get_shape();
        ov::Tensor position_ids(ov::element::i64, shape);
        int64_t* pos_data = position_ids.data<int64_t>();
        for (size_t b = 0; b < shape[0]; ++b) {
            for (size_t s = 0; s < shape[1]; ++s) {
                pos_data[b * shape[1] + s] = static_cast<int64_t>(s);
            }
        }
        request.set_tensor("position_ids", position_ids);
    }

    request.infer();

    // Get output: [batch, seq_len, hidden_size]
    ov::Tensor output = request.get_output_tensor();
    auto out_shape = output.get_shape();

    size_t batch   = out_shape[0];
    size_t seq_len = out_shape[1];
    size_t hidden  = out_shape[2];

    return mean_pool_and_normalize(
        output.data<float>(),
        encoded.attention_mask.data<int64_t>(),
        batch, seq_len, hidden);
}

// ──────────────── benchmark runner ────────────────

static std::pair<TimingStats, EmbeddingOutput> run_benchmark(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const std::string& query_prompt,
    const std::string& doc_prompt,
    const std::string& raw_query,
    const std::vector<std::string>& raw_documents,
    size_t warmup_iters,
    size_t bench_iters)
{
    // Prepare prompted texts
    std::string prompted_query = query_prompt + raw_query;
    std::vector<std::string> prompted_docs;
    for (const auto& d : raw_documents)
        prompted_docs.push_back(doc_prompt + d);

    std::cout << "  Loading model from: " << model_dir << " on " << device << " ..." << std::flush;
    auto t0 = Clock::now();

    ov::Core core;
    auto model = core.read_model(model_dir / "openvino_model.xml");
    auto compiled = core.compile_model(model, device);
    auto request = compiled.create_infer_request();

    ov::genai::Tokenizer tokenizer(model_dir);

    bool model_has_pos_ids = has_input(compiled, "position_ids");

    auto t1 = Clock::now();
    double load_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();
    std::cout << " loaded in " << std::fixed << std::setprecision(1) << load_ms << " ms"
              << " (position_ids: " << (model_has_pos_ids ? "yes" : "no") << ")" << std::endl;

    // ---- Warmup ----
    for (size_t i = 0; i < warmup_iters; ++i) {
        embed_texts(request, tokenizer, model_has_pos_ids, {prompted_query});
        embed_texts(request, tokenizer, model_has_pos_ids, prompted_docs);
    }

    // ---- Benchmark ----
    std::vector<double> query_latencies;
    query_latencies.reserve(bench_iters);
    std::vector<double> doc_latencies;
    doc_latencies.reserve(bench_iters);

    EmbeddingOutput output;

    for (size_t i = 0; i < bench_iters; ++i) {
        // Query embedding
        auto qs = Clock::now();
        auto q_result = embed_texts(request, tokenizer, model_has_pos_ids, {prompted_query});
        auto qe = Clock::now();
        query_latencies.push_back(std::chrono::duration_cast<Ms>(qe - qs).count());

        // Document embeddings
        auto ds = Clock::now();
        auto d_result = embed_texts(request, tokenizer, model_has_pos_ids, prompted_docs);
        auto de = Clock::now();
        doc_latencies.push_back(std::chrono::duration_cast<Ms>(de - ds).count());

        // Keep last iteration's output for accuracy comparison
        if (i == bench_iters - 1) {
            output.query_embedding = q_result[0];
            output.doc_embeddings  = d_result;
        }
    }

    // ---- Compute stats ----
    std::vector<double> all_latencies;
    all_latencies.reserve(query_latencies.size() + doc_latencies.size());
    all_latencies.insert(all_latencies.end(), query_latencies.begin(), query_latencies.end());
    all_latencies.insert(all_latencies.end(), doc_latencies.begin(), doc_latencies.end());

    TimingStats stats;
    stats.iterations       = all_latencies.size();
    stats.first_latency_ms = query_latencies.front();  // TTFT
    stats.mean_latency_ms  = std::accumulate(all_latencies.begin(), all_latencies.end(), 0.0) / all_latencies.size();
    std::sort(all_latencies.begin(), all_latencies.end());
    stats.min_latency_ms = all_latencies.front();
    stats.max_latency_ms = all_latencies.back();
    stats.p50_latency_ms = percentile(all_latencies, 50.0);
    stats.p99_latency_ms = percentile(all_latencies, 99.0);

    return {stats, output};
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

static void print_accuracy(const std::string& label,
                            const EmbeddingOutput& ref,
                            const EmbeddingOutput& test) {
    std::cout << "\n  -- Accuracy: " << label << " --\n";

    double q_cos = cosine_similarity(ref.query_embedding, test.query_embedding);
    std::cout << "  Query cosine similarity : " << std::fixed << std::setprecision(6) << q_cos << "\n";

    size_t n = std::min(ref.doc_embeddings.size(), test.doc_embeddings.size());
    double sum_cos = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double cs = cosine_similarity(ref.doc_embeddings[i], test.doc_embeddings[i]);
        std::cout << "  Doc[" << i << "] cosine similarity : " << cs << "\n";
        sum_cos += cs;
    }
    if (n > 0)
        std::cout << "  Mean doc cosine sim     : " << (sum_cos / n) << "\n";

    // Element-wise abs diff on query
    if (ref.query_embedding.size() == test.query_embedding.size()) {
        double max_diff = 0.0, sum_diff = 0.0;
        size_t cnt_above = 0;
        for (size_t i = 0; i < ref.query_embedding.size(); ++i) {
            double d = std::abs(static_cast<double>(ref.query_embedding[i]) - test.query_embedding[i]);
            max_diff = std::max(max_diff, d);
            sum_diff += d;
            if (d > 1e-3) ++cnt_above;
        }
        std::cout << "  Query max abs diff      : " << max_diff << "\n"
                  << "  Query mean abs diff     : " << (sum_diff / ref.query_embedding.size()) << "\n"
                  << "  Query elts |diff|>1e-3  : " << cnt_above
                  << " / " << ref.query_embedding.size() << "\n";
    }
}

static void print_ranking(const std::string& label,
                           const EmbeddingOutput& emb,
                           const std::vector<std::string>& documents) {
    std::cout << "\n  -- Ranking: " << label << " --\n";
    std::vector<std::pair<double, size_t>> scores;
    for (size_t i = 0; i < emb.doc_embeddings.size(); ++i) {
        double cs = cosine_similarity(emb.query_embedding, emb.doc_embeddings[i]);
        scores.push_back({cs, i});
    }
    std::sort(scores.begin(), scores.end(), [](auto& a, auto& b) { return a.first > b.first; });
    for (auto& [score, idx] : scores) {
        std::cout << "  [" << idx << "] score=" << std::fixed << std::setprecision(6) << score
                  << "  \"" << documents[idx].substr(0, 70) << "\"\n";
    }
}

// ──────────────── test scenario ────────────────

struct TestScenario {
    std::string query;
    std::vector<std::string> documents;
    int expected_top_idx;  // index of the best-matching document (-1 = unknown)
};

// Built-in test scenarios covering diverse domains
static std::vector<TestScenario> get_builtin_scenarios() {
    return {
        // 0 - Astronomy (original notebook example)
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
    // skip header
    if (!std::getline(fin, line)) return scenarios;

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::vector<std::string> fields;
        std::istringstream iss(line);
        std::string field;
        while (std::getline(iss, field, '\t')) {
            fields.push_back(field);
        }
        // need at least: query, expected_idx, doc_0, doc_1
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

// ──────────────── run all scenarios on one model config ────────────────

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

static ConfigResult run_all_scenarios(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const std::string& query_prompt,
    const std::string& doc_prompt,
    const std::vector<TestScenario>& scenarios,
    size_t warmup_iters,
    size_t bench_iters)
{
    ConfigResult cr;
    cr.total_scenarios = scenarios.size();

    std::cout << "  Loading model from: " << model_dir << " on " << device << " ..." << std::flush;
    auto t0 = Clock::now();

    ov::Core core;
    auto model = core.read_model(model_dir / "openvino_model.xml");
    auto compiled = core.compile_model(model, device);
    auto request = compiled.create_infer_request();
    ov::genai::Tokenizer tokenizer(model_dir);
    bool model_has_pos_ids = has_input(compiled, "position_ids");

    auto t1 = Clock::now();
    double load_ms = std::chrono::duration_cast<Ms>(t1 - t0).count();
    std::cout << " loaded in " << std::fixed << std::setprecision(1) << load_ms << " ms" << std::endl;

    // Warmup with first scenario
    {
        std::string wq = query_prompt + scenarios[0].query;
        std::vector<std::string> wd;
        for (auto& d : scenarios[0].documents) wd.push_back(doc_prompt + d);
        for (size_t i = 0; i < warmup_iters; ++i) {
            embed_texts(request, tokenizer, model_has_pos_ids, {wq});
            embed_texts(request, tokenizer, model_has_pos_ids, wd);
        }
    }

    // Benchmark: run all scenarios bench_iters times, collect latencies
    std::vector<double> all_latencies;
    double first_latency = 0.0;

    for (size_t iter = 0; iter < bench_iters; ++iter) {
        for (size_t si = 0; si < scenarios.size(); ++si) {
            const auto& sc = scenarios[si];
            std::string pq = query_prompt + sc.query;
            std::vector<std::string> pd;
            for (auto& d : sc.documents) pd.push_back(doc_prompt + d);

            auto qs = Clock::now();
            auto q_emb = embed_texts(request, tokenizer, model_has_pos_ids, {pq});
            auto qe = Clock::now();
            all_latencies.push_back(std::chrono::duration_cast<Ms>(qe - qs).count());
            if (iter == 0 && si == 0) first_latency = all_latencies.back();

            auto ds = Clock::now();
            auto d_emb = embed_texts(request, tokenizer, model_has_pos_ids, pd);
            auto de = Clock::now();
            all_latencies.push_back(std::chrono::duration_cast<Ms>(de - ds).count());

            // Keep last iteration results
            if (iter == bench_iters - 1) {
                ScenarioResult sr;
                sr.embeddings.query_embedding = q_emb[0];
                sr.embeddings.doc_embeddings = d_emb;

                // Find best match
                double best = -2.0;
                int best_idx = -1;
                for (size_t di = 0; di < d_emb.size(); ++di) {
                    double cs = cosine_similarity(q_emb[0], d_emb[di]);
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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <INT4_MODEL_DIR> <FP32_MODEL_DIR> [warmup] [iters] [dataset.tsv]\n"
                  << "\nExample:\n"
                  << "  " << argv[0]
                  << " models/int4 models/fp32 3 10\n"
                  << "  " << argv[0]
                  << " models/int4 models/fp32 3 10 test_dataset.tsv\n";
        return EXIT_FAILURE;
    }

    std::filesystem::path int4_dir = argv[1];
    std::filesystem::path fp32_dir = argv[2];
    size_t warmup = (argc > 3) ? std::stoul(argv[3]) : 3;
    size_t iters  = (argc > 4) ? std::stoul(argv[4]) : 10;
    std::string dataset_path = (argc > 5) ? argv[5] : "";

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
              << "  EmbeddingGemma-300m Benchmark  (raw OV C++ API)\n"
              << "================================================================\n"
              << "  INT4 model  : " << int4_dir << "\n"
              << "  FP32 model  : " << fp32_dir << "\n"
              << "  Warmup      : " << warmup << " iterations\n"
              << "  Benchmark   : " << iters  << " iterations\n"
              << "  Scenarios   : " << scenarios.size() << "\n"
              << "  Dataset     : " << dataset_source << "\n"
              << "================================================================\n";

    // Devices to test
    std::vector<std::string> devices = {"CPU", "GPU"};
    std::vector<ConfigResult> all_results;

    for (const auto& device : devices) {
        for (const auto& [precision, model_dir] : std::vector<std::pair<std::string, std::filesystem::path>>{
                 {"FP32", fp32_dir}, {"INT4", int4_dir}}) {
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
    }

    // -- Cross-precision accuracy comparison (embedding-level) --
    std::cout << "\n================================================================\n"
              << "  Embedding Accuracy: INT4 vs FP32  (per-scenario cosine sim)\n"
              << "================================================================\n";

    for (const auto& device : devices) {
        ConfigResult* fp32_cr = nullptr;
        ConfigResult* int4_cr = nullptr;
        for (auto& cr : all_results) {
            if (cr.label == "FP32 / " + device && cr.success) fp32_cr = &cr;
            if (cr.label == "INT4 / " + device && cr.success) int4_cr = &cr;
        }
        if (!fp32_cr || !int4_cr) continue;

        std::cout << "\n  -- INT4 vs FP32 on " << device << " --\n";
        double sum_q_cos = 0.0, sum_d_cos = 0.0;
        size_t n_scenarios = std::min(fp32_cr->scenario_results.size(), int4_cr->scenario_results.size());
        for (size_t si = 0; si < n_scenarios; ++si) {
            double q_cos = cosine_similarity(
                fp32_cr->scenario_results[si].embeddings.query_embedding,
                int4_cr->scenario_results[si].embeddings.query_embedding);
            sum_q_cos += q_cos;

            size_t nd = std::min(fp32_cr->scenario_results[si].embeddings.doc_embeddings.size(),
                                 int4_cr->scenario_results[si].embeddings.doc_embeddings.size());
            double scenario_doc_cos = 0.0;
            for (size_t di = 0; di < nd; ++di) {
                scenario_doc_cos += cosine_similarity(
                    fp32_cr->scenario_results[si].embeddings.doc_embeddings[di],
                    int4_cr->scenario_results[si].embeddings.doc_embeddings[di]);
            }
            if (nd > 0) scenario_doc_cos /= nd;
            sum_d_cos += scenario_doc_cos;

            std::cout << "  [" << std::setw(2) << si << "] query_cos=" << std::fixed << std::setprecision(6) << q_cos
                      << "  doc_cos=" << scenario_doc_cos << "\n";
        }
        if (n_scenarios > 0) {
            std::cout << "  Mean query cosine sim : " << (sum_q_cos / n_scenarios) << "\n"
                      << "  Mean doc cosine sim   : " << (sum_d_cos / n_scenarios) << "\n";
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
