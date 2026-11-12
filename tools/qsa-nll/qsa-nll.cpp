// qsa-nll: teacher-forced full-vocabulary NLL at selected positions (R3 quality gate for the RDNA2 QSA prefill kernel).
//
// Not a benchmark framework: loads the model with the normal llama-server flags (parsed by common_params_parse), prefills
// token ids in n_ubatch chunks requesting logits ONLY at the selected positions, and for each of them computes
//   NLL = logsumexp(logits) - logits[target]      (double precision, whole vocabulary)
// then discards the logits vector.
//
// Modes (env, so that all normal llama flags stay untouched):
//   QSA_NLL_MODE=tok  QSA_NLL_IN=text.txt  QSA_NLL_OUT=ids.txt    tokenise (vocab only, special tokens parsed, no BOS)
//   QSA_NLL_MODE=run  QSA_NLL_IN=spec.txt  QSA_NLL_OUT=out.json   score
// spec.txt: repeated records
//   REC <name> <n_tokens> <n_pos>\n<token ids...>\n<positions...>
// position p (1 <= p < n_tokens) means: score token[p] given tokens[0..p-1]; its logits are those of batch index p-1.
#include "arg.h"
#include "common.h"
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::string envs(const char * k) { const char * v = getenv(k); return v ? v : ""; }

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }
    const std::string mode = envs("QSA_NLL_MODE"), in = envs("QSA_NLL_IN"), outp = envs("QSA_NLL_OUT");
    if (in.empty() || outp.empty()) { fprintf(stderr, "QSA_NLL_IN / QSA_NLL_OUT required\n"); return 2; }

    llama_backend_init();
    if (mode == "tok") {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        llama_model * m = llama_model_load_from_file(params.model.path.c_str(), mp);
        if (!m) { fprintf(stderr, "vocab load failed\n"); return 3; }
        const llama_vocab * vocab = llama_model_get_vocab(m);
        std::ifstream f(in, std::ios::binary);
        std::stringstream ss; ss << f.rdbuf();
        const std::string text = ss.str();
        std::vector<llama_token> toks(text.size() + 16);
        const int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true);
        if (n < 0) { fprintf(stderr, "tokenize failed %d\n", n); return 4; }
        std::ofstream o(outp);
        for (int i = 0; i < n; i++) o << toks[i] << (i + 1 < n ? " " : "\n");
        return 0;
    }

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    llama_context * ctx = init->context();
    if (!model || !ctx) { fprintf(stderr, "model/context init failed\n"); return 3; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int chunk = (int) llama_n_ubatch(ctx);

    std::ifstream f(in);
    std::ofstream o(outp);
    o << "{\"n_vocab\": " << n_vocab << ", \"n_ubatch\": " << chunk << ", \"n_ctx\": " << llama_n_ctx(ctx) << ", \"records\": [\n";
    std::string tag;
    bool first_rec = true;
    llama_batch batch = llama_batch_init(chunk, 0, 1);
    while (f >> tag) {
        if (tag != "REC") { fprintf(stderr, "bad spec\n"); return 5; }
        std::string name; int nt, np;
        f >> name >> nt >> np;
        std::vector<llama_token> tok(nt);
        for (auto & t : tok) f >> t;
        std::vector<int> pos(np);
        for (auto & p : pos) f >> p;
        std::vector<char> want(nt, 0);      // want[i]: logits needed at batch index i (predicts token i+1)
        for (int p : pos) { if (p < 1 || p >= nt) { fprintf(stderr, "bad pos %d\n", p); return 5; } want[p - 1] = 1; }

        llama_memory_clear(llama_get_memory(ctx), true);
        std::vector<double> nll(nt, NAN);
        const auto t0 = std::chrono::steady_clock::now();
        bool err = false;
        for (int s = 0; s < nt && !err; s += chunk) {
            const int n = std::min(chunk, nt - s);
            batch.n_tokens = n;
            for (int i = 0; i < n; i++) {
                batch.token[i] = tok[s + i]; batch.pos[i] = s + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                batch.logits[i] = want[s + i];
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "llama_decode failed\n"); err = true; break; }
            for (int i = 0; i < n; i++) {
                if (!want[s + i]) continue;
                const float * lg = llama_get_logits_ith(ctx, i);
                if (!lg) { fprintf(stderr, "no logits at %d\n", s + i); err = true; break; }
                double mx = -INFINITY;
                for (int v = 0; v < n_vocab; v++) mx = std::max(mx, (double) lg[v]);
                double se = 0;
                for (int v = 0; v < n_vocab; v++) se += std::exp((double) lg[v] - mx);
                nll[s + i + 1] = (mx + std::log(se)) - (double) lg[tok[s + i + 1]];
            }
        }
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!first_rec) o << ",\n";
        first_rec = false;
        o.precision(9);
        o << "{\"name\": \"" << name << "\", \"n_tokens\": " << nt << ", \"elapsed_s\": " << el << ", \"error\": " << (err ? "true" : "false")
          << ", \"positions\": [";
        for (int i = 0; i < np; i++) o << pos[i] << (i + 1 < np ? "," : "");
        o << "], \"target_ids\": [";
        for (int i = 0; i < np; i++) o << tok[pos[i]] << (i + 1 < np ? "," : "");
        o << "], \"nll\": [";
        for (int i = 0; i < np; i++) { if (std::isfinite(nll[pos[i]])) o << nll[pos[i]]; else o << "null"; o << (i + 1 < np ? "," : ""); }
        o << "]}";
        o.flush();
        fprintf(stderr, "record %s done in %.1f s\n", name.c_str(), el);
    }
    o << "\n]}\n";
    llama_batch_free(batch);
    return 0;
}
