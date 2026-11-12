# source me: ROCm runtime paths + patched P2 binary dir for R3
V="${ROCM_VENV:-$HOME/.local/share/lifeos-inference/avalon-q8-prep-e09e775/rocm-7.14.1-venv}"
R="$(ls -d "$V"/lib/python3*/site-packages/_rocm_sdk_devel)"
export ROCM_PATH="$R" HIP_PATH="$R"
export R3_BIN_DIR="${R3_BIN_DIR:-$HOME/work/avalon-rdna2/build-hip-gfx1030-qsa/bin}"
export LD_LIBRARY_PATH="$R3_BIN_DIR:$R/lib:$R/lib/llvm/lib:${LD_LIBRARY_PATH:-}"
export QSA_MODEL="${QSA_MODEL:-$HOME/.local/share/lifeos-inference/models/Qwen3.8-Flash-Next-Avalon-IQ3KS/Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf}"
export QSA_NLL_BIN="$R3_BIN_DIR/llama-qsa-nll"
