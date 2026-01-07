from __future__ import annotations
from typing import Optional
import numpy as np
from portimp.patchcore.config import PatchCoreTrainConfig

def sample_train_vectors_for_ivf(features: np.ndarray, nlist: int, seed: int) -> np.ndarray:
    """
    IVF/PQ 的 index.train() 需要训练向量（用于聚类/码本学习）。
    一个实用规则：至少 max(50000, 20*nlist) 条（不足则用全部）。
    """
    rng = np.random.RandomState(seed)
    n = features.shape[0]
    # Faiss warning 给你的参考：2048 -> 至少 79872
    target = min(n, max(80_000, 40 * int(nlist)))
    idx = rng.choice(n, size=target, replace=False) if target < n else np.arange(n)
    return features[idx].astype(np.float32)

def build_faiss_index(
    cfg: PatchCoreTrainConfig,
    memory_bank: np.ndarray,
    train_vectors: Optional[np.ndarray] = None
):
    """
    根据 cfg.index_type 构建不同 Faiss 索引（Flat / IVFFlat / IVFPQ / HNSW）

    关键原则（为了“训练端可落盘、推理端可复用”）：
    - 自动检测 faiss-gpu + GPU 是否可用；可用则用 cfg.gpu_device_id 对应 GPU 加速建库
    - 但：无论是否用 GPU，最终都返回 “CPU index”
      （因为 GPU index 不能直接 write_index，必须转回 CPU 才能保存）
    """
    import numpy as np
    import faiss

    d = int(memory_bank.shape[1])
    index_type = str(cfg.index_type)

    # Faiss 通常要求 contiguous float32
    mb = np.ascontiguousarray(memory_bank, dtype=np.float32)
    tv = np.ascontiguousarray(
        train_vectors if train_vectors is not None else memory_bank,
        dtype=np.float32
    )

    # ========== 自动检测是否能用 faiss-gpu ==========
    def _can_use_faiss_gpu() -> bool:
        # 必须同时具备：StandardGpuResources / index_cpu_to_gpu / index_gpu_to_cpu
        if not (hasattr(faiss, "StandardGpuResources")
                and hasattr(faiss, "index_cpu_to_gpu")
                and hasattr(faiss, "index_gpu_to_cpu")):
            return False
        try:
            ng = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        except Exception:
            ng = 0
        return ng > 0

    def _norm_gpu_id(gpu_id: int) -> int:
        try:
            ng = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        except Exception:
            ng = 0
        if ng <= 0:
            return 0
        if gpu_id < 0:
            return 0
        if gpu_id >= ng:
            print(f"[WARN] gpu_device_id={gpu_id} out of range (num_gpus={ng}) -> fallback to 0")
            return 0
        return gpu_id

    def _build_on_gpu_then_back_to_cpu(cpu_index, do_train_add_fn, index_name: str):
        """
        cpu_index: 先在 CPU 上构建出 index 结构
        do_train_add_fn: 一个函数，接收 gpu_index，在 gpu 上 train/add，并返回 gpu_index
        返回：cpu_index（已包含向量，可保存）
        """
        if not _can_use_faiss_gpu():
            print("[INFO] faiss-gpu not available -> build index on CPU.")
            # 直接 CPU train/add
            return do_train_add_fn(cpu_index)

        gpu_id = _norm_gpu_id(int(getattr(cfg, "gpu_device_id", 0)))

        try:
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
            print(f"[INFO] {index_name}: using Faiss GPU on cuda:{gpu_id}")

            gpu_index = do_train_add_fn(gpu_index)

            # 重要：转回 CPU，保证能 write_index
            cpu_index_done = faiss.index_gpu_to_cpu(gpu_index)
            return cpu_index_done
        except Exception as e:
            print(f"[WARN] {index_name}: Faiss GPU path failed -> fallback CPU. err={e}")
            # 回退 CPU
            return do_train_add_fn(cpu_index)

    # ---------- Flat（精确） ----------
    if index_type == "Flat":
        cpu_index = faiss.IndexFlatL2(d)

        def _do(idx):
            idx.add(mb)
            return idx

        return _build_on_gpu_then_back_to_cpu(cpu_index, _do, "Flat")

    # ---------- IVF Flat（近似，需 train） ----------
    if index_type == "IVFFlat":
        nlist = int(cfg.ivf_nlist)
        if nlist <= 0:
            raise ValueError("IVFFlat 需要 ivf_nlist > 0")

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

        def _do(idx):
            idx.train(tv)
            idx.add(mb)
            # nprobe 属于推理参数，这里设不设都行；建议写进 meta
            idx.nprobe = int(cfg.ivf_nprobe)
            return idx

        return _build_on_gpu_then_back_to_cpu(cpu_index, _do, "IVFFlat")

    # ---------- IVF PQ（更省内存，需 train） ----------
    if index_type == "IVFPQ":
        nlist = int(cfg.ivf_nlist)
        m = int(cfg.pq_m)
        nbits = int(cfg.pq_nbits)

        if nlist <= 0:
            raise ValueError("IVFPQ 需要 ivf_nlist > 0")
        if m <= 0:
            raise ValueError("IVFPQ 需要 pq_m > 0")
        if d % m != 0:
            raise ValueError(f"IVFPQ 建议满足 d % m == 0，但当前 d={d}, m={m}（请调整 pq_m 或 projection_dim）")

        quantizer = faiss.IndexFlatL2(d)
        cpu_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

        def _do(idx):
            idx.train(tv)
            idx.add(mb)
            idx.nprobe = int(cfg.ivf_nprobe)
            return idx

        return _build_on_gpu_then_back_to_cpu(cpu_index, _do, "IVFPQ")

    # ---------- HNSW（图索引，通常 CPU 更稳；Faiss GPU 也不支持这个） ----------
    if index_type == "HNSW":
        m = int(cfg.hnsw_m)
        if m <= 0:
            raise ValueError("HNSW 需要 hnsw_m > 0")

        index = faiss.IndexHNSWFlat(d, m)
        index.hnsw.efConstruction = int(cfg.hnsw_ef_construction)
        index.hnsw.efSearch = int(cfg.hnsw_ef_search)
        index.add(mb)
        print("[INFO] index_type=HNSW -> keep on CPU.")
        return index

    raise ValueError(f"Unsupported index_type: {index_type}")

# index type的详细内容
def index_tag(cfg) -> str:
    t = str(cfg.index_type)
    if t == "Flat":
        return "idxFlat"
    if t == "IVFFlat":
        return f"idxIVFFlat_nl{cfg.ivf_nlist}_np{cfg.ivf_nprobe}"
    if t == "IVFPQ":
        return f"idxIVFPQ_nl{cfg.ivf_nlist}_np{cfg.ivf_nprobe}_m{cfg.pq_m}_b{cfg.pq_nbits}"
    if t == "HNSW":
        return f"idxHNSW_m{cfg.hnsw_m}_efc{cfg.hnsw_ef_construction}_efs{cfg.hnsw_ef_search}"
    return f"idx{t}"
