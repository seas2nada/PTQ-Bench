import torch
import matplotlib.pyplot as plt

# Load Hessian
H = torch.load("/home/ptq_docker/Workspace/PTQ-Bench/output/llama-2-7b-GPTQ-wikitext2-3bit-g128-nsamples128/h_out.pt/model.layers.0.self_attn.k_proj_H.pt")
H = H['H_sum'].detach().float().cpu()  # [4096,4096]
print("H shape:", tuple(H.shape))

# ----------------------------
# 0) 보기 좋게 스케일링 유틸
# ----------------------------
def robust_view(A, take_abs=True, log1p=True, clip_percentile=99.5):
    X = A
    if take_abs:
        X = X.abs()
    if log1p:
        X = torch.log1p(X)
    # 퍼센타일 클리핑 (극단값 때문에 전체가 까맣게 되는 것 방지)
    flat = X.flatten()
    hi = torch.quantile(flat, clip_percentile/100.0).item()
    X = X.clamp(max=hi)
    return X

# ----------------------------
# 1) (가장 간단) 좌상단 일부만 보기
# ----------------------------
k = 256  # 128/256/512 중 골라서
X = robust_view(H[:k, :k])

plt.figure(figsize=(6, 6))
plt.imshow(X.numpy(), aspect='auto')
plt.title(f"H[:{k}, :{k}]  (abs + log1p + p{99.5} clip)")
plt.xlabel("col")
plt.ylabel("row")
plt.colorbar()
plt.tight_layout()
plt.savefig("fig1.png")

# ----------------------------
# 2) (추천) 블록 평균으로 4096->256 다운샘플
#    4096/256 = 16 이라 딱 떨어짐
# ----------------------------
out = 256
b = H.shape[0] // out  # block size (16)
Hr = H.view(out, b, out, b).mean(dim=(1, 3))  # [256,256]
Xr = robust_view(Hr)

plt.figure(figsize=(6, 6))
plt.imshow(Xr.numpy(), aspect='auto')
plt.title(f"Block-mean downsample: 4096→{out} (abs + log1p + p{99.5} clip)")
plt.xlabel("block-col")
plt.ylabel("block-row")
plt.colorbar()
plt.tight_layout()
plt.savefig("fig2.png")

# ----------------------------
# 3) 대각선 vs 비대각 성분 분포 (패턴 진단용)
# ----------------------------
diag = H.diag()
off = H[~torch.eye(H.shape[0], dtype=torch.bool)]

plt.figure(figsize=(6, 4))
plt.hist(diag.abs().numpy(), bins=100, alpha=0.6, label="|diag|")
plt.hist(off.abs().numpy(), bins=100, alpha=0.6, label="|offdiag|")
plt.yscale("log")
plt.title("Magnitude distribution (log y)")
plt.legend()
plt.tight_layout()
plt.savefig("fig3.png")