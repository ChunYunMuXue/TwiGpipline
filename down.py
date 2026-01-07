"""
批量下载 FLUX-Reason-6M 的 Aesthetics-Part01 parquet 分片。

默认下载 00000~00099（共 100 个），下载到:
  /nfs/wenjie/wenjie_0104/data/Aesthetics-Part01/
并做轻量校验：读取 parquet metadata，确认文件可被 pyarrow 打开。
"""

from __future__ import annotations

import os
from huggingface_hub import hf_hub_download, login
import pyarrow.parquet as pq

# ===================== 关键修正 =====================
# 1. 先登录（如果是私有仓库，必须执行！）
# 替换成你的 Hugging Face Token（带 read 权限），或注释掉这行后手动执行 login()
# login(token="你的HF Token")  # 比如：login(token="hf_xxxxxxxxx")

# 2. 仓库信息（从你的链接复制，确保准确）
REPO_ID = "LucasFang/FLUX-Reason-6M"
REPO_TYPE = "dataset"  # 必须指定是数据集仓库！之前漏了这个
# 分片文件命名规则：fluxdb-aesthetics-part01-{idx:05d}-of-00415.parquet
FILENAME_TEMPLATE = "Text/fluxdb-text-{idx:05d}-of-00143.parquet"
# ====================================================

def _validate_parquet(path: str) -> tuple[int, int]:
    """返回 (num_rows, num_row_groups)，用于快速校验文件可读。"""
    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows), int(pf.metadata.num_row_groups)

def main(
    start_idx: int = 0,
    end_idx: int = 14,
    local_dir: str = "/nfs/wenjie/wenjie_0104/data",
    force_download: bool = False,
) -> None:
    os.makedirs(os.path.join(local_dir, "Text"), exist_ok=True)

    ok, failed = 0, 0
    failed_list: list[tuple[int, str]] = []

    for idx in range(int(start_idx), int(end_idx) + 1):
        filename = FILENAME_TEMPLATE.format(idx=idx)
        print(f"\n[download] {idx:05d} -> {filename}")
        try:
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=filename,
                local_dir=local_dir,  # 会保留 Aesthetics-Part01 子目录
                force_download=force_download,
                # token=True,  # 如果你用 login() 登过，会自动用缓存 token；也可以打开这行强制使用 token
            )
            rows, rgs = _validate_parquet(file_path)
            print(f"✅ 成功：{file_path} (rows={rows}, row_groups={rgs})")
            ok += 1
        except Exception as e:
            msg = str(e).strip().replace("\n", " ")
            print(f"❌ 失败：{idx:05d} {msg[:300]}")
            failed += 1
            failed_list.append((idx, msg[:300]))

    print("\n==================== 汇总 ====================")
    print(f"成功：{ok} | 失败：{failed}")
    if failed_list:
        print("失败列表（idx -> reason）：")
        for idx, reason in failed_list:
            print(f"  - {idx:05d}: {reason}")


if __name__ == "__main__":
    # 如果是私有数据集/需要权限：先取消注释并填 token
    # login(token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    main()