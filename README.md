# 双机 DRAM BM 内存池 + SDMA/MTE 传输效率 Microbench

用 **Big Memory（BM，SDMA data op）** 把两台机器的 **host DRAM** 组成一个全局统一地址空间
（GVA）内存池，然后写一个 microbench，在**同一份数据**上对比两种传输引擎的效率：

| 引擎 | 触发方式 | 硬件执行单元 | 数据通路 |
|-|-|-|-|
| **SDMA** | `copy_data(..., flags=0)` | NPU SDMA 引擎（TS 调度 SQE） | DVA → DVA 一次 DMA |
| **MTE** | `copy_data(..., flags=COPY_EXTEND_FLAG)` | AI Core（AIV）上的 `hybm_copy_kernel` + MTE2/MTE3 | GM→UB→GM（经 64KB UB 中转） |

两条路径都落在 `HostDataOpSDMA::CopyG2G`
（`src/hybm/csrc/data_operation/host/hybm_data_op_sdma.cpp`）里，由 `flags` 分叉：

```cpp
if (flags & COPY_EXTEND_FLAG) {                       // MTE：dlopen libmf_hybm_copy_extend.so
    HybmCopyExtend(srcVA, destVA, count, 32, st);     //   -> hybm_copy_kernel（AIV 内核）
    AclrtSynchronizeStream(st);
    return BM_OK;
}
// SDMA：填充 rtStarsMemcpyAsyncSqe_t，HalSqTaskSend 发给 SDMA 引擎
InitG2GStreamTask(task, destVA, srcVA, count);
hStream->SubmitTasks(task);
hStream->Synchronize();
```

## 测试数据与方向

- 数据格式：`torch(shape=(N, 1, 576), device="npu")`，默认 `int32`。**N 由 `--scales` 扫描**，
  默认 `64,128,256,512,1024,2048`（各档字节数均为 64B 整数倍，满足 `hybm_copy_kernel` 的拷贝粒度）。
  最大档 `(2048, 1, 576)` → 4,718,592 B ≈ 4.5 MiB。可用 `--dtype` 切换。
- 方向（G = 对端 rank 的 DRAM 池 GVA）：
  - **L2G**：本地 NPU → 对端 DRAM 池
  - **G2L**：对端 DRAM 池 → 本地 NPU

> 注：DRAM 池 GVA 属 `GLOBAL_HOST`。Python 侧传 `BmCopyType.L2G/G2L` 时，
> `SmemBmEntry::TransToHybmDirection` 会按 `GetHybmMemTypeFromGva` 自动判定，实际映射为
> `LOCAL_DEVICE_TO_GLOBAL_HOST` / `GLOBAL_HOST_TO_LOCAL_DEVICE`，与 `L2GH/GH2L` 等价，都进 `CopyG2G`。

## 指标

1. **单条时延/带宽（每个 scale，每条 方向 × 引擎 组合）**：`iters` 次 `copy_data` 的平均端到端耗时
   （同步调用，内部已 `AclrtSynchronizeStream` / `hStream->Synchronize`），单位 us；
   带宽 = `nbytes / 单条时延`，单位 GB/s。
2. **批量吞吐（只在最大 scale）**：`copy_data_batch` 聚合带宽。默认 `--batch-mode random` 从**独立的
   批量源张量 `(16384, 1, 576)`**（`--batch-rows` 可调）随机取 `B` 条 `(1,576)` 行，scatter 到对端
   DRAM 批量区 / 从批量区 gather 回本地（`--seed` 固定随机行序）；`--batch-mode strided` 从同一批量源
   按固定间隔取 `B` 行（`--step-size` 控制源行间隔，0=连续取 0,1,2,...，1=隔一行取 0,2,4,...），dst
   保持连续；`--batch-mode broadcast` 则把整张最大 scale 的 src 广播到对端 DRAM 的 `B` 个连续切片 /
   从 `B` 个切片收进 dst。单位 GB/s。

## 前提

- 昇腾 NPU + CANN Toolkit（`torch` + `torch_npu`），已 source CANN 环境变量；
- 已安装 memfabric_hybrid run 包并 source 其 `set_env.sh`（提供
  `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 和 `libmf_hybm_copy_extend.so`，MTE 路径依赖）；
- **MTE 直写远端 host DRAM 依赖 A3 超节点（Device UB 1.0）**：需把 DRAM 池 GVA 映射进本机设备页表，
  SDMA/MTE 才能直接访问。普通 A2/跨机服务器若报 `dram segment does not support sdma` 或 MTE 返回非 0，
  说明当前拓扑不支持直访（会回退 RDMA 路径），本 microbench 会把对应行标成 `N/A`；
- 双机（各 1 rank / 1 卡）可互通 config store（TCP）与数据面网络；
- DRAM 池大小 2MiB 对齐，单机贡献的 DRAM 至少能容纳 `最大 scale 的 tensor_nbytes + batch_size × 单行字节`
  （`random`/`strided` 模式；`broadcast` 模式为 `batch_size × 最大 scale 的 tensor_nbytes`），
  还要给末尾 64B 握手块留位，脚本运行期会断言。

## 角色与运行

双机下角色是**对称翻转**的，避免"rank0 一 join 就开测、撞上 rank1 切片尚未 import 的 GVA"的竞态：

| 角色 | 启动 | 做什么 |
|-|-|-|
| **rank0** | 先启动 | config store host + **纯内存持有方**：建池 join 后循环 sleep，直到 Ctrl+C 退出。不碰 peer GVA、不执行任何拷贝，所以不会因 peer 未就绪而报错 |
| **rank1** | 后启动 | **benchmark 驱动/计时**：join 后 `send_peer_ready` 循环向 rank0 的 DRAM 池末尾写 READY_MAGIC（写成功 = rank0 切片已 import 进本机设备页表，见常见问题），然后跑全部单条/批量计时 |

```bash
# ===== rank0 机器（config store host + 内存持有方，Ctrl+C 退出）=====
python3 microbench_sdma_mte.py --rank 0 --url tcp://<rank0-ip>:8570 --nic <rank0-数据面ip>

# ===== rank1 机器（benchmark 驱动/计时）=====
python3 microbench_sdma_mte.py --rank 1 --url tcp://<rank0-ip>:8570 --nic <rank1-数据面ip>
```

### 单机自测（可选）

没有第二台机器时，`--world_size 1` 在本机验证整条链路（L2G/G2L 都落到本机 host DRAM 池）：

```bash
python3 microbench_sdma_mte.py --rank 0 --world_size 1 --url tcp://127.0.0.1:8570
```

## 预期输出（双机）

rank0（内存持有方）侧：

```
[rank 0] DRAM pool joined (local=1.00GiB, max=1.00GiB/rank)
[rank 0] holding DRAM pool, Ctrl+C to exit ...
   ...（一直持有，直到 Ctrl+C）
[rank 0] Ctrl+C received, exiting
[rank 0] done.
```

rank1（benchmark 驱动）侧：

```
[rank 1] joined, waiting for rank0's DRAM slice to be importable ...
[rank 1] peer_rank=0, remote DRAM GVA=0x...
[rank 1] peer handshake sent (attempt ...)
[rank 1] MTE path lib: /path/to/lib64/libmf_hybm_copy_extend.so
[rank 1] scales=[64, 128, 256, 512, 1024, 2048], iters=100, batch_size=16 x batch_iters=20 (batch_mode=random, batch_rows=16384, seed=42)
[rank 1] batch source tensor (16384, 1, 576) nbytes=36.00 MiB, ptr=0x...
[rank 1] scale (64, 1, 576) nbytes=147456 (0.14 MiB), src ptr=0x...
[rank 1] scale (128, 1, 576) nbytes=294912 (0.28 MiB), src ptr=0x...
...
[rank 1] scale (2048, 1, 576) nbytes=4718592 (4.50 MiB), src ptr=0x...
scale sweep: single-copy latency & bandwidth (dtype=int32)
shape           direction engine  latency(us)   single-BW(GB/s)
----------------------------------------------------------------
64x1x576        L2G       SDMA   ...
64x1x576        L2G       MTE    ...
...
2048x1x576      G2L       MTE    ...
shape=(2048, 1, 576), dtype=int32, nbytes=4718592 (4.50 MiB)
direction engine latency(us) single-BW(GB/s) batch-BW(GB/s)
----------------------------------------------------------------
L2G       SDMA   ...
L2G       MTE    ...
G2L       SDMA   ...
G2L       MTE    ...
>> L2G: MTE vs SDMA latency x.xx x, single-BW x.xx x, batch-BW x.xx x
>> G2L: MTE vs SDMA latency x.xx x, single-BW x.xx x, batch-BW x.xx x
[rank 1] done.
```

> 所有数据校验都走**设备拷贝通路**（bench_one 对每条 方向×引擎 组合做 L2G 写 → G2L 读回 →
> 逐元素相等），不用 `gva_to_va` + CPU `memmove` 直读 GVA：A3/Ascend 910C + GVA_V4 下 DRAM 池是
> `HybmVmmBasedSegment`，其 LVA 由驱动保留在设备侧地址空间，CPU 进程直读会段错误。

## 常用参数

| 参数 | 默认 | 说明 |
|-|-|-|
| `--scales` | `64,128,256,512,1024,2048` | 单条时延/带宽的 scale 扫描，shape=(N,1,576)；批量吞吐只在最大档测 |
| `--iters` / `--warmup` | 100 / 10 | 单条时延的测量/预热次数 |
| `--batch-size` / `--batch-iters` | 16 / 20 | 批量吞吐的每批张量数 / 测量次数 |
| `--batch-mode` | `random` | 批量拷贝模式：`random` 从独立的 `(batch_rows,1,576)` 批量源张量随机取 B 条 `(1,576)` 行 scatter/gather；`strided` 从同一批量源按固定间隔取 B 行（源行间隔由 `--step-size` 控制，dst 保持连续）；`broadcast` 广播整张最大 scale 的 src 到 B 个连续切片 |
| `--batch-rows` | 16384 | `random`/`strided` 模式批量源张量的第一维（shape=(batch_rows,1,576)） |
| `--step-size` | 0 | `--batch-mode strided`：相邻源行的间隔（以 (1,576) 行为单位），0=连续取 0,1,2,...，1=隔一行取 0,2,4,...；dst 保持连续 |
| `--seed` | 42 | `--batch-mode random` 随机行选择的种子 |
| `--dtype` | `int32` | 张量 dtype（int8/int16/int32/int64/float16/float32/float64） |
| `--local-dram` / `--max-dram` | 1GiB / 1GiB | 每机贡献/上限 DRAM，2MiB 对齐 |
| `--extend-lib-path` | 环境变量 | 手动指定 `libmf_hybm_copy_extend.so` 目录 |
| `--ready-timeout` | 120s | rank1 就绪握手超时：循环向 rank0 的 DRAM 池末尾写 READY_MAGIC，直到 rank0 切片 import 到本机设备页表 |

例如只测 4.5MiB 与 144KiB 两档的单条时延（在 rank1 上跑，rank0 仍持有内存）：

```bash
python3 microbench_sdma_mte.py --rank 1 --scales 64,2048 --url tcp://<rank0-ip>:8570 --nic <rank1-数据面ip>
```

## 结果解读

- **单条时延**反映单笔拷贝的端到端开销，包含 host 侧 launch + 引擎执行。SDMA 路径是
  SQE 下发 + SDMA DMA；MTE 路径是内核 launch + MTE2/MTE3（经 UB 中转），量级上 MT 通常更高。
- **批量吞吐**反映稳态带宽：`copy_data_batch` 会把同一 stream 上的一批小张量串/并发起来，
  聚合带宽一般明显高于单条带宽。若 `batch-BW` 为 `N/A`，通常是 A5/x86 不支持
  `BatchCopyExtend`（见 `HostDataOpSDMA::Initialize` 的告警）。
- 倍率行 `x` 表示 MTE 相对 SDMA 的比值（<1 表示 MTE 更快/更高）。

## 常见问题

- `MEMFABRIC_HYBRID_EXTEND_LIB_PATH 未设置` / `not found: .../libmf_hybm_copy_extend.so`：
  确认已 source `set_env.sh`，或 `--extend-lib-path <目录>` 指定，或先
  `bash ../hybm_copy_extend/build.sh` 编译；
- 整列 `N/A` 且提示 `dram segment does not support sdma` / `copy_data failed`：
  当前拓扑（非 A3 超节点 / A2 跨机）不支持 SDMA/MTE 直访远端 DRAM，数据面会回退 RDMA；
- 数据校验不过：确认 tensor `nbytes` 和 `data_ptr` 都是 64B 整数倍/对齐（脚本已断言），
  且双机 `--local-dram`/`--max-dram` 一致、都 join 成功；
- 批量区域超出 `--local-dram`：脚本会报错，调小 `--batch-size` / `--step-size` 或调大池大小。批量区域公式：
  `random`/`strided` 模式为 `最大 scale 的 nbytes + batch_size × 单行字节`（批量区放在 `peer_gva + nbytes`
  之后，不碰 offset 0 的单条拷贝回读区；行从独立的 `(batch_rows,1,576)` 批量源张量里挑，无需整张落盘），
  `broadcast` 模式为 `batch_size × 最大 scale 的 nbytes`；均需给末尾 64B 握手块留位（见运行期断言）；
- rank1 报 `timeout ... sending ready handshake to rank0's DRAM pool`：rank0 未启动 / join 未完成，
  rank0 的切片没 import 进 rank1 的设备页表。确认 rank0 已起来并打印出 `holding DRAM pool`；
- rank1 一直不打印 benchmark 输出 / "有时跑会出错"：BM 组引擎是**动态成员模式**，`join()` 只等自己
  就返回，对端切片 import 进本机设备页表是异步的。当前版本 rank1 开测前先 `send_peer_ready` 握手：
  循环向 rank0 的 DRAM 池末尾写 READY_MAGIC，**写成功即说明本机对 rank0 的 import 已完成**（能写就
  说明映射好了；且 JoinHandle 内的 GroupGatherResult barrier 两边一起过，rank0 侧 import 也同步完成），
  之后才开测——rank0 起得慢会自动等它，rank0 完全没起则 `--ready-timeout`（默认 120s）超时明确报错。
  rank0 侧设计上不碰 peer GVA，不会因对端未就绪而失败。正常启动顺序仍是先 rank0、再 rank1；
- 想换参数重跑：rank1 退出后，保持 rank0 的持有进程不动，重新起一个 rank1 即可（rank0 持有期间可反复跑）。
