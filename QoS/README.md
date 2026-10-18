# HBM 读取 QoS benchmark

`qos_bench.py` 对应《A3超节点内直接访存竞争实验设置.md》，保留仓库现有调用逻辑：

- SDMA：`handle.copy_data(remote_hbm_gva, dst_ptr, size, G2L, 0)`。
- MTE：同一接口，`flags=1 << 1`（`COPY_EXTEND_FLAG`）。
- 建池使用 `BmDataOpType.SDMA`；rank0 贡献 HBM，rank1 不贡献 HBM。测试方向始终为对端 HBM → 本地 NPU，不使用 DRAM 池替代。

## 当前接口能测什么

**这是同步接口的多线程调用 benchmark。硬件并发、直访路径和绑定层线程安全仍须在实际版本上验证。**

D1/rank0 是数据源，D0/rank1 是请求端，与物理设备编号无关。请求端一条 H 线程、K 条 P 线程；每条线程至多执行一个同步 `copy_data`，每条 P 线程拥有独立目标缓冲。计时期间不持锁调用后端，不使用 `copy_data_batch` 或线程池待提交队列。

K 限制的是**尚未返回的应用层同步调用数**。计数从调用前开始，可能包含 GIL、绑定层或后端锁等待；接口没有暴露接受请求的时刻，因此这不是方案中已经验证的“后端接受后在途数”，也不是硬件队列深度。`inflight.csv` 为时间加权统计提供状态变化记录，`software_pending=0` 表示没有额外的软件待提交队列。

后端返回即按现有同步契约判定完成，`t_done=t_return`。`t_return-t_call` 包含同步等待，不能称为纯提交开销。H 的 `t_ready` 在计算地址与开始调用之前记录，主指标为 `t_done-t_ready`；所有时间戳来自同一进程的 `perf_counter_ns()`。H 以闭环方式运行，可设置完成后间隔 G。

`summary.json` 始终保存 `device_overlap=unverified`。主机调用区间交叠只说明调用并发，不能证明设备执行交叠；若绑定持有 GIL、内部使用同一 stream 或互斥锁，结果可能是串行对照。`status=ok` 仅代表工作量完成且校验通过，不代表通过了路径和并发资格验证。请单独采集 CANN/MemFabric profiling，确认 SDMA/MTE 真实路径、H/P 执行交叠、stream 分配以及无 RDMA 回退/主机中转，再进行同路径因果比较。脚本不会自行切换引擎重试。

E4 仅实现 S0/S1。**S2 暂不可用**：同步调用只暴露完成返回，无法在“后端接受 H”时恢复 P；持锁等待 H 返回会变成另一种策略，故未将其标成 S2。完整 S2 需要 MemFabric 提供独立异步提交/完成接口，或补充原生后端封装。MTE block 数与 pipeline 参数继承已安装的 extend 库，当前 Python 接口不支持独立扫描，需由该库和独立 profiling 补足。

## 环境与启动

需要 A3 超节点内支持直接访问的两个 NPU、CANN、`torch`、`torch_npu`、`memfabric_hybrid`；MTE 还需 source 包内 `set_env.sh`，确保 extend 库可加载。先检查实际拓扑和设备映射。两端必须使用相同 H/P 工作集和 pool ID；控制握手会检查布局一致性。

从仓库根目录启动。下面是同机双卡的预试验，rank0 使用逻辑设备 0，rank1 使用逻辑设备 1：

```bash
# 终端一：D1 数据源；在请求端完成后自动退出
python3 QoS/qos_bench.py --rank 0 --device 0 \
  --url tcp://127.0.0.1:8570 --nic 127.0.0.1

# 终端二：D0 请求端；短时预试验，不满足正式样本预算
python3 QoS/qos_bench.py --rank 1 --device 1 \
  --url tcp://127.0.0.1:8570 --nic 127.0.0.1 --source-host 127.0.0.1 \
  --engines sdma --modes sequential --experiments E1 E2 \
  --p-sizes 65536 --ks 1 2 4 --repeats 1 \
  --warmup-seconds 1 --seconds 3 --min-h-samples 1000 \
  --output QoS/results/pilot-sdma
```

跨服务器时，双方 `--url` 使用数据源服务器 IP；请求端 `--source-host` 也使用该 IP，双方 `--nic` 各自填写数据面 IP。同机也需按部署填写可用 NIC，回环地址只适用于相应部署支持的场景。开放 TCP 8570（BM store）、8571（控制），数据面连接要求沿用 MemFabric。设备 ID 为各进程可见的逻辑 ID。

控制连接确保源数据填充完成后才开测；随后等待 HBM 在本设备地址空间的映射并执行指定引擎正确性预检。源端只在计时前初始化 HBM，之后保持不动。源端 staging 默认使用 SDMA，可通过 `--stage-engine mte` 显式选择。读取校验失败会退出并保留日志，不会悄悄换引擎。

## 实验参数

默认 H=64 KiB，H/P 工作集分别为 64/256 MiB，分别占用源 HBM 的连续非重叠区域；池大小共 320 MiB。工作集必须为 2 MiB 的正整数倍。源数据使用依赖地址和区域的确定性字节模式，分块填充；预检先 poison 目标，再检查读取内容。测试结束先验证每个实际最后写入的目标槽，再执行额外读取校验。初始化、分配、源填充和正确性校验均在采样区间外。

连续地址在各自工作集循环；离散地址使用固定 seed 对 1 MiB tile 做一一置换，tile 内连续遍历。所有请求大小按同一 tile 顺序访问，保持相同逻辑字节序列；不是独立逐请求均匀随机采样。地址重复周期为工作集字节数，H/P 的 seed 分别为 `seed`/`seed+1`。缓存状态不可控，结果不能称为纯远端 HBM 时延。

| 实验 | 运行设置 |
| --- | --- |
| E1 | 自动在每个后端、地址模式、重复轮次的混合负载批次前后各测一次 |
| E2 | P=4096/65536/1048576 B，K=1/2/4/8/16/32，配置顺序按 seed 打乱 |
| E3 | 固定 P 总量；按共同目标字节率准入，H 在 P 活跃期间持续运行 |
| E4 | H 数量和 P 总量固定；S0=`k-high`，S1=`k-low`；直至两类任务全部完成 |

默认执行 E1/E2，5 轮，每配置预热 5 秒，采样同时满足 30 秒及 100,000 个 H 样本才结束。预热时长不是自动稳定性判定，请在预试验中检查状态漂移。`--max-seconds` 可设每轮时间预算，未完成的轮次标为无效并退出；它不能中断卡死在原生同步调用内的线程，需要进程级 watchdog。长测保留逐笔内存日志，内存占用随样本量增长，应为日志和原始 CSV 留足空间。

正式 E1/E2 请求端命令（源端按上面方法重新启动，每个源进程服务一个请求进程）：

```bash
python3 QoS/qos_bench.py --rank 1 --device 1 \
  --url tcp://127.0.0.1:8570 --nic 127.0.0.1 \
  --engines sdma mte --experiments E1 E2 \
  --output QoS/results/e1-e2
```

E3 应先独立测量各 P 大小的可持续带宽，再选择所有大小均能达到的目标率。脚本不自动猜测目标率；例如已确认共同可达 **1 GiB/s** 后：

```bash
python3 QoS/qos_bench.py --rank 1 --device 1 \
  --engines sdma --experiments E3 --target-gib-s 1 \
  --p-total-bytes 8589934592 --k-high 16 --output QoS/results/e3
```

限速使用全局单请求字节预算：第 n 笔请求不早于 `start+n*P_size/R` 调用。启动时最多提前一笔请求；线程调度延误后可以追赶历史预算，故不是严格的滑动窗口平滑器。通过原始提交时间戳检查突发；在相近**实际**提交/完成负载的共同活跃窗口内比较。短任务或未达到目标率的配置不能用于匹配负载结论。

E4 从 E2 中选择 K 后运行，例如：

```bash
python3 QoS/qos_bench.py --rank 1 --device 1 \
  --engines sdma --experiments E4 --e4-p-size 65536 \
  --k-high 16 --k-low 2 --h-count 100000 --p-total-bytes 8589934592 \
  --output QoS/results/e4
```

E3/E4 正式阶段按固定工作量终止，30 秒/100,000 样本目标只记录是否达成，不改变终止规则。E4 H/P 的地址分配按逻辑请求序号确定，S0/S1 保持相同请求数量、字节数和地址序列；线程之间实际调用先后受调度影响。

## 输出与解释

每次运行使用新的输出目录，拒绝覆盖。每轮保存：

```text
config.json          参数、版本、设备、源端信息、时间/地址/K 的口径
h_requests.csv       H 请求序号、偏移、字节、四个时间戳、状态
p_completions.csv    P 请求序号、独立目标槽、偏移、调用/返回/完成时间、状态
inflight.csv         P 未返回调用数的变化；无额外软件队列
summary.json         nearest-rank 分位数、完成吞吐、请求率、共同活跃窗口、在途统计
validation.log       正确性结果、路径/重叠待验证说明、异常
profiling/           留给独立 profiling 轮次，不混入正常采样
```

输出目录根部另有全轮次 `summary.json` 和配对 `comparisons.json`：按相同后端、地址模式、重复轮次匹配 E1；记录 H p99 放大、前后基线漂移，以及 E4 S1 相对 S0 的 p99 改善、P 吞吐损失和总完成时间变化。不合并不同轮次的分位数，不从单轮最优值推导结论。

全程和共同活跃窗口分开统计。共同活跃区间为双方首次调用较晚者到双方末次完成较早者；H 样本必须从 ready 到 done 完全处于区间内，P 按区间内完成字节计数。此窗口不额外截除稳态边缘，必须检查实际持续时间。另存共同活跃 H 样本占比、P 活跃吞吐和全任务完成时间。低样本量时 `sample_target_met=false`，应结合原始样本判断 p99 的统计限制。

失败调用保留异常状态，排除在成功完成吞吐和分位数之外，并使整轮无效。预检失败写入日志并退出；不支持的引擎应分开运行、保存失败原因。路径、并发和硬件信息不足时，即使功能测试通过，也不应报告硬件资源竞争结论。

`--metadata environment.json` 可补充设备物理端点、拓扑/超节点标识、CANN/驱动/固件/MemFabric commit、编译器、MTE kernel/block/pipeline、功耗/频率/温度和 profiling 证据等；脚本自动记录主机、OS、Python、torch/MemFabric 可获取版本、`npu-smi info`、CPU 亲和性、Git 状态与脚本 SHA256。无法自动获取的字段不会虚构。

## 无 NPU 的功能验证

```bash
python3 -m unittest discover -s QoS/tests -v

python3 QoS/qos_bench.py --dry-run

python3 QoS/qos_bench.py --simulate --engines sdma --modes sequential \
  --experiments E1 E2 E3 E4 --p-sizes 4096 1048576 --ks 1 2 \
  --k-high 2 --k-low 1 --seconds .02 --warmup-seconds .002 \
  --min-h-samples 10 --repeats 1 --h-count 20 --p-total-bytes 1048576 \
  --target-gib-s .1 --output /tmp/qos-functional-smoke
```

模拟后端只测试调度、工作量、终止和结果文件；所有结果明确标记 `simulation=true`，数字没有硬件性能意义。
