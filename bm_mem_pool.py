#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
# MemFabric_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
"""
双机 DRAM Big Memory 全局内存池封装（BmDramPool）。

把两台机器的 **host DRAM** 贡献进同一个 BM（Big Memory，SDMA data op）全局统一地址空间
（GVA），提供 peer GVA 获取、GVA->VA 转换、单条/批量拷贝等常用操作的薄封装。

典型用法见同目录 `microbench_sdma_mte.py`，也可以直接复用：

    pool = BmDramPool(rank=0, world_size=2, device=0,
                      store_url="tcp://<rank0-ip>:8570", nic="<本机数据面ip>",
                      local_dram_size=1 << 30, max_dram_size=1 << 30)
    pool.initialize()
    pool.create_pool()                      # bm.create2(SDMA) + join
    peer_gva = pool.peer_gva(1, bm.BmMemType.HOST)
    pool.copy(src_ptr, peer_gva, size, bm.BmCopyType.L2G, flags=0)
    pool.destroy()
"""
import memfabric_hybrid
from memfabric_hybrid import bm


class BmDramPool:
    """双机 DRAM 全局内存池：初始化 -> 建池 join -> 拷贝 -> destroy。"""

    def __init__(self, rank, world_size, device, store_url, nic,
                 local_dram_size, max_dram_size,
                 start_store=True, log_level=3):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.store_url = store_url
        self.nic = nic
        self.local_dram_size = local_dram_size
        self.max_dram_size = max_dram_size
        self.start_store = start_store
        self._handle = None
        self._bm_inited = False
        self._mf_inited = False
        memfabric_hybrid.set_log_level(log_level)

    # ------------------------------------------------------------------ init
    def initialize(self):
        """memfabric_hybrid.initialize() + bm.initialize()（rank0 起 config store）。"""
        assert memfabric_hybrid.initialize() == 0, "memfabric_hybrid.initialize failed"
        self._mf_inited = True

        cfg = bm.BmConfig()
        cfg.auto_ranking = False
        cfg.rank_id = self.rank
        cfg.start_store = self.start_store
        cfg.set_nic(f"tcp://{self.nic}:1234")
        ret = bm.initialize(self.store_url, self.world_size, self.device, cfg)
        assert ret == 0, f"bm.initialize failed: {ret}"
        self._bm_inited = True
        return 0

    def create_pool(self, data_op_type=bm.BmDataOpType.SDMA):
        """把本机 host DRAM 贡献进全局池并 join。SDMA data op 是 CopyG2G 生效的前提。"""
        handle = bm.create2(id=0,
                            local_dram_size=self.local_dram_size,
                            max_dram_size=self.max_dram_size,
                            local_hbm_size=0,
                            max_hbm_size=0,
                            data_op_type=data_op_type)
        assert handle is not None, "bm.create2 failed"
        ret = handle.join()
        assert ret == 0, f"bm pool join failed: {ret}"
        self._handle = handle
        assert bm.bm_rank_id() == self.rank, f"rank mismatch: {bm.bm_rank_id()} != {self.rank}"
        return handle

    # ------------------------------------------------------------- accessors
    @property
    def handle(self):
        return self._handle

    @property
    def rank_id(self):
        return bm.bm_rank_id()

    def peer_gva(self, peer_rank, mem_type=bm.BmMemType.HOST):
        """返回 peer rank 在全局空间中的 GVA（DRAM 池用 BmMemType.HOST）。"""
        gva = self._handle.peer_rank_ptr(peer_rank, mem_type)
        assert gva != 0, f"peer_rank_ptr({peer_rank}, {mem_type}) returned 0"
        return gva

    def gva_to_va(self, gva, mem_type=bm.BmMemType.LOCAL_HOST):
        """把 GVA 转成本进程可读写的 VA（本机 host DRAM 池 GVA==HVA）。0 表示失败。"""
        return self._handle.gva_to_va(gva, mem_type)

    # ---------------------------------------------------------------- copies
    def copy(self, src_ptr, dst_ptr, size, copy_type, flags=0):
        """单条同步拷贝。flags & COPY_EXTEND_FLAG 走 MTE(AI Core)，否则走 SDMA 引擎。"""
        return self._handle.copy_data(src_ptr, dst_ptr, size, copy_type, flags)

    def copy_batch(self, src_addrs, dst_addrs, sizes, count, copy_type, flags=0):
        """批量同步拷贝（合并/批量下发 SDMA 或 HybmBatchCopyExtend）。"""
        return self._handle.copy_data_batch(src_addrs, dst_addrs, sizes, count, copy_type, flags)

    # ----------------------------------------------------------------- cleanup
    def destroy(self):
        """leave -> destroy handle -> 反初始化，幂等，异常兜底。"""
        try:
            if self._handle is not None:
                try:
                    self._handle.leave()
                finally:
                    self._handle.destroy()
                    self._handle = None
        finally:
            if self._bm_inited:
                try:
                    bm.uninitialize()
                finally:
                    self._bm_inited = False
            if self._mf_inited:
                memfabric_hybrid.uninitialize()
                self._mf_inited = False
