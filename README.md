DistributedAutoDiff/
├── autograd/
│   ├── Tensor.java                # Tensor数据结构及自动微分逻辑
│   └── DenseLayer.java           # 仅支持全连接层（forward & backward）
│
├── comms/
│   ├── NCCLSimulator.java        # 模拟NCCL通信接口（all-reduce, broadcast等）
│   └── TorchDistributedMock.java# 模拟torch.distributed接口（进程组，rank等）
│
├── parallel/
│   ├── ModelParallel.java        # 模型并行逻辑
│   └── DataParallel.java         # 数据并行逻辑
│
├── dataset/
│   └── MNISTLoader.java          # 加载 MNIST（本地或 HDFS）
│
├── engine/
│   └── Trainer.java              # 训练引擎（支持分布式）
│
├── export/
│   └── SafeTensorExporter.java   # （可选）导出为 safetensor 格式
│
├── benchmark/
│   └── PerformanceComparator.java# （可选）与 PyTorch 运行时间/准确率对比
│
├── Main.java                     # 程序入口（Spark 作业提交）
└── pom.xml                       # Maven 依赖管理（使用 Spark Java API）
