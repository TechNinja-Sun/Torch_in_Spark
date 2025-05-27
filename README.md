# Torch\_in\_Spark

`Torch_in_Spark` 是一个模拟 PyTorch 分布式训练机制的 Java 项目，基于 Apache Spark 实现。该项目旨在探索在 Spark 环境中实现自动微分、模型并行、数据并行等深度学习核心功能的可能性。它提供了一个轻量级的框架，用于在分布式环境中训练简单的神经网络模型。

---

## 项目结构

项目的主要代码位于 `src/main/java/DistributedAutoDiff/` 目录下，按照功能模块划分为以下子包：

```

DistributedAutoDiff/
├── autograd/
│   ├── Tensor.java
│   └── DenseLayer.java
├── comms/
│   ├── NCCLSimulator.java
│   └── TorchDistributedMock.java
├── parallel/
│   ├── ModelParallel.java
│   └── DataParallel.java
├── dataset/
│   └── MNISTLoader.java
├── engine/
│   └── Trainer.java
├── export/
│   └── SafeTensorExporter.java
├── benchmark/
│   └── PerformanceComparator.java
├── Main.java
└── pom.xml
:contentReference[oaicite:20]{index=20}
```

---

## 模块与类详解

### 1. `autograd/` 自动微分模块

#### `Tensor.java`

该类定义了张量（Tensor）数据结构，并实现了自动微分的基本功能。每个张量对象包含数据值、梯度信息以及与之相关的计算图节点。通过构建计算图，支持前向传播和反向传播的操作，模拟 PyTorch 的自动求导机制。

#### `DenseLayer.java`

实现了一个全连接层（Fully Connected Layer），支持前向传播和反向传播操作。该类利用 `Tensor` 类构建计算图，并在反向传播时计算梯度，用于参数更新。

### 2. `comms/` 通信模拟模块

#### `NCCLSimulator.java`

模拟了 NVIDIA 的 NCCL 通信库，提供了 `all-reduce`、`broadcast` 等分布式通信操作的接口。这些操作在分布式训练中用于同步参数和梯度。

#### `TorchDistributedMock.java`

模拟了 PyTorch 的 `torch.distributed` 接口，管理进程组、rank 等信息。该类为分布式训练提供了基础设施，支持多进程之间的协调与通信。

### 3. `parallel/` 并行训练模块

#### `ModelParallel.java`

实现了模型并行的逻辑，将神经网络的不同层分布到多个计算节点上进行训练。该类负责将模型划分为子模块，并协调各部分的前向和反向传播。

#### `DataParallel.java`

实现了数据并行的逻辑，将训练数据划分为多个子集，分别在不同的计算节点上进行训练。该类负责在各节点上复制模型，独立计算梯度，并通过通信模块同步参数。

### 4. `dataset/` 数据加载模块

#### `MNISTLoader.java`

提供了加载 MNIST 数据集的功能，支持从本地文件系统或 HDFS 加载数据。该类将图像数据和标签转换为 `Tensor` 对象，供模型训练使用。

### 5. `engine/` 训练引擎模块

#### `Trainer.java`

封装了模型训练的主要流程，包括数据加载、前向传播、反向传播、参数更新等步骤。该类支持分布式训练，利用通信模块同步参数，并根据配置选择模型并行或数据并行策略。

### 6. `export/` 模型导出模块

#### `SafeTensorExporter.java`

提供了将训练好的模型参数导出为 safetensor 格式的功能，便于模型的保存和部署。该类将 `Tensor` 对象序列化为标准格式，支持跨平台加载。

### 7. `benchmark/` 性能评估模块

#### `PerformanceComparator.java`

用于评估本项目与 PyTorch 在运行时间和准确率方面的差异。该类运行相同的模型和数据集，记录训练时间和模型性能，供用户参考。

### 8. `Main.java`

项目的入口类，负责解析命令行参数，初始化训练环境，并启动训练流程。该类使用 Spark 提交作业，协调各模块的运行。

### 9. `pom.xml`

Maven 项目的配置文件，定义了项目的依赖关系和构建设置。该文件包含了 Spark、Hadoop 等必要的依赖项，确保项目的可构建性和可运行性。

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/TechNinja-Sun/Torch_in_Spark.git
cd Torch_in_Spark
```



### 2. 构建项目

使用 Maven 构建项目：

```bash
mvn clean package
```



### 3. 运行训练

使用 Spark 提交训练作业：

```bash
spark-submit --class DistributedAutoDiff.Main target/Torch_in_Spark-1.0.jar
```



根据需要，可以在命令行中添加参数以配置训练设置，例如并行策略、数据路径等。

---

## 注意事项

* 该项目为实验性质，主要用于探索在 Spark 环境中实现深度学习训练的可能性。
* 目前仅支持全连接层的模型结构，适用于简单的任务，如 MNIST 分类。
* 通信模块为模拟实现，主要用于验证分布式训练的流程，并不具备高性能通信能力。
* 模型导出和性能评估模块为可选组件，可根据需要启用。

---

## 贡献指南

欢迎对本项目提出建议或贡献代码。如有兴趣，请提交 Pull Request 或在 Issues 中讨论。

---

如需进一步了解项目的实现细节或有其他问题，欢迎访问项目主页或联系维护者。

---
