package parallel;

import autograd.DenseLayer;
import autograd.Tensor;

import java.io.Serializable;
import java.util.List;

public class ModelParallel implements Serializable {

    private static final long serialVersionUID = 1L;
    private final List<DenseLayer> layers;

    public ModelParallel(List<DenseLayer> layers) {
        this.layers = layers;
    }

    public Tensor forward(Tensor input) {
        Tensor x = input;
        for (int i = 0; i < layers.size(); i++) {
            System.out.println("[PP] Forward at Stage " + i + " on device " + i);
            x = layers.get(i).forward(x);
            simulateTensorParallelSync(i);
        }
        return x;
    }

    public Tensor backward(Tensor gradOutput) {
        Tensor grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            System.out.println("[PP] Backward at Stage " + i + " on device " + i);
            grad = layers.get(i).backward(grad);
            simulateTensorParallelSync(i);
        }
        return grad;
    }

    //  stage 的 forward，仅适用于 PP 流水线并行调度
    public Tensor forwardStage(int stageId, Tensor input) {
        if (stageId < 0 || stageId >= layers.size()) {
            throw new IllegalArgumentException("Invalid stageId");
        }
        System.out.println("[PP] Forward Stage " + stageId + " executing on device " + stageId);
        Tensor output = layers.get(stageId).forward(input);
        simulateTensorParallelSync(stageId);
        return output;
    }

    //  stage 的 backward
    public Tensor backwardStage(int stageId, Tensor gradOutput) {
        if (stageId < 0 || stageId >= layers.size()) {
            throw new IllegalArgumentException("Invalid stageId");
        }
        System.out.println("[PP] Backward Stage " + stageId + " executing on device " + stageId);
        Tensor grad = layers.get(stageId).backward(gradOutput);
        simulateTensorParallelSync(stageId);
        return grad;
    }

    //  Tensor Parallel 的梯度同步（例如多 GPU 的 NCCL AllReduce）
    private void simulateTensorParallelSync(int stageId) {
        System.out.println("[TP] Simulating tensor/gradient sync at stage " + stageId);
    }
}
