package parallel;

import autograd.DenseLayer;
import autograd.Tensor;
import comms.NCCLSimulator;

import java.util.ArrayList;
import java.util.List;

public class DataParallel {

    private final List<DenseLayer> modelReplicas;

    public DataParallel(DenseLayer baseModel, int numReplicas) {
        modelReplicas = new ArrayList<>();
        for (int i = 0; i < numReplicas; i++) {
            // 每个副本复制一份 DenseLayer
            modelReplicas.add(baseModel.copy());
        }
    }

    public List<Tensor> forward(List<Tensor> inputs) {
        List<Tensor> outputs = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            Tensor out = modelReplicas.get(i).forward(inputs.get(i));
            outputs.add(out);
        }
        return outputs;
    }

    public void backward(List<Tensor> gradOutputs) {
        List<Tensor> grads = new ArrayList<>();
        for (int i = 0; i < gradOutputs.size(); i++) {
            Tensor g = modelReplicas.get(i).backward(gradOutputs.get(i));
            grads.add(g);
        }

        // 聚合梯度
        NCCLSimulator.allReduce(grads);
    }

    public List<DenseLayer> getModelReplicas() {
        return modelReplicas;
    }
}
