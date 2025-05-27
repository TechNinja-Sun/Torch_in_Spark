package parallel;

import autograd.DenseLayer;
import autograd.Tensor;

import java.util.ArrayList;
import java.util.List;

public class ParallelTest {

    public static void main(String[] args) {
        // 先准备一批层
        DenseLayer layer1 = new DenseLayer(2, 3);
        DenseLayer layer2 = new DenseLayer(3, 2);
        List<DenseLayer> layers = new ArrayList<>();
        layers.add(layer1);
        layers.add(layer2);

        // === 测试模型并行 ModelParallel ===
        System.out.println("=== ModelParallel 测试 ===");
        ModelParallel modelParallel = new ModelParallel(layers);
        Tensor input = new Tensor(new double[][]{{1.0, 2.0}});
        Tensor output = modelParallel.forward(input);
        System.out.println("Forward 输出:");
        output.printData();

        Tensor gradOutput = new Tensor(new double[][]{{1.0, 1.0}});
        modelParallel.backward(gradOutput);

        // === 测试数据并行 DataParallel ===
        System.out.println("\n=== DataParallel 测试 ===");
        DenseLayer baseModel = new DenseLayer(2, 2);
        int numReplicas = 2;
        DataParallel dataParallel = new DataParallel(baseModel, numReplicas);

        // 准备两个输入，分别送入两个副本
        List<Tensor> inputs = new ArrayList<>();
        inputs.add(new Tensor(new double[][]{{1.0, 0.5}}));
        inputs.add(new Tensor(new double[][]{{0.2, 0.3}}));

        List<Tensor> outputs = dataParallel.forward(inputs);
        System.out.println("Forward 输出:");
        for (int i = 0; i < outputs.size(); i++) {
            System.out.printf("Replica %d output:\n", i);
            outputs.get(i).printData();
        }

        // 反向传播梯度
        List<Tensor> gradOutputs = new ArrayList<>();
        gradOutputs.add(new Tensor(new double[][]{{1.0, 1.0}}));
        gradOutputs.add(new Tensor(new double[][]{{0.5, 0.5}}));

        dataParallel.backward(gradOutputs);

        // 输出各副本梯度查看
        List<DenseLayer> replicas = dataParallel.getModelReplicas();
        for (int i = 0; i < replicas.size(); i++) {
            System.out.printf("Replica %d 权重梯度:\n", i);
            replicas.get(i).printWeightGrad();
        }
    }
}
