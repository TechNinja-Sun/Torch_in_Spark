package parallel;

public class TensorParallel {
    public static float[][] splitWeights(float[][] weights, int numDevices) {
        // 简化：横向切分 weights 到 numDevices 份
        int rows = weights.length;
        int cols = weights[0].length / numDevices;
        float[][] split = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(weights[i], 0, split[i], 0, cols);  // 拿第一个切片
        }
        return split;
    }

    public static void synchronize(float[][] gradients) {
        // 模拟同步所有梯度（在实际中使用通信/NCCL）
        System.out.println("Synchronizing gradients across devices (TP).");
    }
}
