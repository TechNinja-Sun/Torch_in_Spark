package comms;

import autograd.Tensor;

import java.util.List;

public class NCCLSimulator {

    // AllReduce: 所有 Tensor 相加，结果同步到每个 Tensor
    public static void allReduce(List<Tensor> tensors) {
        int m = tensors.get(0).data.length;
        int n = tensors.get(0).data[0].length;
        double[][] sum = new double[m][n];

        // 累加
        for (Tensor t : tensors) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    sum[i][j] += t.data[i][j];
        }

        // 写回平均值（或总和）
        for (Tensor t : tensors) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    t.data[i][j] = sum[i][j];
        }

        System.out.println("[NCCLSimulator] AllReduce completed on " + tensors.size() + " tensors.");
    }

    // Broadcast: 将 source 的数据复制到所有 targets 中
    public static void broadcast(Tensor source, List<Tensor> targets) {
        int m = source.data.length;
        int n = source.data[0].length;

        for (Tensor t : targets) {
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    t.data[i][j] = source.data[i][j];
        }

        System.out.println("[NCCLSimulator] Broadcast from source to " + targets.size() + " targets.");
    }
}
