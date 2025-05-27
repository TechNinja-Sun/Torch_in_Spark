package parallel;

public class PipelineParallel {
    public static void runStage(int stageId, float[][] input) {
        // 模拟每个 stage 的计算
        System.out.println("Running stage " + stageId + " with input size " + input.length);
        // 实际中：各 stage 使用 RPC 或网络发送到下一个 stage
    }
}
