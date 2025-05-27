import autograd.DenseLayer;
import autograd.Tensor;
import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import parallel.ModelParallel;

import java.util.*;

public class Main {
    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("SparkModelParallelTest").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        // 输入数据
        List<Tensor> inputList = Collections.singletonList(Tensor.rand(1, 784)); // MNIST
        JavaRDD<Tensor> inputRDD = sc.parallelize(inputList, 3); // 3个Stage的并行

        // 初始化模型层（每层是一个 stage）
        List<DenseLayer> layers = Arrays.asList(
                new DenseLayer(784, 256),
                new DenseLayer(256, 128),
                new DenseLayer(128, 10)
        );
        ModelParallel mp = new ModelParallel(layers);

        // 每个partition对应一个stage，执行 forwardStage
        JavaRDD<Tensor> stageResults = inputRDD.mapPartitionsWithIndex((stageId, iterator) -> {
            List<Tensor> outputs = new ArrayList<>();
            while (iterator.hasNext()) {
                Tensor input = iterator.next();
                Tensor output = mp.forwardStage(stageId, input); 
                outputs.add(output);
            }
            return outputs.iterator();
        }, true);

        // 收集结果（此处只为调试演示）
        List<Tensor> finalOutputs = stageResults.collect();
        System.out.println("Final Output Tensor(s):");
        for (Tensor t : finalOutputs) {
            System.out.println(t);
        }

        sc.close();
    }
}
