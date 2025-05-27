import dataset.MNISTLoader;
import engine.Trainer;
import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;

public class Main_test {
    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("SparkTorchExperiment").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        String mnistPath = "data/mnist.csv";
        Tuple2<JavaRDD<float[]>, JavaRDD<Integer>> data = MNISTLoader.load(sc, mnistPath);
        JavaRDD<float[]> features = data._1();
        JavaRDD<Integer> labels = data._2();

        Trainer trainer = new Trainer(10, 0.01f);
        trainer.train(features, labels);

        sc.close();
    }
}
