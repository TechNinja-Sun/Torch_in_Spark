package dataset;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

public class MNISTLoader {
    public static Tuple2<JavaRDD<float[]>, JavaRDD<Integer>> load(JavaSparkContext sc, String path) {
        JavaRDD<String> lines = sc.textFile(path);

        JavaRDD<float[]> features = lines.map(line -> {
            String[] tokens = line.split(",");
            float[] pixels = new float[tokens.length - 1];
            for (int i = 0; i < tokens.length - 1; i++) {
                pixels[i] = Float.parseFloat(tokens[i]);
            }
            return pixels;
        });

        JavaRDD<Integer> labels = lines.map(line -> {
            String[] tokens = line.split(",");
            return Integer.parseInt(tokens[tokens.length - 1]);
        });

        return new Tuple2<>(features, labels);
    }
}
