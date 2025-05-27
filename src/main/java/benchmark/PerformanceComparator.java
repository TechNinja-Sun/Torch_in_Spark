package benchmark;

public class PerformanceComparator {
    public static void compare(long javaTime, long pytorchTime) {
        System.out.println("Java Spark Time: " + javaTime + " ms");
        System.out.println("PyTorch Time: " + pytorchTime + " ms");
        System.out.println("Speedup: " + ((double) pytorchTime / javaTime));
    }
}
