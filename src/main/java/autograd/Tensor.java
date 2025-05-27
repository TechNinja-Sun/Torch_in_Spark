package autograd;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.util.Random;
import java.io.Serializable;



public class Tensor implements Serializable{
    private static final long serialVersionUID = 1L;

    public double[][] data;
    public double[][] grad;

    public boolean requiresGrad = false;
    private Function<double[][], double[][]> gradFn = null;
    private List<Tensor> parents = new ArrayList<>();

    // 构造函数
    public Tensor(double[][] data, boolean requiresGrad) {
        this.data = data;
        this.requiresGrad = requiresGrad;
        if (requiresGrad) {
            grad = new double[data.length][data[0].length];
        }
    }

    public Tensor(double[][] data) {
        this(data, false);
    }

    // 支持 1D 向量构造函数：自动转成 2D 行向量
    public Tensor(double[] vector) {
        this.data = new double[1][vector.length];
        for (int i = 0; i < vector.length; i++) {
            this.data[0][i] = vector[i];
        }
        this.requiresGrad = false;
    }

    // 逐元素加法
    public Tensor add(Tensor other) {
        int m = data.length;
        int n = data[0].length;
        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                result[i][j] = this.data[i][j] + other.data[i][j];

        Tensor out = new Tensor(result, this.requiresGrad || other.requiresGrad);
        if (out.requiresGrad) {
            out.parents.add(this);
            out.parents.add(other);
            out.gradFn = (gradOutput) -> gradOutput; // d(out)/d(input) = 1
        }
        return out;
    }

    // 矩阵乘法 matmul
    public Tensor matmul(Tensor other) {
        int m = data.length;
        int p = data[0].length;
        int n = other.data[0].length;

        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < p; k++)
                    result[i][j] += this.data[i][k] * other.data[k][j];

        Tensor out = new Tensor(result, this.requiresGrad || other.requiresGrad);
        if (out.requiresGrad) {
            out.parents.add(this);
            out.parents.add(other);
            out.gradFn = (gradOutput) -> {
                // dummy: real backward in backward() below
                return null;
            };
        }
        return out;
    }

    // backward 入口：触发反向传播
    public void backward() {
        if (grad == null)
            grad = new double[data.length][data[0].length];

        // 默认设置为1（loss.backward()）
        for (int i = 0; i < grad.length; i++)
            for (int j = 0; j < grad[0].length; j++)
                grad[i][j] = 1.0;

        _backward(this);
    }

    private void _backward(Tensor t) {
        if (t.gradFn == null || t.parents.isEmpty()) return;

        Tensor a = t.parents.get(0);
        Tensor b = t.parents.get(1);

        if (a.requiresGrad) {
            double[][] gradA = matmulGradA(t.grad, b.data);
            addGrad(a.grad, gradA);
            _backward(a);
        }
        if (b.requiresGrad) {
            double[][] gradB = matmulGradB(a.data, t.grad);
            addGrad(b.grad, gradB);
            _backward(b);
        }
    }

    // grad A: d(A×B)/dA = gradOutput × B^T
    private double[][] matmulGradA(double[][] gradOut, double[][] B) {
        int m = gradOut.length;
        int n = B.length;
        int p = B[0].length;
        double[][] gradA = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int k = 0; k < p; k++)
                    gradA[i][j] += gradOut[i][k] * B[j][k];
        return gradA;
    }

    // grad B: d(A×B)/dB = A^T × gradOutput
    private double[][] matmulGradB(double[][] A, double[][] gradOut) {
        int m = A[0].length;
        int n = gradOut[0].length;
        int k = A.length;
        double[][] gradB = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                for (int l = 0; l < k; l++)
                    gradB[i][j] += A[l][i] * gradOut[l][j];
        return gradB;
    }

    private void addGrad(double[][] to, double[][] delta) {
        for (int i = 0; i < to.length; i++)
            for (int j = 0; j < to[0].length; j++)
                to[i][j] += delta[i][j];
    }

    public void printData() {
        for (double[] row : data) {
            for (double v : row)
                System.out.printf("%.4f ", v);
            System.out.println();
        }
    }

    public void printGrad() {
        for (double[] row : grad) {
            for (double v : row)
                System.out.printf("%.4f ", v);
            System.out.println();
        }
    }

    public double get(int i, int j) {
        return data[i][j];
    }

    public double get(int i) {
        return data[0][i]; // 默认作为一维行向量处理
    }

    public void set(int i, int j, double value) {
        data[i][j] = value;
    }

    public static Tensor rand(int rows, int cols) {
        double[][] data = new double[rows][cols];
        Random rand = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                data[i][j] = rand.nextFloat();  // 0~1之间的随机数
            }
        }
        return new Tensor(data);
    }
    private JavaSparkContext sc;
    // 假设传入当前stage所有Tensor的RDD
    public Tensor synchronizeTensorParallel(JavaRDD<Tensor> tensorRDD) {
        // 1. 利用RDD的reduce操作实现AllReduce求和
        Tensor sumTensor = tensorRDD.reduce((t1, t2) -> t1.add(t2));

        // 2. 广播同步后的Tensor，方便所有节点使用
        Broadcast<Tensor> broadcastTensor = sc.broadcast(sumTensor);

        // 3. 返回广播的Tensor（一般调用端可以直接用这个结果）
        return broadcastTensor.getValue();
    }


}
