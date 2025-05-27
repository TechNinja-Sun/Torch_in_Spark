package autograd;
import java.io.Serializable;


public class DenseLayer implements Serializable{
    private static final long serialVersionUID = 1L;

    private int inputSize;
    private int outputSize;
    public double[][] weights;
    public double[] biases;

    private transient Tensor input; // transient 避免序列化失败
    public double[][] gradWeights;
    public double[] gradBiases;

    public DenseLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        weights = new double[outputSize][inputSize];
        biases = new double[outputSize];
        gradWeights = new double[outputSize][inputSize];
        gradBiases = new double[outputSize];

        // 初始化权重为小随机数
        for (int i = 0; i < outputSize; i++)
            for (int j = 0; j < inputSize; j++)
                weights[i][j] = Math.random() * 0.01;
    }

    public Tensor forward(Tensor input) {
        this.input = input;
        double[] out = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            out[i] = biases[i];
            for (int j = 0; j < inputSize; j++) {
                out[i] += weights[i][j] * input.get(j);
            }
        }
        return new Tensor(out);
    }

    public Tensor backward(Tensor gradOutput) {
        // 计算梯度
        double[] gradInput = new double[inputSize];

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                gradWeights[i][j] += gradOutput.get(i) * input.get(j);
                gradInput[j] += gradOutput.get(i) * weights[i][j];
            }
            gradBiases[i] += gradOutput.get(i);
        }

        return new Tensor(gradInput);
    }

    public DenseLayer copy() {
        DenseLayer clone = new DenseLayer(inputSize, outputSize);
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                clone.weights[i][j] = this.weights[i][j];
            }
            clone.biases[i] = this.biases[i];
        }
        return clone;
    }
    
    public void printWeightGrad() {
        System.out.println("权重梯度:");
        if (gradWeights != null) {
            for (int i = 0; i < gradWeights.length; i++) {
                for (int j = 0; j < gradWeights[0].length; j++) {
                    System.out.printf("%.6f ", gradWeights[i][j]);
                }
                System.out.println();
            }
        } else {
            System.out.println("权重梯度为空");
        }

        System.out.println("偏置梯度:");
        if (gradBiases != null) {
            for (int i = 0; i < gradBiases.length; i++) {
                System.out.printf("%.6f ", gradBiases[i]);
            }
            System.out.println();
        } else {
            System.out.println("偏置梯度为空");
        }
    }

}
