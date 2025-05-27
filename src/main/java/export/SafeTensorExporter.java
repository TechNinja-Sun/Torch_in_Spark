package export;

import java.io.FileOutputStream;
import java.io.IOException;

public class SafeTensorExporter {
    public static void export(float[][] data, String path) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            for (float[] row : data) {
                for (float value : row) {
                    fos.write(Float.floatToIntBits(value));
                }
            }
        }
    }
}
