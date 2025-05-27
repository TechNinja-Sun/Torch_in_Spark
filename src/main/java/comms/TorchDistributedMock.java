package comms;

public class TorchDistributedMock {
    private static int worldSize = 1;
    private static int rank = 0;
    private static boolean initialized = false;

    public static void initProcessGroup(int size, int currentRank) {
        worldSize = size;
        rank = currentRank;
        initialized = true;
        System.out.println("[TorchDistributedMock] Initialized. Rank: " + rank + " / " + worldSize);
    }

    public static int getWorldSize() {
        ensureInitialized();
        return worldSize;
    }

    public static int getRank() {
        ensureInitialized();
        return rank;
    }

    public static boolean isInitialized() {
        return initialized;
    }

    private static void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("TorchDistributedMock not initialized.");
        }
    }
}
