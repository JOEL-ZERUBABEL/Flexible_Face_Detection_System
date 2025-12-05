import numpy as np
class EmbeddingUtils:

    @staticmethod
    def l2_normalize(emb: np.ndarray) -> np.ndarray:
        """Normalize embedding using L2 norm."""
        if emb is None or emb.size == 0:
            return np.zeros((1,), dtype=np.float32)

        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    @staticmethod
    def to_numpy(emb_list):
        """Convert list to numpy array cleanly."""
        return np.array(emb_list, dtype=np.float32)

    @staticmethod
    def to_list(emb: np.ndarray):
        """Convert numpy embedding to list for JSON/pickle."""
        return emb.astype(float).tolist()

    @staticmethod
    def from_list(lst):
        """Convert list back to embedding np array."""
        return np.array(lst, dtype=np.float32)