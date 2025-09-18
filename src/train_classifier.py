"""
Treina e salva um classificador sklearn a partir de embeddings e labels
"""
import joblib
from sklearn.svm import SVC


def train_svm(embeddings, labels, out_path):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(embeddings, labels)
    joblib.dump(clf, out_path)
    return out_path
