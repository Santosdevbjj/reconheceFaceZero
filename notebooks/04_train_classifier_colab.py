# %%
"""Notebook 04 - Treinar classificador (SVM)
- Carrega embeddings e labels
- Treina SVM e salva modelo pickled
"""
# %%
import os
import numpy as np
from sklearn.svm import SVC
import joblib

DRIVE_ROOT = '/content/drive/MyDrive/reconheceFaceZero'
EMB_PATH = os.path.join(DRIVE_ROOT, 'models', 'embeddings.npy')
LBL_PATH = os.path.join(DRIVE_ROOT, 'models', 'labels.npy')
CLASS_MODEL = os.path.join(DRIVE_ROOT, 'models', 'classifier', 'svm_face_recog.pkl')

embs = np.load(EMB_PATH, allow_pickle=True)
labels = np.load(LBL_PATH, allow_pickle=True)

clf = SVC(kernel='linear', probability=True)
clf.fit(embs, labels)

os.makedirs(os.path.dirname(CLASS_MODEL), exist_ok=True)
joblib.dump(clf, CLASS_MODEL)
print('Classificador salvo em', CLASS_MODEL)
