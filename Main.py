import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# === Load Data ===
joint_angles = pd.read_csv("Subject_1_Seq_1_JointAngles_N.csv")
gait_cycle = pd.read_csv("Subject_1_Seq_1_GaitCycle_N.csv")

# === Smooth Ankle Distance (like JRD) ===
ankle_diff = gait_cycle.iloc[:, 0].rolling(window=5).mean()

# === Detect Peaks ===
peaks, _ = find_peaks(ankle_diff.dropna(), prominence=0.13, width=(7, 18))
print("Detected Peaks:", peaks)

if len(peaks) >= 3:
    frame_start = peaks[0]
    frame_end = peaks[2]
    print("Gait cycle frames:", frame_start, "to", frame_end)
else:
    raise ValueError("Insufficient peaks detected for gait cycle.")

# === Smooth Knee Angles and Get Min ===
left_knee = joint_angles.iloc[:, 0].rolling(window=5).mean()
right_knee = joint_angles.iloc[:, 1].rolling(window=5).mean()

min_left_knee = left_knee[frame_start:frame_end].min()
min_right_knee = right_knee[frame_start:frame_end].min()

print(f"Min Left Knee Angle: {min_left_knee:.2f}")
print(f"Min Right Knee Angle: {min_right_knee:.2f}")

# === Prepare Dataset (Example with Dummy Labels) ===
X = [[min_left_knee, min_right_knee]]
y = ['Normal']  # Replace with real labels for multi-subject training

# === Dummy Expansion (add more subjects later) ===
# For now, duplicate row for demo purposes
X *= 10
y = ['Normal']*5 + ['Left limp']*3 + ['Right limp']*2

# === Classification (Exp 1) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n--- LDA Classifier ---")
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
preds = lda.predict(X_test)
print(classification_report(y_test, preds))

print("\n--- SVM Classifier ---")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
preds = svm.predict(X_test)
print(classification_report(y_test, preds))

print("\n--- KNN Classifier ---")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
print(classification_report(y_test, preds))
