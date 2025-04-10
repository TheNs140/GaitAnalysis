import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# === Load Data ===
#joint_angles = pd.read_csv("Data/Subject_1_Seq_1_JointAngles_N.csv")
#Must first clean the geometric positions file

with open('Data/Subject_1_Seq_1_GeometricPositions_N.csv', 'r') as infile, \
     open('Data/cleaned_file.csv', 'w') as outfile:
    for line in infile:
        # Strip any trailing comma at the end of the line
        cleaned_line = line.rstrip(',\n') + '\n'
        outfile.write(cleaned_line)

# Manually set column names since original headers are broken
columns = [
    'Foot_Left_X', 'Foot_Left_Y', 'Foot_Left_Z',
    'Ankle_Left_X', 'Ankle_Left_Y', 'Ankle_Left_Z',
    'Knee_Left_X', 'Knee_Left_Y', 'Knee_Left_Z',
    'Hip_Left_X', 'Hip_Left_Y', 'Hip_Left_Z',
    'Foot_Right_X', 'Foot_Right_Y', 'Foot_Right_Z',
    'Ankle_Right_X', 'Ankle_Right_Y', 'Ankle_Right_Z',
    'Knee_Right_X', 'Knee_Right_Y', 'Knee_Right_Z',
    'Hip_Right_X', 'Hip_Right_Y', 'Hip_Right_Z',
    'Hip_Center_X', 'Hip_Center_Y', 'Hip_Center_Z'

]

# Load with no header, then apply column names
positions = pd.read_csv("Data/cleaned_file.csv", header=None, names=columns)

gait_cycle = pd.read_csv("Data/Subject_1_Seq_1_GaitCycle_N.csv")

#ankle_diff = gait_cycle.iloc[:, 0].rolling(window=5).mean()

#Detect Ankle Distance
ankle_distance = gait_cycle.iloc[:, 0].rolling(window=5).mean()
peaks, _ = find_peaks(ankle_distance.dropna(), prominence=0.13, width=(7, 18))

""" # === Detect Peaks ===
peaks, _ = find_peaks(ankle_diff.dropna(), prominence=0.13, width=(7, 18))
print("Detected Peaks:", peaks) """

if len(peaks) < 3:
    raise ValueError("Not enough gait cycles detected.")
frame_start = peaks[0]
frame_end = peaks[2]

foot_left_z = positions['Foot_Left_Z']
foot_right_z = positions['Foot_Right_Z']

foot_left_z = pd.to_numeric(positions['Foot_Left_Z'], errors='coerce')
foot_right_z = pd.to_numeric(positions['Foot_Right_Z'], errors='coerce')

stride_left = abs(foot_left_z.iloc[frame_end] - foot_left_z.iloc[frame_start])
stride_right = abs(foot_right_z.iloc[frame_end] - foot_right_z.iloc[frame_start])

print(f"Stride Length - Left: {stride_left:.2f}, Right: {stride_right:.2f}")




""" print("\n--- LDA Classifier ---")
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
print(classification_report(y_test, preds)) """
