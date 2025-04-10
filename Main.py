import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ====================================
# Preprocessing: Clean and Load Raw Joint Data
# ====================================
with open('Data/Subject_1_Seq_1_GeometricPositions_N.csv', 'r') as infile, \
     open('Data/cleaned_file.csv', 'w') as outfile:
    for line in infile:
        cleaned_line = line.rstrip(',\n') + '\n'
        outfile.write(cleaned_line)

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

positions = pd.read_csv("Data/cleaned_file.csv", header=None, names=columns)
gait_cycle = pd.read_csv("Data/Subject_1_Seq_1_GaitCycle_N.csv")

# ====================================
# Compute Stride Lengths
# ====================================
ankle_distance = gait_cycle.iloc[:, 0].rolling(window=5).mean()
peaks, _ = find_peaks(ankle_distance.dropna(), prominence=0.13, width=(7, 18))

if len(peaks) < 3:
    raise ValueError("Not enough gait cycles detected.")

frame_start, frame_end = peaks[0], peaks[2]
foot_left_z = pd.to_numeric(positions['Foot_Left_Z'], errors='coerce')
foot_right_z = pd.to_numeric(positions['Foot_Right_Z'], errors='coerce')

stride_left = abs(foot_left_z.iloc[frame_end] - foot_left_z.iloc[frame_start])
stride_right = abs(foot_right_z.iloc[frame_end] - foot_right_z.iloc[frame_start])
print(f"Stride Length - Left: {stride_left:.2f}, Right: {stride_right:.2f}")

# ====================================
# Load Labeled Gait Data
# ====================================
df = pd.read_csv("Data/Combined_TData.csv")
df['Status'] = df['Status'].replace({'No Limp': 'Normal'})
df = df.dropna(subset=['Min_R_Knee_Angle', 'Min_L_Knee_Angle', 'Status'])

# ====================================
# Experiment 1: Classification on Knee Angles
# ====================================
X = df[['Min_R_Knee_Angle', 'Min_L_Knee_Angle']]
y = df['Status']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.30, random_state=0)

classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'SVM': SVC(C=5, gamma=0.001),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

print("\n=== Experiment 1: Knee Angles Only ===")
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n{name} Classification Report")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Decision boundary plot
    x_min, x_max = X['Min_R_Knee_Angle'].min() - 1, X['Min_R_Knee_Angle'].max() + 1
    y_min, y_max = X['Min_L_Knee_Angle'].min() - 1, X['Min_L_Knee_Angle'].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid_df = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)
    Z = clf.predict(grid_df).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X_test['Min_R_Knee_Angle'], X_test['Min_L_Knee_Angle'],
                          c=y_test, cmap='Set1', edgecolor='k')
    plt.title(f"{name} Classifier - Knee Angle Scatter")
    plt.xlabel('Min_R_Knee_Angle')
    plt.ylabel('Min_L_Knee_Angle')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.gca().add_artist(legend1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ====================================
# Experiment 2: Add Stride Length Features
# ====================================
df['Stride_Left'] = stride_left
df['Stride_Right'] = stride_right

X2 = df[['Min_R_Knee_Angle', 'Min_L_Knee_Angle', 'Stride_Left', 'Stride_Right']]
y2 = df['Status']
y_encoded2 = le.fit_transform(y2)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y_encoded2, test_size=0.30, random_state=0)

print("\n=== Experiment 2: With Stride Features ===")
for name, clf in classifiers.items():
    clf.fit(X_train2, y_train2)
    y_pred2 = clf.predict(X_test2)

    print(f"\n{name} Classification Report")
    print(classification_report(y_test2, y_pred2, target_names=le.classes_))

# ====================================
# Experiment 3: Hip Joint Movement Analysis
# ====================================

hip_right_y = pd.to_numeric(positions['Hip_Right_Y'], errors='coerce')
hip_left_y = pd.to_numeric(positions['Hip_Left_Y'], errors='coerce')
hip_symmetry = abs(hip_right_y - hip_left_y)

# Plot hip vertical position
plt.figure(figsize=(10, 5))
plt.plot(hip_right_y, label='Hip Right Y')
plt.plot(hip_left_y, label='Hip Left Y')
plt.title("Hip Joint Vertical Position Over Time")
plt.xlabel("Frame")
plt.ylabel("Y Position")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot symmetry (Y difference between hips)
plt.figure()
plt.plot(hip_symmetry)
plt.title("Vertical Asymmetry Between Hips")
plt.xlabel("Frame")
plt.ylabel("Y-Difference (Right - Left)")
plt.grid(True)
plt.tight_layout()
plt.show()
