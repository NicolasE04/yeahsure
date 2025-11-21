import numpy as np
import pandas as pd
from math import atan2, sqrt

from scipy.io import arff
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


RAW_ARFF = "train.arff"
CAL_ARFF = "test_calibrated.arff"   # calibrated version of same data


# Helper
def load_arff_to_df(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
    return df

def complementary_filter(acc, gyro, fs=104.0, alpha=0.98):
    dt = 1.0 / fs
    gyro_r = np.deg2rad(gyro)

    N = len(acc)
    roll = np.zeros(N)
    pitch = np.zeros(N)
    yaw = np.zeros(N)

    ax0, ay0, az0 = acc[0]
    roll[0] = atan2(ay0, az0)
    pitch[0] = atan2(-ax0, sqrt(ay0**2 + az0**2))

    for i in range(1, N):
        ax, ay, az = acc[i]
        gx, gy, gz = gyro_r[i]

        # integrate gyro
        roll_gyro = roll[i-1] + gx*dt
        pitch_gyro = pitch[i-1] + gy*dt
        yaw_gyro = yaw[i-1] + gz*dt

        # accel-based
        roll_acc = atan2(ay, az)
        pitch_acc = atan2(-ax, sqrt(ay**2 + az**2))

        roll[i] = alpha*roll_gyro + (1-alpha)*roll_acc
        pitch[i] = alpha*pitch_gyro + (1-alpha)*pitch_acc
        yaw[i] = yaw_gyro

    return np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)

def print_model_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    mae  = mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2   = r2_score(y_true, y_pred, multioutput='raw_values')

    comps = ["Roll", "Pitch", "Yaw"]
    print(f"\n{name.upper()}")
    for i, comp in enumerate(comps):
        print(f"{comp}:")
        print(f"  RMSE: {rmse[i]:.4f}")
        print(f"  MAE:  {mae[i]:.4f}")
        print(f"  R²:   {r2[i]:.4f}")


# Data
raw_df = load_arff_to_df(RAW_ARFF)
cal_df = load_arff_to_df(CAL_ARFF)
assert len(raw_df) == len(cal_df), "raw and calibrated must align row-by-row"

feature_cols = ["gyro_x","gyro_y","gyro_z",
                "mag_x","mag_y","mag_z",
                "acc_x","acc_y","acc_z"]

X_raw = raw_df[feature_cols].astype(float).values

# calibrated data -> math Euler
acc_cal  = cal_df[["acc_x","acc_y","acc_z"]].astype(float).values
gyro_cal = cal_df[["gyro_x","gyro_y","gyro_z"]].astype(float).values

roll_math, pitch_math, yaw_math = complementary_filter(acc_cal, gyro_cal)
y_math = np.vstack([roll_math, pitch_math, yaw_math]).T   # ground truth


# Train models
lin_reg = LinearRegression().fit(X_raw, y_math)

poly_reg = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin", LinearRegression())
]).fit(X_raw, y_math)

tree_reg = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_raw, y_math)


#  Predictions
y_lin  = lin_reg.predict(X_raw)
y_poly = poly_reg.predict(X_raw)
y_tree = tree_reg.predict(X_raw)


# Compare
print_model_metrics("Linear Regression",      y_math, y_lin)
print_model_metrics("Polynomial Regression",  y_math, y_poly)
print_model_metrics("Decision Tree",          y_math, y_tree)
