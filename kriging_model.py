import math
import os
import random

import arcpy
from arcpy.sa import ExtractValuesToPoints, Kriging, KrigingModelOrdinary, RadiusVariable


INPUT_FC = (
    r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean"
    r"\Default.gdb\fishnet_30x30_with_ue_3d_interpolated_labeled"
)
GDB_PATH = os.path.dirname(INPUT_FC)

TARGET_FIELD = "SS_RSRP"
X_FIELD = "Center_X"
Y_FIELD = "Center_Y"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
CELL_SIZE = 30

TRAIN_POINTS = os.path.join(GDB_PATH, "_tmp_kriging_train_points")
TEST_POINTS = os.path.join(GDB_PATH, "_tmp_kriging_test_points")
TRAIN_POINTS_PROJECTED = os.path.join(GDB_PATH, "_tmp_kriging_train_points_projected")
TEST_POINTS_PROJECTED = os.path.join(GDB_PATH, "_tmp_kriging_test_points_projected")
PREDICT_RASTER = os.path.join(GDB_PATH, "_tmp_kriging_ss_rsrp")
TEST_EXTRACTED = os.path.join(GDB_PATH, "_tmp_kriging_test_predicted")


def require_exists(path, label):
    """Check whether an ArcGIS dataset exists."""
    if not arcpy.Exists(path):
        raise RuntimeError(f"{label} does not exist: {path}")


def require_fields(feature_class, field_names):
    """Check required fields before running the kriging workflow."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    missing_fields = [field for field in field_names if field not in existing_fields]
    if missing_fields:
        raise RuntimeError(f"Missing fields in {feature_class}: {missing_fields}")


def delete_if_exists(paths):
    """Delete temporary ArcGIS datasets from previous runs."""
    for path in paths:
        if arcpy.Exists(path):
            arcpy.management.Delete(path)


def create_point_feature_class(path, rows, spatial_reference):
    """Create a point feature class from Center_X/Center_Y and SS_RSRP values."""
    arcpy.management.CreateFeatureclass(
        out_path=os.path.dirname(path),
        out_name=os.path.basename(path),
        geometry_type="POINT",
        spatial_reference=spatial_reference,
    )
    arcpy.management.AddField(path, TARGET_FIELD, "DOUBLE")

    with arcpy.da.InsertCursor(path, ["SHAPE@XY", TARGET_FIELD]) as cursor:
        for x_value, y_value, target_value in rows:
            cursor.insertRow(((x_value, y_value), target_value))


def rmse(actual, predicted):
    """Return root mean squared error."""
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mae(actual, predicted):
    """Return mean absolute error."""
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def r2_score(actual, predicted):
    """Return coefficient of determination."""
    mean_actual = sum(actual) / len(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


arcpy.env.workspace = GDB_PATH
arcpy.env.overwriteOutput = True

require_exists(INPUT_FC, "Input feature class")
require_fields(INPUT_FC, [X_FIELD, Y_FIELD, TARGET_FIELD])

rows = []
with arcpy.da.SearchCursor(INPUT_FC, [X_FIELD, Y_FIELD, TARGET_FIELD]) as cursor:
    for x_value, y_value, target_value in cursor:
        if x_value is None or y_value is None or target_value is None:
            continue
        rows.append((float(x_value), float(y_value), float(target_value)))

if len(rows) < 10:
    raise RuntimeError(f"Not enough valid rows for kriging: {len(rows)}")

random.Random(RANDOM_SEED).shuffle(rows)
train_size = int(len(rows) * TRAIN_RATIO)
train_rows = rows[:train_size]
test_rows = rows[train_size:]

delete_if_exists(
    [
        TRAIN_POINTS,
        TEST_POINTS,
        TRAIN_POINTS_PROJECTED,
        TEST_POINTS_PROJECTED,
        PREDICT_RASTER,
        TEST_EXTRACTED,
    ]
)

try:
    arcpy.CheckOutExtension("Spatial")

    # Center_X and Center_Y are longitude/latitude values.
    wgs84 = arcpy.SpatialReference(4326)
    target_sr = arcpy.Describe(INPUT_FC).spatialReference

    create_point_feature_class(TRAIN_POINTS, train_rows, wgs84)
    create_point_feature_class(TEST_POINTS, test_rows, wgs84)

    if target_sr and target_sr.factoryCode != 4326:
        arcpy.management.Project(TRAIN_POINTS, TRAIN_POINTS_PROJECTED, target_sr)
        arcpy.management.Project(TEST_POINTS, TEST_POINTS_PROJECTED, target_sr)
        kriging_train_points = TRAIN_POINTS_PROJECTED
        kriging_test_points = TEST_POINTS_PROJECTED
    else:
        kriging_train_points = TRAIN_POINTS
        kriging_test_points = TEST_POINTS

    kriging_model = KrigingModelOrdinary("SPHERICAL")
    search_radius = RadiusVariable(12)
    prediction = Kriging(
        kriging_train_points,
        TARGET_FIELD,
        kriging_model,
        CELL_SIZE,
        search_radius,
    )
    prediction.save(PREDICT_RASTER)

    ExtractValuesToPoints(
        in_point_features=kriging_test_points,
        in_raster=PREDICT_RASTER,
        out_point_features=TEST_EXTRACTED,
        interpolate_values="INTERPOLATE",
        add_attributes="VALUE_ONLY",
    )

    actual_values = []
    predicted_values = []

    with arcpy.da.SearchCursor(TEST_EXTRACTED, [TARGET_FIELD, "RASTERVALU"]) as cursor:
        for actual, predicted in cursor:
            if actual is None or predicted is None:
                continue
            actual_values.append(float(actual))
            predicted_values.append(float(predicted))

    if not actual_values:
        raise RuntimeError("No valid kriging predictions were extracted for the test set.")

    print(f"Input feature class: {INPUT_FC}")
    print(f"Rows used: {len(rows)}")
    print(f"Train rows: {len(train_rows)}")
    print(f"Test rows: {len(test_rows)}")
    print(f"Evaluated test rows: {len(actual_values)}")
    print(f"Target field: {TARGET_FIELD}")
    print(f"Input fields: {X_FIELD}, {Y_FIELD}")
    print(f"Kriging model: Ordinary Kriging, spherical semivariogram")
    print(f"MAE: {mae(actual_values, predicted_values):.6f}")
    print(f"RMSE: {rmse(actual_values, predicted_values):.6f}")
    print(f"R2: {r2_score(actual_values, predicted_values):.6f}")
    print(f"Prediction raster: {PREDICT_RASTER}")
finally:
    # Temporary outputs are intentionally kept for inspection.
    pass
