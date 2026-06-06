import math
import os

import arcpy


# Target feature class after Height_AGL correction and outlier deletion.
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")
feature_class = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d_height_agl")

source_field = "source_file"
height_agl_field = "Height_AGL"
altitude_level_field = "UAV_Altitude_Level"


def require_exists(path, label):
    """Check whether an ArcGIS dataset or normal file exists."""
    if label == "ArcGIS Pro project":
        exists = os.path.exists(path)
    else:
        exists = arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label} does not exist: {path}")


def field_map(feature_class_path):
    """Return a field-name to arcpy.Field mapping."""
    return {field.name: field for field in arcpy.ListFields(feature_class_path)}


def require_fields(feature_class_path, field_names):
    """Check that the target feature class contains required fields."""
    existing_fields = field_map(feature_class_path)
    missing_fields = [field for field in field_names if field not in existing_fields]
    if missing_fields:
        raise RuntimeError(f"Missing fields in {feature_class_path}: {missing_fields}")


def add_text_field_if_missing(feature_class_path, field_name, length=20):
    """Add a TEXT label field if it does not already exist."""
    existing_fields = field_map(feature_class_path)
    if field_name not in existing_fields:
        try:
            arcpy.management.AddField(feature_class_path, field_name, "TEXT", field_length=length)
        except arcpy.ExecuteError as exc:
            raise RuntimeError(
                f"Cannot add field '{field_name}' to {feature_class_path}. "
                "Close its map layer, attribute table, or edit session in ArcGIS Pro, then run again."
            ) from exc
        return

    if existing_fields[field_name].type not in ("String", "Text"):
        raise RuntimeError(f"Existing field '{field_name}' is not a text field.")


def altitude_label(height_agl):
    """Classify Height_AGL by 10 m steps; values below 10 m are labeled 5m."""
    if height_agl is None:
        return None

    height_agl = float(height_agl)
    if height_agl < 10:
        return "5m"

    level = int(math.floor(height_agl / 10.0) * 10)
    return f"{level}m"


arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro project")
require_exists(gdb_path, "Geodatabase")
require_exists(feature_class, "Feature class")
require_fields(feature_class, [source_field, height_agl_field])
add_text_field_if_missing(feature_class, altitude_level_field)

total_count = 0
ground_cleared_count = 0
uav_labeled_count = 0
null_height_count = 0

fields = [source_field, height_agl_field, altitude_level_field]

with arcpy.da.UpdateCursor(feature_class, fields) as cursor:
    for row in cursor:
        source_value, height_agl, _ = row
        total_count += 1

        # Ground records are excluded from UAV height labels.
        if source_value is not None and str(source_value).lower() == "ground":
            row[1] = None
            row[2] = None
            ground_cleared_count += 1
        else:
            row[2] = altitude_label(height_agl)
            if row[2] is None:
                null_height_count += 1
            else:
                uav_labeled_count += 1

        cursor.updateRow(row)

print(f"Feature class: {feature_class}")
print(f"Total rows: {total_count}")
print(f"Ground rows with Height_AGL and UAV_Altitude_Level cleared: {ground_cleared_count}")
print(f"UAV rows labeled: {uav_labeled_count}")
print(f"Rows with NULL UAV_Altitude_Level: {null_height_count + ground_cleared_count}")
print(f"Label field: {altitude_level_field}")
