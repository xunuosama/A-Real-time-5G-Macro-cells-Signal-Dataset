import os

import arcpy
from arcpy.sa import ExtractMultiValuesToPoints


# ArcGIS Pro project path. The target geodatabase is the project's Default.gdb.
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

# Keep the source feature class unchanged. DEM and Height_AGL values are written
# only to this copied output feature class.
input_points = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d")
output_points = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d_height_agl")
dem_raster = r"E:\arcgis_workspace\data\Chongqing_DEM_utm48n.tif"

# ALT_M_ is in meters. The DEM raster stores elevation in centimeters, so the
# sampled DEM value is divided by 100 before writing DEM_center in meters.
# Height_AGL is calculated as ALT_M_ - DEM_center, both in meters.
alt_field = "ALT_M_"
dem_field = "DEM_center"
height_agl_field = "Height_AGL"
dem_cm_to_m = 100.0


def require_exists(path, label):
    """Check whether an ArcGIS dataset or normal file exists."""
    if label in ("ArcGIS Pro project", "DEM raster"):
        exists = os.path.exists(path)
    else:
        exists = arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label} does not exist: {path}")


def require_fields(feature_class, field_names):
    """Check that the input feature class contains required fields."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    missing_fields = [field for field in field_names if field not in existing_fields]
    if missing_fields:
        raise RuntimeError(f"Missing fields in {feature_class}: {missing_fields}")


def delete_field_if_exists(feature_class, field_name):
    """Delete old result fields before rerunning the DEM sampling workflow."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if field_name in existing_fields:
        arcpy.management.DeleteField(feature_class, field_name)


def add_double_field_if_missing(feature_class, field_name):
    """Add a DOUBLE field only when it is missing."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if field_name not in existing_fields:
        arcpy.management.AddField(feature_class, field_name, "DOUBLE")


arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro project")
require_exists(gdb_path, "Geodatabase")
require_exists(input_points, "Input feature class")
require_exists(dem_raster, "DEM raster")
require_fields(input_points, [alt_field])

# ExtractMultiValuesToPoints requires the Spatial Analyst extension.
arcpy.CheckOutExtension("Spatial")

# Rebuild the output feature class on each run. If ArcGIS Pro has this layer or
# its attribute table open, Delete may fail because of a schema lock.
if arcpy.Exists(output_points):
    try:
        arcpy.management.Delete(output_points)
    except arcpy.ExecuteError as exc:
        raise RuntimeError(
            f"Cannot delete existing output feature class: {output_points}. "
            "Close its map layer or attribute table in ArcGIS Pro, then run again."
        ) from exc

# Copy first so the original UE_Points_proj_48N_with_base_matched_3d is not
# modified by DEM sampling or Height_AGL calculation.
arcpy.management.CopyFeatures(input_points, output_points)

delete_field_if_exists(output_points, dem_field)
delete_field_if_exists(output_points, height_agl_field)

# Sample the DEM raster at each point. ArcGIS writes the raw DEM value in cm to
# DEM_center first; the update cursor below converts it to meters in place.
ExtractMultiValuesToPoints(output_points, [[dem_raster, dem_field]], "NONE")
add_double_field_if_missing(output_points, height_agl_field)

updated_count = 0
skipped_count = 0

with arcpy.da.UpdateCursor(output_points, [alt_field, dem_field, height_agl_field]) as cursor:
    for row in cursor:
        alt_value, dem_value_cm, _ = row

        # Leave Height_AGL NULL when either input value is NULL.
        if alt_value is None or dem_value_cm is None:
            row[1] = None
            row[2] = None
            skipped_count += 1
        else:
            dem_value_m = float(dem_value_cm) / dem_cm_to_m
            row[1] = round(dem_value_m, 2)
            row[2] = round(float(alt_value) - dem_value_m, 2)
            updated_count += 1

        cursor.updateRow(row)

total_count = int(arcpy.management.GetCount(output_points)[0])

print(f"Input feature class: {input_points}")
print(f"Output feature class: {output_points}")
print(f"DEM raster: {dem_raster}")
print("DEM_center unit: meters")
print("Height_AGL unit: meters")
print(f"Total rows: {total_count}")
print(f"Updated Height_AGL rows: {updated_count}")
print(f"Skipped rows with NULL ALT_M_ or DEM_center: {skipped_count}")
