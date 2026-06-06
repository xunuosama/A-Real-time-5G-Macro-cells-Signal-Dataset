import arcpy


feature_class = (
    r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean"
    r"\Default.gdb\fishnet_30x30_with_ue_3d_interpolated_labeled"
)

center_x_field = "Center_X"
center_y_field = "Center_Y"
wgs84 = arcpy.SpatialReference(4326)


def require_exists(path, label):
    """Check whether an ArcGIS dataset exists."""
    if not arcpy.Exists(path):
        raise RuntimeError(f"{label} does not exist: {path}")


def add_double_field_if_missing(feature_class_path, field_name):
    """Add a DOUBLE field if it does not already exist."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class_path)}
    if field_name not in existing_fields:
        arcpy.management.AddField(feature_class_path, field_name, "DOUBLE")


require_exists(feature_class, "Feature class")
add_double_field_if_missing(feature_class, center_x_field)
add_double_field_if_missing(feature_class, center_y_field)

desc = arcpy.Describe(feature_class)
source_sr = desc.spatialReference

updated_count = 0
skipped_count = 0

fields = ["SHAPE@", center_x_field, center_y_field]

with arcpy.da.UpdateCursor(feature_class, fields) as cursor:
    for row in cursor:
        geometry = row[0]

        if geometry is None:
            row[1] = None
            row[2] = None
            skipped_count += 1
            cursor.updateRow(row)
            continue

        center_point = geometry.centroid
        center_geometry = arcpy.PointGeometry(center_point, source_sr)

        if source_sr and source_sr.factoryCode != 4326:
            center_geometry = center_geometry.projectAs(wgs84)

        projected_point = center_geometry.firstPoint
        row[1] = projected_point.X
        row[2] = projected_point.Y
        cursor.updateRow(row)
        updated_count += 1

total_count = int(arcpy.management.GetCount(feature_class)[0])

print(f"Feature class: {feature_class}")
print(f"Total rows: {total_count}")
print(f"Updated center rows: {updated_count}")
print(f"Skipped rows without geometry: {skipped_count}")
print(f"Center_X: longitude in WGS84")
print(f"Center_Y: latitude in WGS84")
