import arcpy


feature_class = (
    r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean"
    r"\Default.gdb\fishnet_30x30_with_ue_3d_interpolated_labeled"
)

row_field = "Grid_Row"
col_field = "Grid_Col"
expected_grid_count = 100
coordinate_round_digits = 6


def require_exists(path, label):
    """Check whether an ArcGIS dataset exists."""
    if not arcpy.Exists(path):
        raise RuntimeError(f"{label} does not exist: {path}")


def add_long_field_if_missing(feature_class_path, field_name):
    """Add a LONG integer field if it does not already exist."""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class_path)}
    if field_name not in existing_fields:
        try:
            arcpy.management.AddField(feature_class_path, field_name, "LONG")
        except arcpy.ExecuteError as exc:
            raise RuntimeError(
                f"Cannot add field '{field_name}' to {feature_class_path}. "
                "Close its map layer, attribute table, or edit session in ArcGIS Pro, then run again."
            ) from exc


def get_center_xy(geometry):
    """Return centroid XY in the feature class coordinate system."""
    if geometry is None:
        return None, None
    center = geometry.centroid
    if center is None:
        return None, None
    return center.X, center.Y


require_exists(feature_class, "Feature class")
add_long_field_if_missing(feature_class, row_field)
add_long_field_if_missing(feature_class, col_field)

center_records = []

with arcpy.da.SearchCursor(feature_class, ["OID@", "SHAPE@"]) as cursor:
    for oid, geometry in cursor:
        center_x, center_y = get_center_xy(geometry)
        if center_x is None or center_y is None:
            continue
        center_records.append(
            (
                oid,
                round(center_x, coordinate_round_digits),
                round(center_y, coordinate_round_digits),
            )
        )

if not center_records:
    raise RuntimeError(f"No valid grid geometries found in {feature_class}")

# Build row and column indexes from the fishnet geometry itself:
# - columns are unique center X values sorted from left to right
# - rows are unique center Y values sorted from top to bottom
unique_x_values = sorted({record[1] for record in center_records})
unique_y_values = sorted({record[2] for record in center_records}, reverse=True)

if len(unique_x_values) != expected_grid_count or len(unique_y_values) != expected_grid_count:
    raise RuntimeError(
        "Unexpected fishnet dimensions. "
        f"Expected {expected_grid_count} columns and {expected_grid_count} rows, "
        f"got {len(unique_x_values)} columns and {len(unique_y_values)} rows."
    )

col_index_by_x = {x_value: index + 1 for index, x_value in enumerate(unique_x_values)}
row_index_by_y = {y_value: index + 1 for index, y_value in enumerate(unique_y_values)}

index_by_oid = {
    oid: (row_index_by_y[center_y], col_index_by_x[center_x])
    for oid, center_x, center_y in center_records
}

updated_count = 0
skipped_count = 0

fields = ["OID@", row_field, col_field]

with arcpy.da.UpdateCursor(feature_class, fields) as cursor:
    for row in cursor:
        oid = row[0]

        if oid not in index_by_oid:
            row[1] = None
            row[2] = None
            skipped_count += 1
            cursor.updateRow(row)
            continue

        row[1], row[2] = index_by_oid[oid]
        cursor.updateRow(row)
        updated_count += 1

total_count = int(arcpy.management.GetCount(feature_class)[0])

print(f"Feature class: {feature_class}")
print(f"Total rows: {total_count}")
print(f"Updated grid index rows: {updated_count}")
print(f"Skipped rows without geometry: {skipped_count}")
print(f"Detected rows: {len(unique_y_values)}")
print(f"Detected columns: {len(unique_x_values)}")
print(f"Row field: {row_field}")
print(f"Column field: {col_field}")
