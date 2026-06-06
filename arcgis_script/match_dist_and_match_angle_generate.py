import os
from math import asin, atan2, cos, degrees, hypot, radians, sin, sqrt

import arcpy


def haversine(lon1, lat1, lon2, lat2):
    """Return geodesic distance in meters for WGS84-like lon/lat coordinates."""
    radius = 6371000.0
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def geodesic_azimuth(lon1, lat1, lon2, lat2):
    """Return initial bearing from point 1 to point 2, degrees clockwise from north."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    return (degrees(atan2(x, y)) + 360.0) % 360.0


def planar_azimuth(x1, y1, x2, y2):
    """Return projected-coordinate bearing, degrees clockwise from grid north."""
    return (degrees(atan2(x2 - x1, y2 - y1)) + 360.0) % 360.0


def point_xy(geometry):
    """Return representative XY for point or non-point geometry."""
    if geometry is None:
        return None, None
    point = geometry.firstPoint if geometry.type == "point" else geometry.centroid
    if point is None:
        return None, None
    return point.X, point.Y


def same_spatial_reference(sr_a, sr_b):
    if sr_a is None or sr_b is None:
        return False
    if sr_a.factoryCode and sr_b.factoryCode:
        return sr_a.factoryCode == sr_b.factoryCode
    return sr_a.name == sr_b.name


def add_base_field(feature_class, source_field, existing_names):
    """Add a Base_ copy field and return its output field name."""
    out_name = f"Base_{source_field.name}"[:64]
    original_name = out_name
    suffix = 1
    while out_name in existing_names:
        suffix_text = f"_{suffix}"
        out_name = f"{original_name[:64 - len(suffix_text)]}{suffix_text}"
        suffix += 1

    if source_field.type in ("String", "Text"):
        arcpy.management.AddField(feature_class, out_name, "TEXT", field_length=source_field.length)
    elif source_field.type in ("Double", "Single", "Integer", "SmallInteger"):
        arcpy.management.AddField(feature_class, out_name, "DOUBLE")
    elif source_field.type == "Date":
        arcpy.management.AddField(feature_class, out_name, "DATE")
    else:
        arcpy.management.AddField(feature_class, out_name, "TEXT")

    existing_names.add(out_name)
    return out_name


def to_output_value(value, source_type):
    if value is None:
        return None
    if source_type in ("Double", "Single", "Integer", "SmallInteger"):
        return float(value)
    if source_type == "Date":
        return value
    return str(value)


def normalize_match_key(value):
    if value is None:
        return None
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    return value


def require_fields(feature_class, field_names):
    existing = {field.name for field in arcpy.ListFields(feature_class)}
    missing = [field_name for field_name in field_names if field_name not in existing]
    if missing:
        raise RuntimeError(f"Missing fields in {feature_class}: {missing}")


# Paths
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_proj = os.path.join(base_path, "Default.gdb")

feature_a = os.path.join(gdb_proj, "UE_Points_proj_48N_fixed")
feature_out = os.path.join(gdb_proj, "UE_Points_proj_48N_with_base")
feature_b = os.path.join(gdb_proj, "Base_Points_UTM48N")

ue_key_field = "NCI"
base_key_field = "ECI"


if not os.path.exists(aprx_path):
    raise RuntimeError(f"ArcGIS Pro project does not exist: {aprx_path}")

if not arcpy.Exists(gdb_proj):
    raise RuntimeError(f"Geodatabase does not exist: {gdb_proj}")

arcpy.env.workspace = gdb_proj
arcpy.env.overwriteOutput = True

for feature_class in (feature_a, feature_b):
    if not arcpy.Exists(feature_class):
        raise RuntimeError(f"Feature class does not exist: {feature_class}")

require_fields(feature_a, [ue_key_field])
require_fields(feature_b, [base_key_field])


# Copy UE points so the original data remains unchanged.
if arcpy.Exists(feature_out):
    try:
        arcpy.management.Delete(feature_out)
    except arcpy.ExecuteError as exc:
        raise RuntimeError(
            f"Failed to delete existing output feature class: {feature_out}. "
            "Close any map/table view using this layer, then run again."
        ) from exc
arcpy.management.CopyFeatures(feature_a, feature_out)
print("Created output feature class.")


a_sr = arcpy.Describe(feature_out).spatialReference
b_sr = arcpy.Describe(feature_b).spatialReference
can_project_base = bool(a_sr and a_sr.name != "Unknown")
can_project_between_layers = bool(can_project_base and b_sr and b_sr.name != "Unknown")
if can_project_between_layers and not same_spatial_reference(a_sr, b_sr):
    print(f"Spatial references differ. Base geometries will be projected to: {a_sr.name}")
elif can_project_base and not same_spatial_reference(a_sr, b_sr):
    print("Warning: base spatial reference is unknown; base geometries will not be projected.")

if a_sr is None or a_sr.name == "Unknown":
    print("Warning: output spatial reference is unknown; distance may be unreliable.")


# Read base-station fields. Skip geometry/OID and unsupported binary-like fields.
skip_types = {"OID", "Geometry", "Blob", "Raster", "GUID", "GlobalID", "XML"}
b_source_fields = [f for f in arcpy.ListFields(feature_b) if f.type not in skip_types]
b_fields = [f.name for f in b_source_fields]


# Add Base_* fields to the output layer.
existing_fields = {f.name for f in arcpy.ListFields(feature_out)}
base_field_map = []
for source_field in b_source_fields:
    out_field = f"Base_{source_field.name}"
    if out_field in existing_fields:
        existing_fields.add(out_field)
    else:
        out_field = add_base_field(feature_out, source_field, existing_fields)
    base_field_map.append((source_field.name, source_field.type, out_field))

if "Match_Dist" not in existing_fields:
    arcpy.management.AddField(feature_out, "Match_Dist", "DOUBLE")
    existing_fields.add("Match_Dist")
if "Match_Angle" not in existing_fields:
    arcpy.management.AddField(feature_out, "Match_Angle", "DOUBLE")
    existing_fields.add("Match_Angle")


# Build base-station dictionary: base ECI -> geometry + source field values.
b_dict = {}
duplicate_keys = set()
with arcpy.da.SearchCursor(feature_b, [base_key_field, "SHAPE@"] + b_fields) as cursor:
    for row in cursor:
        match_key = normalize_match_key(row[0])
        if match_key is None:
            continue
        if match_key in b_dict:
            duplicate_keys.add(match_key)

        geometry = row[1]
        if (
            geometry is not None
            and can_project_between_layers
            and not same_spatial_reference(geometry.spatialReference, a_sr)
        ):
            geometry = geometry.projectAs(a_sr)

        b_dict[match_key] = {
            "geometry": geometry,
            "attributes": dict(zip(b_fields, row[2:])),
        }

if duplicate_keys:
    print(f"Warning: {len(duplicate_keys)} duplicated {base_key_field} values in base data; the last row was used.")

print(f"Matching UE.{ue_key_field} to Base.{base_key_field}. Base keys loaded: {len(b_dict)}.")


use_geographic = bool(a_sr and a_sr.type == "Geographic")
if use_geographic:
    print("Using geodesic distance/azimuth from geographic coordinates.")
else:
    unit_name = a_sr.linearUnitName if a_sr and a_sr.linearUnitName else "map units"
    print(f"Using planar distance/azimuth from projected geometry units: {unit_name}.")


update_fields = [ue_key_field, "SHAPE@", "Match_Dist", "Match_Angle"] + [
    out_field for _, _, out_field in base_field_map
]

matched_count = 0
unmatched_count = 0
invalid_geometry_count = 0

with arcpy.da.UpdateCursor(feature_out, update_fields) as cursor:
    for row in cursor:
        match_key = normalize_match_key(row[0])
        a_geometry = row[1]
        b_record = b_dict.get(match_key)

        if b_record is None:
            unmatched_count += 1
            row[2] = None
            row[3] = None
            for i in range(len(base_field_map)):
                row[4 + i] = None
            cursor.updateRow(row)
            continue

        b_geometry = b_record["geometry"]
        ax, ay = point_xy(a_geometry)
        bx, by = point_xy(b_geometry)

        if None in (ax, ay, bx, by):
            invalid_geometry_count += 1
            row[2] = None
            row[3] = None
        elif use_geographic:
            row[2] = haversine(ax, ay, bx, by)
            row[3] = geodesic_azimuth(ax, ay, bx, by)
        else:
            row[2] = hypot(bx - ax, by - ay)
            row[3] = planar_azimuth(ax, ay, bx, by)

        attributes = b_record["attributes"]
        for i, (source_name, source_type, _) in enumerate(base_field_map):
            try:
                row[4 + i] = to_output_value(attributes.get(source_name), source_type)
            except Exception as exc:
                print(f"Warning: failed to write Base_{source_name}: {exc}")
                row[4 + i] = None

        matched_count += 1
        cursor.updateRow(row)


print(
    "Done. "
    f"Matched: {matched_count}; unmatched: {unmatched_count}; "
    f"invalid geometry: {invalid_geometry_count}."
)
