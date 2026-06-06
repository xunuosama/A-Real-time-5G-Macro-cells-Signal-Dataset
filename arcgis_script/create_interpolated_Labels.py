import os

import arcpy


# ========= 路径配置 =========
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

input_fishnet = os.path.join(gdb_path, "fishnet_30x30_with_ue_3d_interpolated")
ue_points = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d")
output_fishnet = os.path.join(gdb_path, "fishnet_30x30_with_ue_3d_interpolated_labeled")
tmp_join = os.path.join(gdb_path, "_tmp_interpolated_label_counts")


# ========= 标签字段 =========
observed_field = "is_observed"
interpolated_field = "is_interpolated"
raw_count_field = "n_raw_measurements"
method_field = "interpolation_method"

observed_method = "none_observed"
interpolated_method = "iterative_3x3_mean_mode"


def require_exists(path, label):
    """检查 ArcGIS 数据或普通文件路径是否存在。"""
    exists = os.path.exists(path) if label == "ArcGIS Pro 项目" else arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label}不存在：{path}")


def add_field_if_missing(feature_class, field_name, field_type, field_length=None):
    """如果字段不存在，则新增字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if field_name in existing_fields:
        return

    if field_length is None:
        arcpy.management.AddField(feature_class, field_name, field_type)
    else:
        arcpy.management.AddField(
            feature_class,
            field_name,
            field_type,
            field_length=field_length,
        )


def delete_if_exists(feature_class):
    """删除已存在的临时或输出图层。"""
    if arcpy.Exists(feature_class):
        try:
            arcpy.management.Delete(feature_class)
        except arcpy.ExecuteError as exc:
            raise RuntimeError(
                f"无法删除已有图层：{feature_class}。"
                "请关闭 ArcGIS Pro 中正在使用该图层的地图或属性表后重试。"
            ) from exc


# ========= 运行前检查 =========
arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro 项目")
require_exists(gdb_path, "地理数据库")
require_exists(input_fishnet, "输入插值 fishnet 图层")
require_exists(ue_points, "输入 UE 点图层")


# ========= 复制插值后 fishnet，避免直接修改原图层 =========
delete_if_exists(output_fishnet)
delete_if_exists(tmp_join)

arcpy.management.CopyFeatures(input_fishnet, output_fishnet)

add_field_if_missing(output_fishnet, observed_field, "SHORT")
add_field_if_missing(output_fishnet, interpolated_field, "SHORT")
add_field_if_missing(output_fishnet, raw_count_field, "LONG")
add_field_if_missing(output_fishnet, method_field, "TEXT", field_length=50)


# ========= 重新统计每个网格内的原始 UE 点数量 =========
# Join_Count 是每个 fishnet 网格内相交的 UE 点数，用它区分 observed 和 interpolated。
arcpy.analysis.SpatialJoin(
    target_features=output_fishnet,
    join_features=ue_points,
    out_feature_class=tmp_join,
    join_operation="JOIN_ONE_TO_ONE",
    join_type="KEEP_ALL",
    match_option="INTERSECT",
)

raw_count_by_oid = {}
with arcpy.da.SearchCursor(tmp_join, ["TARGET_FID", "Join_Count"]) as cursor:
    for target_oid, join_count in cursor:
        raw_count_by_oid[target_oid] = int(join_count or 0)


# ========= 写入标签字段 =========
observed_grid_count = 0
interpolated_grid_count = 0
raw_measurement_count = 0

update_fields = [
    "OID@",
    observed_field,
    interpolated_field,
    raw_count_field,
    method_field,
]

with arcpy.da.UpdateCursor(output_fishnet, update_fields) as cursor:
    for row in cursor:
        oid = row[0]
        raw_count = raw_count_by_oid.get(oid, 0)
        is_observed = 1 if raw_count > 0 else 0
        is_interpolated = 0 if raw_count > 0 else 1

        row[1] = is_observed
        row[2] = is_interpolated
        row[3] = raw_count
        row[4] = observed_method if is_observed else interpolated_method
        cursor.updateRow(row)

        raw_measurement_count += raw_count
        if is_observed:
            observed_grid_count += 1
        else:
            interpolated_grid_count += 1


delete_if_exists(tmp_join)

print(f"输出图层：{output_fishnet}")
print(f"observed 网格数量：{observed_grid_count}")
print(f"interpolated 网格数量：{interpolated_grid_count}")
print(f"原始 UE 测量点总数：{raw_measurement_count}")
