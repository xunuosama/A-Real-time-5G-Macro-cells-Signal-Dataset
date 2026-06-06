import math
import os

import arcpy


# ========= 路径配置 =========
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

input_ue = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched")
output_ue = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d")


# ========= 字段配置 =========
ue_alt_field = "ALT_M_"
horizontal_dist_field = "Match_Dist"
base_height_field = "Base_height"
base_dem_field = "Base_Base_DEM"
true_dist_field = "True_3D_Dist"

required_fields = [
    ue_alt_field,
    horizontal_dist_field,
    base_height_field,
    base_dem_field,
]


def require_exists(path, label):
    """检查 ArcGIS 数据或普通文件路径是否存在。"""
    if label == "ArcGIS Pro 项目":
        exists = os.path.exists(path)
    else:
        exists = arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label}不存在：{path}")


def require_fields(feature_class, field_names):
    """检查输入图层是否包含计算所需字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    missing_fields = [field for field in field_names if field not in existing_fields]
    if missing_fields:
        raise RuntimeError(f"图层缺少字段 {missing_fields}：{feature_class}")


def add_field_if_missing(feature_class, field_name, field_type):
    """如果字段不存在，则新增字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if field_name not in existing_fields:
        arcpy.management.AddField(feature_class, field_name, field_type)


# ========= 运行前检查 =========
arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro 项目")
require_exists(gdb_path, "地理数据库")
require_exists(input_ue, "输入 UE 图层")
require_fields(input_ue, required_fields)


# ========= 复制图层，避免直接修改原始 matched 图层 =========
if arcpy.Exists(output_ue):
    try:
        arcpy.management.Delete(output_ue)
    except arcpy.ExecuteError as exc:
        raise RuntimeError(
            f"无法删除已有输出图层：{output_ue}。"
            "请关闭 ArcGIS Pro 中正在使用该图层的地图或属性表后重试。"
        ) from exc

arcpy.management.CopyFeatures(input_ue, output_ue)
add_field_if_missing(output_ue, true_dist_field, "DOUBLE")


# ========= 计算 UE 到基站的真实三维距离 =========
update_fields = [
    ue_alt_field,
    horizontal_dist_field,
    base_height_field,
    base_dem_field,
    true_dist_field,
]

updated_count = 0
skipped_count = 0

with arcpy.da.UpdateCursor(output_ue, update_fields) as cursor:
    for row in cursor:
        ue_alt, horizontal_dist, base_height, base_dem, _ = row

        if None in (ue_alt, horizontal_dist, base_height, base_dem):
            skipped_count += 1
            row[4] = None
            cursor.updateRow(row)
            continue

        vertical_diff = float(ue_alt) - (float(base_dem) + float(base_height))
        true_dist = math.sqrt(float(horizontal_dist) ** 2 + vertical_diff ** 2)

        row[4] = round(true_dist, 2)
        cursor.updateRow(row)
        updated_count += 1


input_count = int(arcpy.management.GetCount(input_ue)[0])
output_count = int(arcpy.management.GetCount(output_ue)[0])

print(f"输入 UE 点数量：{input_count}")
print(f"输出 UE 点数量：{output_count}")
print(f"成功计算 True_3D_Dist 数量：{updated_count}")
print(f"因字段为空跳过数量：{skipped_count}")
print(f"输出图层：{output_ue}")
