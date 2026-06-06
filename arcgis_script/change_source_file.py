import os

import arcpy


# ArcGIS Pro 工程路径；目标要素类位于工程默认地理数据库 Default.gdb 中。
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

# 直接更新该要素类的 source_file 字段。
feature_class = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d")
source_field = "source_file"


def require_exists(path, label):
    """检查 ArcGIS 数据或普通文件是否存在。"""
    if label == "ArcGIS Pro project":
        exists = os.path.exists(path)
    else:
        exists = arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label} does not exist: {path}")


def require_field(feature_class_path, field_name):
    """检查目标字段是否存在，避免游标打开后才报错。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class_path)}
    if field_name not in existing_fields:
        raise RuntimeError(f"Missing field '{field_name}' in {feature_class_path}")


arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro project")
require_exists(gdb_path, "Geodatabase")
require_exists(feature_class, "Feature class")
require_field(feature_class, source_field)

ground_count = 0
uav_count = 0

# 写游标会获取要素类写锁；运行前需关闭 ArcGIS Pro 中打开的属性表或图层编辑状态。
with arcpy.da.UpdateCursor(feature_class, [source_field]) as cursor:
    for row in cursor:
        # 空值按“非 ground”处理；匹配 ground 时忽略大小写。
        source_value = "" if row[0] is None else str(row[0])

        if "ground" in source_value.lower():
            row[0] = "ground"
            ground_count += 1
        else:
            row[0] = "UAV"
            uav_count += 1

        cursor.updateRow(row)

total_count = int(arcpy.management.GetCount(feature_class)[0])

print(f"Feature class: {feature_class}")
print(f"Total rows: {total_count}")
print(f"Updated to ground: {ground_count}")
print(f"Updated to UAV: {uav_count}")
