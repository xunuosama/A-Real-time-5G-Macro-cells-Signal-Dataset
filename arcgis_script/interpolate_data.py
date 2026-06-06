import math
import os
from collections import Counter

import arcpy
import numpy as np
from scipy.ndimage import uniform_filter


# ========= 路径配置 =========
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

input_fishnet = os.path.join(gdb_path, "fishnet_30x30_with_ue_3d")
output_fishnet = os.path.join(gdb_path, "fishnet_30x30_with_ue_3d_interpolated")


# ========= 网格参数 =========
rows = 100
cols = 100
max_iter = 20


# ========= 需要填充的字段 =========
target_fields = [
    "SPEED_M_s_",
    "ALT_M_",
    "NETWORK_TYPE",
    "NR_TAC",
    "NR_BAND",
    "NR_PCI",
    "SS_RSRP",
    "SS_RSRQ",
    "SS_SINR",
    "Base_CGI",
    "Base_LONGITUDE",
    "Base_LATITUDE",
    "Base_Direction_angle",
    "Base_Central_frequency_point",
    "Base_Bandwidth",
    "Base_Electronic_downtilt",
    "Base_Mechanical_downtilt",
    "Base_Power",
    "Match_Dist",
    "Match_Angle",
    "True_3D_Dist",
]

# 字符串类别字段使用众数填充，其余字段使用均值填充。
categorical_fields = {"NETWORK_TYPE", "NR_BAND", "Base_Bandwidth"}

building_field = "Building_Coverage"
building_5x5_field = "Building_Coverage_5x5"


def require_exists(path, label):
    """检查 ArcGIS 数据或普通文件路径是否存在。"""
    exists = os.path.exists(path) if label == "ArcGIS Pro 项目" else arcpy.Exists(path)
    if not exists:
        raise RuntimeError(f"{label}不存在：{path}")


def require_fields(feature_class, field_names):
    """检查图层是否包含所需字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    missing_fields = [field_name for field_name in field_names if field_name not in existing_fields]
    if missing_fields:
        raise RuntimeError(f"图层缺少字段 {missing_fields}：{feature_class}")


def add_field_if_missing(feature_class, field_name, field_type):
    """如果输出字段不存在，则新增字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if field_name not in existing_fields:
        arcpy.management.AddField(feature_class, field_name, field_type)


def oid_to_rc(oid):
    """将 fishnet 的 OBJECTID 映射为 100x100 矩阵中的行列号。"""
    index = oid - 1
    row = rows - 1 - index // cols
    col = index % cols
    return row, col


def is_empty(value):
    """判断字段值是否为空。"""
    return value is None or value == "" or value == " "


def create_numeric_matrix(records, field_name):
    """根据字段值创建数值矩阵。"""
    matrix = np.full((rows, cols), np.nan, dtype=float)
    for oid, values in records.items():
        row, col = oid_to_rc(oid)
        value = values.get(field_name)
        if not is_empty(value):
            matrix[row, col] = float(value)
    return matrix


def create_category_matrix(records, field_name):
    """根据字段值创建类别矩阵。"""
    matrix = np.full((rows, cols), None, dtype=object)
    for oid, values in records.items():
        row, col = oid_to_rc(oid)
        value = values.get(field_name)
        if not is_empty(value):
            matrix[row, col] = value
    return matrix


def fill_once_mean(matrix):
    """使用 3x3 邻域均值执行一轮填充。"""
    new_matrix = matrix.copy()
    filled_count = 0

    for row in range(rows):
        for col in range(cols):
            if math.isnan(matrix[row, col]):
                neighbors = matrix[
                    max(0, row - 1) : min(rows, row + 2),
                    max(0, col - 1) : min(cols, col + 2),
                ]
                values = neighbors[~np.isnan(neighbors)]
                if len(values) > 0:
                    new_matrix[row, col] = float(values.mean())
                    filled_count += 1

    return new_matrix, filled_count


def fill_once_mode(matrix):
    """使用 3x3 邻域众数执行一轮填充。"""
    new_matrix = matrix.copy()
    filled_count = 0

    for row in range(rows):
        for col in range(cols):
            if is_empty(matrix[row, col]):
                neighbors = matrix[
                    max(0, row - 1) : min(rows, row + 2),
                    max(0, col - 1) : min(cols, col + 2),
                ].ravel()
                values = [value for value in neighbors if not is_empty(value)]
                if values:
                    new_matrix[row, col] = Counter(values).most_common(1)[0][0]
                    filled_count += 1

    return new_matrix, filled_count


def iterative_fill(matrix, field_name, method):
    """多轮执行邻域填充，直到没有新填充值或达到最大轮数。"""
    for index in range(max_iter):
        if method == "mode":
            matrix, filled_count = fill_once_mode(matrix)
        else:
            matrix, filled_count = fill_once_mean(matrix)

        print(f"{field_name} 第 {index + 1} 轮填充：{filled_count} 个值")
        if filled_count == 0:
            break

    return matrix


def write_matrix_to_records(records, field_name, matrix):
    """将矩阵填充结果写回记录字典。"""
    for oid in records:
        row, col = oid_to_rc(oid)
        value = matrix[row, col]
        records[oid][field_name] = None if is_empty(value) else value


def create_building_coverage_5x5(records):
    """生成 Building_Coverage 的 5x5 均值特征。"""
    matrix = create_numeric_matrix(records, building_field)
    matrix = np.nan_to_num(matrix, nan=0.0)
    coverage_5x5 = uniform_filter(matrix, size=5, mode="constant")

    for oid in records:
        row, col = oid_to_rc(oid)
        records[oid][building_5x5_field] = float(coverage_5x5[row, col])


# ========= 运行前检查 =========
arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro 项目")
require_exists(gdb_path, "地理数据库")
require_exists(input_fishnet, "输入 fishnet 图层")
require_fields(input_fishnet, target_fields + [building_field])


# ========= 复制 fishnet，避免直接修改原图层 =========
if arcpy.Exists(output_fishnet):
    try:
        arcpy.management.Delete(output_fishnet)
    except arcpy.ExecuteError as exc:
        raise RuntimeError(
            f"无法删除已有输出图层：{output_fishnet}。"
            "请关闭 ArcGIS Pro 中正在使用该图层的地图或属性表后重试。"
        ) from exc

arcpy.management.CopyFeatures(input_fishnet, output_fishnet)
add_field_if_missing(output_fishnet, building_5x5_field, "DOUBLE")


# ========= 读取 fishnet 数据到内存 =========
read_fields = ["OID@"] + target_fields + [building_field]
records = {}

with arcpy.da.SearchCursor(output_fishnet, read_fields) as cursor:
    for row_values in cursor:
        oid = row_values[0]
        records[oid] = dict(zip(read_fields[1:], row_values[1:]))

if len(records) != rows * cols:
    raise RuntimeError(f"fishnet 网格数量不是 {rows * cols}：当前为 {len(records)}")


# ========= 执行所有目标字段的邻域填充 =========
for field_name in target_fields:
    print(f"\n开始处理字段：{field_name}")
    if field_name in categorical_fields:
        field_matrix = create_category_matrix(records, field_name)
        filled_matrix = iterative_fill(field_matrix, field_name, method="mode")
    else:
        field_matrix = create_numeric_matrix(records, field_name)
        filled_matrix = iterative_fill(field_matrix, field_name, method="mean")

    write_matrix_to_records(records, field_name, filled_matrix)


# ========= 生成建筑覆盖率 5x5 均值特征 =========
create_building_coverage_5x5(records)


# ========= 将结果写回输出 fishnet =========
update_fields = ["OID@"] + target_fields + [building_5x5_field]
updated_count = 0

with arcpy.da.UpdateCursor(output_fishnet, update_fields) as cursor:
    for row_values in cursor:
        oid = row_values[0]
        values = records[oid]

        for index, field_name in enumerate(update_fields[1:], start=1):
            row_values[index] = values.get(field_name)

        cursor.updateRow(row_values)
        updated_count += 1


print(f"\n输出 fishnet 数量：{updated_count}")
print(f"输出图层：{output_fishnet}")
