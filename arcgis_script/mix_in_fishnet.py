import os
from collections import Counter, defaultdict

import arcpy


# ========= 路径配置 =========
aprx_path = r"E:\arcgis_workspace\main_file\simulation_5G_clean\simulation_5G_clean\simulation_5G_clean.aprx"
base_path = os.path.dirname(aprx_path)
gdb_path = os.path.join(base_path, "Default.gdb")

fishnet = os.path.join(gdb_path, "fishnet_30x30")
ue_points = os.path.join(gdb_path, "UE_Points_proj_48N_with_base_matched_3d")
output_fishnet = os.path.join(gdb_path, "fishnet_30x30_with_ue_3d")
spatial_join_fc = os.path.join(gdb_path, "_tmp_spatialjoin_fishnet_ue_3d")


# ========= 字段配置 =========
# 键为输出到 fishnet 的字段名，值为 UE 图层中的来源字段名。
field_map = {
    "TIME": "TIME",
    "SPEED_M_s_": "SPEED_M_s_",
    "ALT_M_": "ALT_M_",
    "NETWORK_TYPE": "NETWORK_TY",
    "NR_TAC": "NR_TAC",
    "NR_BAND": "NR_BAND",
    "NR_PCI": "NR_PCI",
    "SS_RSRP": "SS_RSRP",
    "SS_RSRQ": "SS_RSRQ",
    "SS_SINR": "SS_SINR",
    "Base_CGI": "Base_CGI",
    "Base_LONGITUDE": "Base_LONGITUDE",
    "Base_LATITUDE": "Base_LATITUDE",
    "Base_Direction_angle": "Base_Direction_angle",
    "Base_Central_frequency_point": "Base_Central_frequency_point",
    "Base_Bandwidth": "Base_Bandwidth",
    "Base_Electronic_downtilt": "Base_Electronic_downtilt",
    "Base_Mechanical_downtilt": "Base_Mechanical_downtilt",
    "Base_Power": "Base_Power",
    "Match_Dist": "Match_Dist",
    "Match_Angle": "Match_Angle",
    "True_3D_Dist": "True_3D_Dist",
}

# 这些字段使用众数；其余数值字段使用按 Base_CGI 频数加权平均。
mode_fields = {
    "TIME",
    "NETWORK_TYPE",
    "NR_TAC",
    "NR_BAND",
    "NR_PCI",
    "Base_CGI",
    "Base_LONGITUDE",
    "Base_LATITUDE",
    "Base_Direction_angle",
    "Base_Central_frequency_point",
    "Base_Bandwidth",
    "Base_Electronic_downtilt",
    "Base_Mechanical_downtilt",
    "Base_Power",
}


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


def normalize_key(value):
    """统一 Base_CGI 这类以浮点保存的整数键，避免频数统计不稳定。"""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def most_common(values):
    """返回非空值中的众数。"""
    valid_values = [value for value in values if value not in (None, "", " ")]
    return Counter(valid_values).most_common(1)[0][0] if valid_values else None


def weighted_avg_by_cgi_freq(values, cgis, cgi_counter):
    """按网格内 Base_CGI 出现频数计算加权平均。"""
    valid_values = []
    for value, cgi in zip(values, cgis):
        if isinstance(value, (int, float)):
            weight = cgi_counter.get(normalize_key(cgi), 1)
            if weight > 0:
                valid_values.append((float(value), weight))

    if not valid_values:
        return None

    return sum(value * weight for value, weight in valid_values) / sum(
        weight for _, weight in valid_values
    )


def add_output_field_if_missing(feature_class, output_field, source_field):
    """按 UE 来源字段类型，在输出 fishnet 中补齐字段。"""
    existing_fields = {field.name for field in arcpy.ListFields(feature_class)}
    if output_field in existing_fields:
        return

    source_fields = {field.name: field for field in arcpy.ListFields(ue_points)}
    source_type = source_fields[source_field].type

    if source_type in ("Double", "Single", "Integer", "SmallInteger"):
        arcpy.management.AddField(feature_class, output_field, "DOUBLE")
    elif source_type == "String":
        arcpy.management.AddField(feature_class, output_field, "TEXT", field_length=100)
    else:
        arcpy.management.AddField(feature_class, output_field, "TEXT", field_length=100)


def make_join_output_name(source_field):
    """为空间连接临时字段增加前缀，避免和 fishnet 原有字段重名。"""
    return f"SJ_{source_field}"[:64]


def create_join_field_mappings(join_features, join_field_names):
    """创建空间连接字段映射，只写入 UE 侧需要聚合的字段。"""
    field_mappings = arcpy.FieldMappings()

    for source_field in join_field_names:
        field_map_item = arcpy.FieldMap()
        field_map_item.addInputField(join_features, source_field)

        output_field = field_map_item.outputField
        output_field.name = make_join_output_name(source_field)
        output_field.aliasName = make_join_output_name(source_field)
        field_map_item.outputField = output_field

        field_mappings.addFieldMap(field_map_item)

    return field_mappings


# ========= 运行前检查 =========
arcpy.env.workspace = gdb_path
arcpy.env.overwriteOutput = True

require_exists(aprx_path, "ArcGIS Pro 项目")
require_exists(gdb_path, "地理数据库")
require_exists(fishnet, "输入 fishnet 图层")
require_exists(ue_points, "输入 UE 图层")
require_fields(ue_points, set(field_map.values()))


# ========= 复制 fishnet，后续只更新复制后的新图层 =========
for feature_class in (output_fishnet, spatial_join_fc):
    if arcpy.Exists(feature_class):
        try:
            arcpy.management.Delete(feature_class)
        except arcpy.ExecuteError as exc:
            raise RuntimeError(
                f"无法删除已有图层：{feature_class}。"
                "请关闭 ArcGIS Pro 中正在使用该图层的地图或属性表后重试。"
            ) from exc

arcpy.management.CopyFeatures(fishnet, output_fishnet)

for output_field, source_field in field_map.items():
    add_output_field_if_missing(output_fishnet, output_field, source_field)


# ========= 空间连接：每个 fishnet 网格对应多个 UE 点 =========
source_fields = list(dict.fromkeys(field_map.values()))
field_mappings = create_join_field_mappings(ue_points, source_fields)

arcpy.analysis.SpatialJoin(
    target_features=output_fishnet,
    join_features=ue_points,
    out_feature_class=spatial_join_fc,
    join_operation="JOIN_ONE_TO_MANY",
    join_type="KEEP_COMMON",
    field_mapping=field_mappings,
    match_option="INTERSECT",
)


# ========= 汇总每个网格内的 UE 字段值 =========
stats = defaultdict(lambda: defaultdict(list))
cgi_counts_per_grid = defaultdict(Counter)

join_source_fields = {source_field: make_join_output_name(source_field) for source_field in source_fields}
join_fields = ["TARGET_FID"] + list(join_source_fields.values())
source_index = {field_name: index for index, field_name in enumerate(join_fields)}

with arcpy.da.SearchCursor(spatial_join_fc, join_fields) as cursor:
    for row in cursor:
        target_oid = row[0]
        if target_oid is None:
            continue

        for output_field, source_field in field_map.items():
            stats[target_oid][output_field].append(row[source_index[join_source_fields[source_field]]])

        cgi = row[source_index[join_source_fields["Base_CGI"]]]
        if cgi is not None:
            cgi_counts_per_grid[target_oid][normalize_key(cgi)] += 1


# ========= 将聚合结果写入新 fishnet =========
update_fields = ["OID@"] + list(field_map.keys())
updated_grid_count = 0

with arcpy.da.UpdateCursor(output_fishnet, update_fields) as cursor:
    for row in cursor:
        oid = row[0]
        if oid not in stats:
            continue

        cgi_counter = cgi_counts_per_grid[oid]
        cgi_values = stats[oid]["Base_CGI"]

        for index, output_field in enumerate(field_map.keys(), start=1):
            values = stats[oid][output_field]
            if output_field in mode_fields:
                row[index] = most_common(values)
            else:
                row[index] = weighted_avg_by_cgi_freq(values, cgi_values, cgi_counter)

        cursor.updateRow(row)
        updated_grid_count += 1


fishnet_count = int(arcpy.management.GetCount(output_fishnet)[0])
join_count = int(arcpy.management.GetCount(spatial_join_fc)[0])

arcpy.management.Delete(spatial_join_fc)

print(f"输出 fishnet 网格数量：{fishnet_count}")
print(f"参与空间连接的 UE 记录数量：{join_count}")
print(f"成功赋值的 fishnet 网格数量：{updated_grid_count}")
print(f"输出图层：{output_fishnet}")
