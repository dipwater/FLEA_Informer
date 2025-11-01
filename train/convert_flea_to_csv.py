import pandas as pd
import os
import re
import glob

# 来自 lowChannel.txt 的标准列名（共20列）
COLUMNS = [
    "date",
    "Desired Position",
    "Actuator X Position",
    "Actuator Y Position",
    "Actuator Z Position",
    "Desired Load",
    "Measured Load",
    "Motor X Current",
    "Motor Y Current",
    "Motor Z Current",
    "Motor X Voltage",
    "Motor Y Voltage",
    "Motor Z Voltage",
    "Motor X Temperature",
    "Motor Y Temperature",
    "Motor Z Temperature",
    "Nut X Temperature",
    "Nut Y Temperature",
    "Nut Z Temperature",
    "Ambient Temperature"
]

def parse_flea_timestamp(ts_str):
    """
    将 '2010-09-03_12:21:49.00000_-0700' 转为 '2010-09-03 12:21:49'
    """
    if pd.isna(ts_str):
        return pd.NaT
    try:
        # 使用正则提取日期和时间部分（忽略微秒和时区）
        match = re.match(r'(\d{4}-\d{2}-\d{2})_(\d{2}:\d{2}:\d{2}.\d{3})', str(ts_str))
        if match:
            date_part, time_part = match.groups()
            return f"{date_part} {time_part}"
        else:
            return pd.NaT
    except:
        return pd.NaT


def convert_flea_to_standard_time(data_path, csv_path, skip_rows=0):
    """
    转换 FLEA_DATA .data 文件：
      - 第一列：原始时间戳 → 'YYYY-MM-DD HH:MM:SS'
      - 其余列：转为数值
    """
    try:
        # 读取原始数据（第一列作为字符串）
        df = pd.read_csv(data_path, sep=r'\s+', header=None, skiprows=skip_rows, dtype={0: str}, engine='python')

        # 对齐列数至 20
        if df.shape[1] < len(COLUMNS):
            for _ in range(len(COLUMNS) - df.shape[1]):
                df[df.shape[1]] = pd.NA
        elif df.shape[1] > len(COLUMNS):
            df = df.iloc[:, :len(COLUMNS)]

        df.columns = COLUMNS

        # 转换时间戳列
        df["date"] = df["date"].apply(parse_flea_timestamp)
        # 可选：转为 datetime 类型（保存为字符串也可）
        # df["Time"] = pd.to_datetime(df["Time"], errors='coerce')

        # 转换其余列为数值
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

        # 保存
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"✅ 转换成功: {data_path} → {csv_path}")

    except Exception as e:
        print(f"❌ 转换失败: {data_path} | 错误: {e}")


def batch_convert_flea(root_dir, output_dir):
    """
    批量转换指定目录下所有 .data 文件
    """
    data_files = glob.glob(os.path.join(root_dir, "**", "*.data"), recursive=True)
    for data_file in data_files:
        rel_path = os.path.relpath(data_file, root_dir)
        csv_file = os.path.join(output_dir, rel_path.replace(".data", ".csv"))
        convert_flea_to_standard_time(data_file, csv_file)


# ======================
# 单文件转换示例（按你需求）
# ======================

if __name__ == "__main__":
    # 输出根目录
    OUTPUT_ROOT = "../data/FLEA2/"
    batch_convert_flea("../FLEA_DATA", OUTPUT_ROOT)
    # # 定义四类状态的原始 .data 文件路径（来自你的文档）
    # file_configs = [
    #     {
    #         "name": "Normal",
    #         "path": "FLEA_DATA/2010_09_03/sdata/BatchProfile_triang14_2010_09_03_12_21_49_Nominal_Low.data",
    #         "skip": 0
    #     },
    #     {
    #         "name": "Position",
    #         "path": "FLEA_DATA/2010_09_10_position_dead/sdata/BatchProfile_sine24_2010_09_10_15_08_45_Position_t+24.38_o+0.00_s+0.00_Low.data",
    #         "skip": 0
    #     },
    #     {
    #         "name": "Jam",
    #         "path": "FLEA_DATA/2010_09_03/sdata/BatchProfile_sine15-2m_2010_09_03_15_57_04_Jam_Low.data",
    #         "skip": 9118  # 从第9119行开始（skip前9118行）
    #     },
    #     {
    #         "name": "Spall",
    #         "path": "FLEA_DATA/2010_09_03/sdata/BatchProfile_sweep12_2010_09_03_16_55_42_Spall_Low.data",
    #         "skip": 0
    #     }
    # ]
    #
    # # 逐个转换
    # for cfg in file_configs:
    #     csv_output = os.path.join(OUTPUT_ROOT, f"{cfg['name']}.csv")
    #     convert_flea_to_standard_time(cfg["path"], csv_output, skip_rows=cfg["skip"])

    print("\n🎉 所有文件转换完成！")
