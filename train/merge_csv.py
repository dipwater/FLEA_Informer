import os
import pandas as pd
import glob
from typing import List

# 定义要保留的列（必须包含 target）
selected_columns = [
    "date",
    "Actuator Z Position",
    "Motor Z Current",
    "Motor Y Temperature",
    "Motor Z Temperature",
    "Nut Y Temperature",
    "Ambient Temperature",
    "Motor Y Voltage"  # target
]

def merge_csv_by_keywords(root_dir: str, keywords: List[str], output_filename: str):
    """
    合并指定目录下文件名包含任一关键字的 CSV 文件，输出为单个文件。

    参数:
        root_dir (str): CSV 文件所在目录路径（如 '../data/flea/'）
        keywords (List[str]): 要匹配的关键字列表（如 ['sine11', 'sine12', 'sine13', 'sine15']）
        output_filename (str): 输出的合并文件名（默认 'Normal.csv'）

    输出:
        保存合并后的 CSV 文件到 root_dir/output_filename
    """
    # 查找所有匹配的文件
    matched_files = set()
    for kw in keywords:
        pattern = os.path.join(root_dir, f"*{kw}*.csv")
        matched_files.update(glob.glob(pattern))

    matched_files = sorted(matched_files)  # 排序保证可重复性

    if not matched_files:
        raise FileNotFoundError(f"在 '{root_dir}' 中未找到包含以下任一关键字的 CSV 文件: {keywords}")

    print(f"🔍 找到 {len(matched_files)} 个匹配文件:")
    for f in matched_files:
        print(f"  - {os.path.basename(f)}")

    # 读取所有文件
    df_list = []
    for file in matched_files:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            print(f"⚠️ 跳过无效文件 {file}: {e}")
            continue
        if 'date' not in df.columns:
            raise ValueError(f"文件 {file} 缺少 'date' 列！")
        df_list.append(df)

    if not df_list:
        raise ValueError("没有有效数据可合并！")

    # 合并并排序
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values('date').reset_index(drop=True)

    # 筛选列（确保顺序一致）
    # 选择 Actuator Z Position、Motor Z Current、Motor Y Temperature、Motor Z Temperature、Nut Y Temperature、Ambient Temperature 以及 Motor Y Voltage 参数共 7 维特征作为输入
    combined_df = combined_df[selected_columns]

    # 保存
    combined_df.to_csv(output_filename, index=False)
    print(f"\n✅ 合并成功！已保存至: {output_filename}")
    print(f"📊 总行数: {len(combined_df)} | 列: {list(combined_df.columns)}")


def merge_csv_files(
        folder_path: str,
        output_file: str = 'merged_output.csv',
        file_pattern: str = '*.csv',
        include_source: bool = False,
        encoding: str = 'utf-8',
        ignore_index: bool = True
) -> pd.DataFrame:
    """
    合并指定文件夹中的所有 CSV 文件为一个 DataFrame，并保存为新的 CSV 文件。

    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径。
        output_file (str): 输出的合并后 CSV 文件名（含路径可选），默认为 'merged_output.csv'。
        file_pattern (str): 文件匹配模式（目前仅支持 '.csv'，保留扩展性），默认 '*.csv'。
        include_source (bool): 是否添加一列 'source_file' 记录每行数据来自哪个文件，默认 False。
        encoding (str): 读取和写入 CSV 文件时使用的编码格式，默认 'utf-8'。
        ignore_index (bool): 合并时是否重置索引，默认 True。

    返回:
        pd.DataFrame: 合并后的 DataFrame。

    示例:
        df = merge_csv_files('data/', 'result.csv', include_source=True)
    """
    # 获取所有 .csv 文件（忽略大小写）
    csv_files: List[str] = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith('.csv')
    ]

    if not csv_files:
        raise ValueError(f"在路径 '{folder_path}' 中未找到任何 CSV 文件。")

    dataframes: List[pd.DataFrame] = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            if include_source:
                df['source_file'] = file
            dataframes.append(df)
        except Exception as e:
            print(f"⚠️ 读取文件 {file_path} 时出错，已跳过：{e}")

    if not dataframes:
        raise ValueError("没有成功读取任何 CSV 文件。")

    # 合并所有 DataFrame
    merged_df = pd.concat(dataframes, ignore_index=ignore_index)

    # 保存到文件
    merged_df.to_csv(output_file, index=False, encoding=encoding)
    print(f"✅ 成功合并 {len(dataframes)} 个 CSV 文件，结果已保存至: {output_file}")

    return merged_df

if __name__ == "__main__":
    # full
    df = merge_csv_files(
        folder_path='../data/FLEA/',
        output_file='../data/FLEA/full.csv',
        include_source=True,
        encoding='utf-8'  # 如果是中文 Windows 系统常见编码
    )

    # # Normal.csv
    # merge_csv_by_keywords(
    #     root_dir="../data/FLEA2/2010_09_03/sdata",
    #     keywords=["sine11", "sine12", "sine13", "sine15"],
    #     output_filename="../data/FLEA/Normal.csv"
    # )
    #
    # # Jam.csv
    # merge_csv_by_keywords(
    #     root_dir="../data/FLEA2/2010_09_03/sdata",
    #     keywords=["sine13", "sine14", "sine15"],
    #     output_filename="../data/FLEA/Jam.csv"
    # )
    #
    # # Position.csv
    # merge_csv_by_keywords(
    #     root_dir="../data/FLEA2/2010_09_10_position_dead/sdata",
    #     keywords=["trap13", "trap14", "trap24", "trap25"],
    #     output_filename="../data/FLEA/Position.csv"
    # )
    #
    # # Spall.csv
    # merge_csv_by_keywords(
    #     root_dir="../data/FLEA2/2010_09_03/sdata",
    #     keywords=["sine14", "sine24", "sine25", "sine33"],
    #     output_filename="../data/FLEA/Spall.csv"
    # )