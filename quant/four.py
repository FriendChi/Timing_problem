import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import akshare as ak
import baostock as bs
import pandas as pd
import logging
from typing import List, Optional
from stragedy import *
import baostock as bs
import pandas as pd
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        action='store',
        type=str,
        default='test',
    )

    return parser

import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks, savgol_filter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def plot_ma20_with_extrema(
        df: pd.DataFrame,
        price_col: str = "nav",
        date_col: str = "date",
        out_path: str = "ma20_extrema.png",
        *,
        ma_window: int = 20,
        extrema_distance: int = 5,
        prominence: float | None = None,
        smooth: bool = True,
        sg_window: int = 7,
        sg_poly: int = 2,
) -> List[pd.Timestamp]:  # ⚡️ 函数返回类型
    """
    计算并绘制 MA20，同时用更稳健的方法标记局部高/低点。

    Parameters
    ----------
    df : pd.DataFrame
        数据，至少包含 date_col 与 price_col。
    price_col : str
        价格列名。
    date_col : str
        日期列名。
    out_path : str
        图片保存路径。
    ma_window : int, default 20
        均线窗口长度。
    extrema_distance : int, default 5
        极值之间最小间隔（交易日数）。
    prominence : float | None
        峰值显著性（scipy 的 prominence）；None = 自动。
    smooth : bool, default True
        是否使用 Savitzky-Golay 滤波先平滑 MA20。
    sg_window : int, default 7
        SG 滤波窗口（必须为奇数）。
    sg_poly : int, default 2
        SG 滤波多项式阶数。

    Returns
    -------
    List[pd.Timestamp]
        极小值对应的日期列表（以索引时间戳形式返回）。
    """
    # 1. 时间索引
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).set_index(date_col)

    # 2. MA20
    df["MA20"] = df[price_col].rolling(ma_window).mean()
    ma = df["MA20"].dropna()

    # 3. 可选平滑
    if smooth and len(ma) >= sg_window:
        if _HAS_SCIPY:
            ma_smooth = pd.Series(
                savgol_filter(ma.values, window_length=sg_window, polyorder=sg_poly),
                index=ma.index,
            )
        else:
            ma_smooth = ma.rolling(sg_window, center=True).mean()
    else:
        ma_smooth = ma

    # 4. 找极大 / 极小
    if _HAS_SCIPY:
        peaks, _ = find_peaks(
            ma_smooth.values,
            distance=extrema_distance,
            prominence=prominence,
        )
        troughs, _ = find_peaks(
            -ma_smooth.values,
            distance=extrema_distance,
            prominence=prominence,
        )
        local_max_idx = ma_smooth.index[peaks]
        local_min_idx = ma_smooth.index[troughs]
    else:
        win = 2 * extrema_distance + 1
        rolling_max = ma_smooth.rolling(win, center=True).max()
        rolling_min = ma_smooth.rolling(win, center=True).min()
        local_max_idx = ma_smooth[(ma_smooth == rolling_max) & ma_smooth.notna()].index
        local_min_idx = ma_smooth[(ma_smooth == rolling_min) & ma_smooth.notna()].index

    # 5. 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_col], label=price_col, linewidth=1)
    plt.plot(df.index, df["MA20"], label=f"MA{ma_window}", linewidth=1.5)
    plt.scatter(local_max_idx, df.loc[local_max_idx, "MA20"],
                marker="^", s=5, label="Local Max", zorder=5, color="red")
    plt.scatter(local_min_idx, df.loc[local_min_idx, "MA20"],
                marker="v", s=5, label="Local Min", zorder=5, color="green")

    plt.title("Price & MA20 with Extrema")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    minima_dates = list(local_min_idx)  # ⚡️ 提取极小值日期列表

    print(
        f"图已保存至 {out_path}；识别到高点 {len(local_max_idx)} 个，低点 {len(local_min_idx)} 个。"
    )

    return minima_dates,df  # ⚡️ 返回

import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

def plot_nav_ema_crosses(
    df: pd.DataFrame,
    nav_col: str = "nav",
    ema_list: tuple[int, ...] = (10, 12, 13, 18, 20, 26),
    figsize=(14, 7),
    save_path: str | None = "ema_crosses_30.png"
) -> dict[str, list[pd.Timestamp]]:
    """
    绘制 NAV，并用 30 种颜色标记所有 up / down EMA 交叉。
    返回 {类别名: 日期列表} 字典，键共 30 个。
    """
    # ---- 1. 生成颜色 ----
    cmap1 = cm.get_cmap("tab20").colors          # 20 种
    cmap2 = cm.get_cmap("tab20b").colors         # 20 种
    all_colors = list(cmap1) + list(cmap2)       # 40 种，足够
    color_iter = iter(all_colors)

    # ---- 2. 计算交叉 ----
    cross_dict: dict[str, list[pd.Timestamp]] = {}

    for fast, slow in itertools.combinations(sorted(ema_list), 2):
        diff = df[f"EMA{fast}"] - df[f"EMA{slow}"]
        prev = diff.shift(1)

        up_mask   = (prev < 0) & (diff > 0)      # 上穿
        down_mask = (prev > 0) & (diff < 0)      # 下穿

        up_dates   = df.index[up_mask].tolist()
        down_dates = df.index[down_mask].tolist()

        cross_dict[f"EMA{fast}_over_EMA{slow}_up"]   = up_dates
        cross_dict[f"EMA{fast}_over_EMA{slow}_down"] = down_dates

    # ---- 3. 绘图 ----
    fig, ax = plt.subplots(figsize=figsize)

    # 3.1 画 NAV
    df[nav_col].plot(ax=ax, lw=1.2, label=nav_col)

    # 3.2 为每个类别画竖线
    for cls_name, dates in cross_dict.items():
        color = next(color_iter)
        for dt in dates:
            ax.axvline(dt, color=color, lw=0.8, alpha=0.9)
        # 把颜色与 label 绑定一次即可
        ax.axvline(dates[0] if dates else df.index[0],
                   color=color, lw=0.8, label=cls_name.split("_down")[0][:15] + "…",
                   alpha=0.9)

    ax.set_title("NAV 与 30 种 EMA 上/下穿标记")
    ax.legend(loc="upper left", fontsize=7, ncol=3, frameon=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

    return cross_dict



def get_high_ret_by_dates(
    df: pd.DataFrame,
    dates ,                  # list / Index / Series 都行
    *,
    date_col: str = "date",
    high_ret_col: str = "high_ret",
) -> pd.Series:
    """
    按给定日期列表提取对应行的 high_ret 特征（顺序与 dates 一致）。

    Parameters
    ----------
    df : pd.DataFrame
        已含 high_ret 列的数据表。
    dates : Sequence
        日期列表、DatetimeIndex、Series 均可。
    date_col : str, default "date"
        日期列名（若日期已设为索引则忽略）。
    high_ret_col : str, default "high_ret"
        要提取的列名。

    Returns
    -------
    pd.Series
        index 为日期，values 为 high_ret。
        若某日期在 df 中找不到，对应值为 NaN。
    """
    # 先统一成 datetime64，避免字符串比较误差
    dates = pd.to_datetime(dates)

    if df.index.name == date_col:  # 日期已经是索引
        out = df.loc[dates, high_ret_col]
    else:                          # 日期还在普通列
        tmp = (
            df[[date_col, high_ret_col]]
            .assign(**{date_col: pd.to_datetime(df[date_col])})
            .set_index(date_col)
        )
        out = tmp.reindex(dates)[high_ret_col]

    # 保证结果顺序与传入 dates 一致
    out.index = dates
    return out

import optuna

N_FLAGS = 30              # 布尔开关数量

def objective(trial: optuna.trial.Trial) -> float:
    """
    由 Optuna 自动调用。trial 持有随机/贝叶斯采样器。
    返回值 = 策略收益率；Optuna 会自动最大化它。
    """
    # ---- 1) 生成 30 个 bool 参数 ----
    params = {
        f"flag_{i}": trial.suggest_int(f"flag_{i}", 0, 1) == 1  # 0/1 -> False/True
        for i in range(N_FLAGS)
    }

    # ---- 2) 运行你的策略 / 回测 ----
    #########################################
    cross_point_sets = merge_selected_by_order(ema_dict,params)

    strategy = CrossPointBuyStrategy(cross_point_sets)
    # 创建回测引擎
    backtester = Backtester(strategy)
    
    # 使用模拟数据运行测试
    backtester.data = df
    backtester.run_backtest()

    # 获取结果
    results = backtester.get_results()
    ###############################################

    r = results['performance']['total_return']-results['performance']['max_drawdown']*0.5
    print(results['performance']['total_return'],results['performance']['max_drawdown'])
    # ---- 3) 返回值越大越好 ----
    return r


from typing import Dict, List, Set, Hashable, Iterable

def merge_selected_by_order(
    list_dict: Dict[Hashable, List],
    flag_dict: Dict[Hashable, bool],
) -> Set:
    """
    按字典插入顺序将 list_dict 与 flag_dict 对齐，
    对应位置 flag 为 True 的列表统一合并并去重，返回 set。

    参数
    ----
    list_dict : Dict[Any, List]
        值为列表的有序字典。
    flag_dict : Dict[Any, bool]
        值为布尔的有序字典。第 i 个布尔对应 list_dict 的第 i 个列表。

    返回
    ----
    Set
        所有满足 flag==True 的列表元素去重后的集合。

    异常
    ----
    ValueError
        两字典长度不同。
    TypeError
        flag_dict 中出现非布尔值。
    """
    if len(list_dict) != len(flag_dict):
        raise ValueError(
            f"Length mismatch: list_dict={len(list_dict)}, flag_dict={len(flag_dict)}"
        )

    merged: Set = set()
    for values, flag in zip(list_dict.values(), flag_dict.values()):
        if not isinstance(flag, bool):
            raise TypeError(f"Expected bool, got {type(flag).__name__}: {flag!r}")
        if flag:
            merged.update(values)

    return merged
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- 改成完全向量化的安全版 RSI，避免 iat 写入失败 ----
def _rsi(series: pd.Series, n: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)

    roll_up   = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    roll_down = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

    rs  = roll_up / roll_down
    rsi = 100 - 100 / (1 + rs)
    return rsi


def plot_nav_rsi(
    df: pd.DataFrame,
    nav_col: str = "nav",
    date_col: str | None = None,
    rsi_periods: tuple[int, ...] = (6, 9, 14, 26),
    save_path: str | Path = "nav_rsi.png",
    rsi_csv_path: str | Path | None = "rsi_values.csv",
    figsize: tuple[int, int] = (12, 7),
) -> Path:
    """
    绘制 nav 折线 + 多条 RSI，并保存图片；可选保存 RSI CSV。
    附带多重异常检查，遇到问题立即抛出精准提示。
    """
    # ---------- 1. 基础校验 ----------
    if nav_col not in df.columns:
        raise ValueError(f"[×] 列 '{nav_col}' 不存在。请确认 df.columns = {list(df.columns)}")

    if date_col is not None and date_col not in df.columns:
        raise ValueError(f"[×] 日期列 '{date_col}' 不存在。")

    if len(df) < max(rsi_periods) + 1:
        raise ValueError(f"[×] 行数 {len(df)} 不足以计算最长周期 RSI({max(rsi_periods)}).")

    # ---------- 2. 准备数据 ----------
    data = df.copy()
    if date_col is not None:
        data = data.set_index(date_col)

    data = data.sort_index()
    # 强制 nav 转数值
    data[nav_col] = pd.to_numeric(data[nav_col], errors="coerce")

    if data[nav_col].isna().all():
        raise ValueError(f"[×] '{nav_col}' 列全部无法转成数值，可能是空值/字符串。")

    # ---------- 3. 计算 RSI ----------
    for n in rsi_periods:
        col = f"RSI{n}"
        data[col] = _rsi(data[nav_col], n)
        if data[col].isna().all():
            raise ValueError(f"[×] 计算 {col} 得到全 NaN。检查 nav 数据是否断档，或样本太短。")

    # ---------- 4. 可选保存 CSV ----------
    if rsi_csv_path is not None:
        rsi_cols = [f"RSI{n}" for n in rsi_periods]
        sub = data
        if sub.empty:
            raise ValueError("[×] RSI 结果全部 NaN，CSV 未写入。")
        sub.to_csv(rsi_csv_path, index=True)
        print(f"[✓] RSI 已保存至: {Path(rsi_csv_path).resolve()} (共 {len(sub)} 行)")

    # ---------- 5. 画图 ----------
    fig, (ax_price, ax_rsi) = plt.subplots(
        2, 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 2]}
    )

    ax_price.plot(data.index, data[nav_col], label=nav_col, lw=1.2)
    ax_price.set_ylabel("NAV")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, ls="--", lw=0.3, alpha=0.6)

    for n in rsi_periods:
        ax_rsi.plot(data.index, data[f"RSI{n}"], label=f"RSI {n}", lw=1)
    ax_rsi.axhline(70, color="gray", ls="--", lw=0.8)
    ax_rsi.axhline(30, color="gray", ls="--", lw=0.8)
    ax_rsi.set_ylabel("RSI")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper left", ncol=2)
    ax_rsi.grid(True, ls="--", lw=0.3, alpha=0.6)

    fig.tight_layout()
    save_path = Path(save_path)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] 图已保存至: {save_path.resolve()}")
    return save_path




# 示例用法
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    df = get_zz500()

    if args.mode == 'test':
        result = test_strategy(df,strategy)
        print(result)
    elif args.mode == 'plot':
        print(df[['high', 'open']].dtypes)
        df['high_ret'] = (df['high']-df['open'])/df['open']
        print(df.head())
        
        date_list,df=plot_ma20_with_extrema(
            df,
            price_col="nav",
            date_col="date",
            out_path="ma20_advanced.png",
            extrema_distance=7,   # 至少相隔 7 个交易日
            prominence=10,        # 峰值显著性阈值；根据数据大小酌情调整
        )
        high_ret_series = get_high_ret_by_dates(df, date_list)
        print(high_ret_series)
    elif args.mode == 'plot_rsi':
      # 假设 df 已经在内存中，包含 'date' 与 'nav' 两列
      plot_nav_rsi(
        df,
        nav_col="nav",
        date_col="date",
        save_path="nav_rsi.png",
        rsi_csv_path="nav_rsi_values.csv"   # 想关掉保存就传 None
    )

    elif args.mode == 'ema':
      ema_dict = plot_nav_ema_crosses(df)
      for key,val in ema_dict.items():
        print(key,len(val))
      study = optuna.create_study(
          direction="maximize",           # 最大化收益率
          sampler=optuna.samplers.TPESampler(multivariate=True, group=True),  # 默认 TPE
      )

      study.optimize(objective, n_trials=1000, show_progress_bar=True)
      print("最佳收益率:", study.best_value)
      print("最佳参数:")
      for k, v in study.best_params.items():
          print(f"  {k}: {v}")
      from pathlib import Path          # ← 新增
      import json                      # 其余已有的 import 保持不变
      

      best_params = study.best_params
      save_path = Path("best_params.json")
      with save_path.open("w", encoding="utf-8") as f:
          json.dump(best_params, f, ensure_ascii=False, indent=2)


      cross_point_sets = merge_selected_by_order(ema_dict,best_488params)
      strategy = CrossPointBuyStrategy(cross_point_sets)
      result = test_strategy(df,strategy)





  








