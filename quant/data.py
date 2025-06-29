import baostock as bs
import pandas as pd
from collections.abc import Iterable
from pathlib import Path

class DataManager:
    """
    通用指数（日线）数据抓取器  
    - 默认前复权；  
    - 只在首次 fetch 时联网，其余调用直接返回缓存；  
    - 额外技术指标通过 `add_indicators()` 单独添加，互不耦合。
    """

    def __init__(
        self,
        code: str = "sh.000905",          # ★ 可以自定义指数代码
        date_str: str = "train",
        frequency: str = "d",
        adjustflag: str = "2",
    ) -> None:
        self.date_dict = {
          'train':["2017-01-01","2024-01-01"],
          'val':["2024-01-01","2025-06-01"],
        }
        self.start_date,self.end_date = self.date_dict[date_str]
        self.code = code
        self.fields = "date,code,open,high,low,close,volume,amount,pctChg"
        self.frequency = frequency
        self.adjustflag = adjustflag
        self._df: pd.DataFrame | None = None   # 缓存
        self._base_dir = Path(__file__).resolve().parent  # ★ 同目录

    # ====== 私有工具 ======
    def _save_df(self, df: pd.DataFrame, suffix: str) -> None:
        fname = f"{self.code}_{self.start_date}_{self.end_date}_{suffix}.csv"
        path  = self._base_dir / fname
        df.to_csv(path, index=False)
        print(f"[✓] DataFrame 已保存: {path}")

    # ---------- 核心流程 ----------
    def fetch(self, copy: bool = True) -> pd.DataFrame:
        """返回基础行情数据（未带额外技术指标）"""
        if self._df is None:
            self._df = self._clean(self._query_raw())
        
        # ★ 保存基础版
        self._save_df(self._df, "basic")

        return self._df.copy() if copy else self._df

    @staticmethod
    def _calc_rsi(series: pd.Series, n: int) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)

        roll_up   = gain.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
        roll_down = loss.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

        rs  = roll_up / roll_down
        return 100 - 100 / (1 + rs)

    # ---------- 可选功能 ----------
    def add_indicators(
        self,
        kinds: Iterable[str] | None = ("ma", "ema", "rsi"),   # ★ 默认三种都可
        ma_windows: Iterable[int] | None  = (18,20),
        ema_spans: Iterable[int]  | None  = (10, 12, 13, 18, 20, 26),
        rsi_periods: Iterable[int] | None = (6, 9, 14, 26),  # ★ 新增
    ) -> pd.DataFrame:
        """
        给内部 DataFrame 追加 MA / EMA / RSI
        参数
        -------
        kinds        选择 'ma' | 'ema' | 'rsi'（可组合，不区分大小写）
        ma_windows   仅当 'ma' 被选中时才计算
        ema_spans    仅当 'ema' 被选中时才计算
        rsi_periods  仅当 'rsi' 被选中时才计算
        """
        df = self.fetch(copy=False)
        kinds_norm = {k.lower() for k in (kinds or [])}

        if "ma" in kinds_norm and ma_windows:
            for w in ma_windows:
                df[f"MA{w}"] = df["nav"].rolling(window=w).mean()

        if "ema" in kinds_norm and ema_spans:
            for s in ema_spans:
                df[f"EMA{s}"] = df["nav"].ewm(span=s, adjust=False).mean()

        if "rsi" in kinds_norm and rsi_periods:
            for p in rsi_periods:
                df[f"RSI{p}"] = self._calc_rsi(df["nav"], p)

        # ★ 保存带指标版
        self._save_df(df, "ind")

        return df


    # ---------- 内部工具 ----------
    def _query_raw(self) -> pd.DataFrame:
        lg = bs.login()
        try:
            rs = bs.query_history_k_data_plus(
                code=self.code,
                fields=self.fields,
                start_date=self.start_date,
                end_date=self.end_date,
                frequency=self.frequency,
                adjustflag=self.adjustflag,
            )
            data = [rs.get_row_data() for _ in iter(rs.next, False) if rs.error_code == "0"]
            return pd.DataFrame(data, columns=rs.fields)
        finally:
            bs.logout()

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"close": "nav"}).copy()
        df["date"] = pd.to_datetime(df["date"])
        num_cols = df.columns.difference(["date"])
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").astype(float)
        return df.drop(columns="code", errors="ignore")

# ---------------- 用例 ----------------
if __name__ == "__main__":
  fetcher = DataManager()
  df_feat = fetcher.add_indicators(
      kinds=["rsi"],           # 只要 RSI
      rsi_periods=(6, 14, 26)  # 自定义周期
  )
  print(df_feat.filter(like="RSI").head(10))
