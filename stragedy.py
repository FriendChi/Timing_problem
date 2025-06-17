import pandas as pd
import numpy as np
from scipy.signal import find_peaks


class BaseStrategy:
    """策略基类，定义策略接口"""
    def __init__(self, initial_cash=100000, **params):
        """
        初始化策略参数
        
        参数:
        initial_cash (float): 初始投资额
        params: 策略特定参数
        """
        self.initial_cash = initial_cash
        self.params = params
        self.strategy_name = "Base Strategy"
        
        # 重置策略状态
        self.reset()
    
    def reset(self):
        """重置策略状态"""
        self.state = {
            'cash': self.initial_cash,
            'total_shares': 0, #持股量
            'total_invested': 0,
            'cost_basis': 0, #每股持仓成本
            'reference_price': None,
            'high_water_mark': 0,
            'in_profit_taking': False
        }
    
    def get_trade_unit(self):
        """计算每次交易的单位（默认初始资金的10%）"""
        return self.initial_cash * 0.1
    
    def on_data(self, date, nav, context):
        """
        处理每日数据，生成交易信号
        
        参数:
        date: 当前日期
        nav: 当前净值
        context: 回测上下文信息
        
        返回:
        tuple: (交易信号, 交易信息)
        交易信号: 'buy' - 买入, 'sell' - 卖出, None - 无操作
        交易信息: 字符串描述或自定义字典
        """
        raise NotImplementedError("子类必须实现此方法")


class BuyXpercent_Substrategy(BaseStrategy):
    """固定十次，分批基于参考点，固定百分比，买入，子策略"""
    def __init__(self, initial_cash=100000, 
                 buy_drop_pct=0.04, 
                 profit_target_pct=0.06, 
                 trade_unit_percent=0.1):
        """
        初始化策略
        
        参数:
        initial_cash (float): 初始投资额，默认为100,000元
        buy_drop_pct (float): 加仓阈值（下跌百分比），默认为4%
        profit_target_pct (float): 盈利目标阈值，默认为6%
        trade_unit_percent (float): 每次交易百分比，默认为10%
        """
        super().__init__(
            initial_cash=initial_cash,
            buy_drop_pct=buy_drop_pct,
            profit_target_pct=profit_target_pct,
            trade_unit_percent=trade_unit_percent
        )

    def buy_logic(self,trade,nav,note):
        if self.state['cost_basis'] is not None: #确定有持仓 
            # 计算相对于参考点的跌幅
            drop_pct = (self.state['bug_reference'] - nav) / self.state['bug_reference']
            
            # 触发买入条件且现金大于购买量
            if drop_pct >= self.buy_drop_pct and self.state['cash'] > self.get_trade_unit(False):
                self.state['bug_reference'] = nav
                trade['trade_type'] = 'buy'
                trade['amount'] = self.get_trade_unit(False)
                note.append(f"相对每股持仓成本净值下跌{drop_pct:.2%}，触发买入")
                return trade, note        

class FixedPercentStrategy_cost_basis(BaseStrategy):
    """固定百分比买卖策略"""
    def __init__(self, initial_cash=100000, 
                 buy_drop_pct=0.04, 
                 profit_target_pct=0.06, 
                 trade_unit_percent=0.1):
        """
        初始化策略
        
        参数:
        initial_cash (float): 初始投资额，默认为100,000元
        buy_drop_pct (float): 加仓阈值（下跌百分比），默认为4%
        profit_target_pct (float): 盈利目标阈值，默认为6%
        trade_unit_percent (float): 每次交易百分比，默认为10%
        """
        super().__init__(
            initial_cash=initial_cash,
            buy_drop_pct=buy_drop_pct,
            profit_target_pct=profit_target_pct,
            trade_unit_percent=trade_unit_percent
        )
        self.strategy_name = "Fixed Percent Strategy with cost basis"
        
        # 初始化特定参数
        self.buy_drop_pct = buy_drop_pct
        self.profit_target_pct = profit_target_pct
        self.trade_unit_percent = trade_unit_percent
        
    def reset(self):
        """重置策略状态"""
        super().reset()
        # 重置特定状态变量
        self.state.update({
            'buy_reference_price': None,  # 上一次买入点的价格
            'sell_reference_price': None,  # 上一次卖出点的价格
            'in_selling_phase': False,    # 是否处于卖出阶段
            'total_shares_sold': 0        # 已卖出总份额
        })
    
    def get_trade_unit(self, sell_mode=False):
        """
        计算每次交易的单位
        
        参数:
        sell_mode (bool): 如果是卖出，返回持仓百分比；买入则返回现金百分比
        """
        if sell_mode and self.state['total_shares'] > 0:
            # 卖出：当前持仓的10%
            return self.state['total_shares'] * self.trade_unit_percent
        else:
            # 买入：初始资金的10%
            return self.initial_cash * self.trade_unit_percent
    
    def on_data(self, date, nav, context):
        """
        处理每日数据，生成交易信号，但不改变当前持仓，这是回测的任务
        
        参数:
        date: 当前日期
        nav: 当前净值
        context: 回测上下文信息
        
        返回:
        trade(dict): 交易详情（信号，数值，份额）
        actions(list): 交易备注
        """
        trade = {'trade_type':None,'amount':None,'share':None,}
        note = []
        
        # 1. 首日建仓逻辑：第一天比建仓trade_unit_percent的资金
        if context['trade_day_idx'] == 0:
            if self.state['cash'] > 0:
                note.append("首日建仓")
                trade['trade_type'] = 'buy'
                trade['amount'] = self.get_trade_unit(False)
                return trade, note
            else:
                raise ValueError('首日建仓没有资金')
        
        # 2. 检查买入条件（净值下跌超过阈值）
        if self.state['cost_basis'] is not None:
            # 计算相对于持仓的跌幅
            drop_pct = (self.state['cost_basis'] - nav) / self.state['cost_basis']
            
            # 触发买入条件且现金大于购买量
            if drop_pct >= self.buy_drop_pct and self.state['cash'] > self.get_trade_unit(False):
                trade['trade_type'] = 'buy'
                trade['amount'] = self.get_trade_unit(False)
                note.append(f"相对每股持仓成本净值下跌{drop_pct:.2%}，触发买入")
                return trade, note
        
        # 3. 卖出处理
        if self.state['total_shares'] > 0 : #确认有持仓
            # 检查是否应该卖出（盈利超过目标百分比）
            profit_pct = (nav - self.state['cost_basis']) / self.state['cost_basis']
            
            if profit_pct >= self.profit_target_pct:
                note.append(f"相对于每股持仓成本盈利{profit_pct:.2%}，触发卖出")
                trade['trade_type'] = 'sell'
                trade['share'] = self.get_trade_unit(True)        
                return trade, note
            else:
                note.append(f"相对于每股持仓成本盈利{profit_pct:.2%}，没有触发卖出")
    
        return trade, note


class MA20ExtremaStrategy(BaseStrategy):
    """基于20日均线极小值点买入，盈利达到4%时全部卖出的策略"""
    
    def __init__(self, initial_cash=100000, 
                 profit_target_pct=0.04, 
                 trade_unit_percent=0.1, 
                 extrema_distance=5, 
                 prominence=0.05):
        """
        初始化策略
        
        参数:
        initial_cash (float): 初始投资额，默认为100,000元
        profit_target_pct (float): 盈利目标阈值，默认为4%
        trade_unit_percent (float): 每次交易的百分比，默认为10%
        extrema_distance (int): 查找极值的最小间距，默认为5
        prominence (float): 查找极值的显著性，默认为0.05
        """
        super().__init__(
            initial_cash=initial_cash
        )

        self.strategy_name = "MA20 Extrema Strategy"
        
        # 初始化特定参数
        self.profit_target_pct=profit_target_pct,
        self.trade_unit_percent=trade_unit_percent
        self.extrema_distance = extrema_distance
        self.prominence = prominence
        self.ma_window = 20  # 使用20日均线
        
    def reset(self):
        """重置策略状态"""
        super().reset()
        # 重置特定状态变量
        self.state.update({
            'last_buy_price': None,  # 上一次买入点的价格
            'in_selling_phase': False,  # 是否处于卖出阶段
        })
    
    def get_trade_unit(self, sell_mode=False):
        """
        计算每次交易的单位
        
        参数:
        sell_mode (bool): 如果是卖出，返回持仓百分比；买入则返回现金百分比
        """
        if sell_mode and self.state['total_shares'] > 0:
            # 卖出：当前持仓的全部
            return self.state['total_shares']
        else:
            # 买入：初始资金的10%
            return self.initial_cash * self.trade_unit_percent
    def get_min_flag(self, 
            context: dict,
            offset: int = 1,          # 0=今天,1=昨天...
            lookback: int = 19,       # 包含今天在内的窗口长度
            smooth_window: int = 7,   # Savitzky-Golay 滑窗长度(必须为奇数且 ≤ lookback)
            polyorder: int = 2        # Savitzky-Golay 多项式阶
        ) -> bool:
        """
        判断 MA20 在指定 offset 处是否是局部极小值
        参数:
            context: {'current_row': Series, 'previous_row': DataFrame}
            offset: 0 表示今天, 1 表示昨天, 依此类推
            lookback: 回看长度(含今天); 必须 > offset
            smooth_window: Savitzky-Golay 的窗口长度, 必须为奇数且 ≤ lookback
            polyorder: Savitzky-Golay 的多项式阶
        返回:
            bool: 指定 offset 处是否为局部极小值
        """
        # ---------- 数据准备 ----------
        current_row = context['current_row']
        previous_row = context['previous_row']
        
        # 取最近 lookback-1 行历史数据 (不含今天) 并加上今天
        recent_hist = previous_row['MA20'].dropna().values[-(lookback - 1):]
        ma_today = current_row['MA20']
        
        # 数据不足
        if len(recent_hist) < (lookback - 1) or np.isnan(ma_today):
            return False
        
        ma_window = np.append(recent_hist, ma_today)  # 长度 = lookback
        
        # ---------- Savitzky-Golay 平滑 ----------
        # 保证 smooth_window 为奇数且不超过 ma_window 长度
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = min(smooth_window, len(ma_window))
        smoothed = savgol_filter(ma_window, smooth_window, polyorder)
        
        # ---------- 极小值判定 ----------
        target_idx = -(offset + 1)  # 0-based 从尾部计数
        if abs(target_idx) > len(smoothed):
            return False
        
        target_val = smoothed[target_idx]
        other_vals = np.delete(smoothed, target_idx)
        
        # target 比其它任何点都小 → 局部极小
        return target_val < other_vals.min()



    def on_data(self, date, nav, context):
        """
        处理每日数据，生成交易信号
        
        参数:
        date: 当前日期
        nav: 当前净值
        context: 回测上下文信息
        
        返回:
        trade(dict): 交易详情（信号，数值，份额）
        actions(list): 交易备注
        """
        trade = {'trade_type': None, 'amount': None, 'share': None}
        note = []
        
        # 计算20日均线
        min20_falg = self.get_min_flag(context)
        
        # 1. 检查是否处于买入点（前天为近20天的最小值）
        if min20_falg and self.state['cash'] > self.get_trade_unit(False):
            if self.state['total_shares'] > 0:
                buy_pct =  (nav - self.state['cost_basis']) / self.state['cost_basis']
            else:
                buy_pct = 0
            if buy_pct<-0.04 or buy_pct >= 0: #若收益为负，则需亏损4%才可买入
                note.append(f"前一天为20日均线极小值点，触发买入，相比与持有成本，亏损{buy_pct}%")
                trade['trade_type'] = 'buy'
                trade['amount'] = self.get_trade_unit(False)
                self.state['last_buy_price'] = nav  # 更新买入价
                return trade, note
        
        # 2. 卖出处理
        if self.state['total_shares'] > 0:  # 确认有持仓
            # 检查是否达到盈利目标（4%）
            profit_pct = (nav - self.state['last_buy_price']) / self.state['last_buy_price']
            if profit_pct >= self.profit_target_pct:
                note.append(f"盈利达到{profit_pct:.2%}，触发卖出")
                trade['trade_type'] = 'sell'
                trade['share'] = self.get_trade_unit(True)
                return trade, note
            else:
                note.append(f"盈利{profit_pct:.2%}，未达到卖出目标")
        
        return trade, note