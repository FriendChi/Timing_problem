import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import akshare as ak
import baostock as bs
import pandas as pd
import logging
from typing import List, Optional

import baostock as bs
import pandas as pd

def get_zz500():

    # 登录系统
    lg = bs.login()

    # 查询历史K线数据（示例为日线）
    rs = bs.query_history_k_data_plus(
        code="sh.000905",  # 中证500指数代码
        fields="date,code,open,high,low,close,volume,amount,pctChg",
        start_date="2017-01-01",
        end_date="2025-06-08",
        frequency="d",  # d为日线，w为周线，m为月线
        adjustflag="2"   # 2表示前复权
    )

    # valuation_df = ak.index_value_hist(
    #     symbol="000905",
    #     start_date="20170101",
    #     end_date="20250608"
    # )
    # valuation_df['date'] = pd.to_datetime(valuation_df['date'])


    # 转换为DataFrame
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)

    # 登出系统
    bs.logout()
    df.rename(columns={'close': 'nav'}, inplace=True)
    df['nav'] = pd.to_numeric(df['nav'])
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

    # 删除 code 列（若不存在则跳过）
    out = df.drop(columns=['code'], errors="ignore").copy()

    # 找到需要转型的列（除 date 之外）
    cols_to_convert = out.columns.difference(['date'])

    # 用 pd.to_numeric 转 float，非数字→NaN
    out[cols_to_convert] = out[cols_to_convert].apply(
        pd.to_numeric, errors="coerce"
    ).astype(float)


    # # 4. 合并行情数据和估值数据
    # df = pd.merge(
    #     df, 
    #     valuation_df,
    #     on='date',
    #     how='left'  # 保留所有交易日数据，非交易日估值可能为NaN
    # )

    # 查看数据
    print(df.head())
    return df


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


class FixedPercentStrategy(BaseStrategy):
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
        self.strategy_name = "Fixed Percent Strategy"
        
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
        处理每日数据，生成交易信号
        
        参数:
        date: 当前日期
        nav: 当前净值
        context: 回测上下文信息
        
        返回:
        tuple: (交易信号, 交易信息)
        """
        actions = []
        
        # 1. 首日建仓逻辑
        if context['first_day']:
            if self.state['cash'] > 0:
                actions.append("首日建仓")
                self.state['buy_reference_price'] = nav
                return 'buy', actions
            else:
                return None, actions
        
        # 2. 检查买入条件（净值下跌超过阈值）
        if self.state['buy_reference_price'] is not None:
            # 计算相对于买入参考点的跌幅
            drop_pct = (self.state['buy_reference_price'] - nav) / self.state['buy_reference_price']
            
            # 触发买入条件且仍有现金
            if drop_pct >= self.buy_drop_pct and self.state['cash'] > 0:
                trade_unit = min(self.state['cash'], self.get_trade_unit())
                actions.append(f"净值下跌{drop_pct:.2%}，触发买入")
                self.state['buy_reference_price'] = nav  # 更新买入参考点
                return 'buy', actions
        
        # 3. 检查是否进入卖出阶段（盈利超过阈值）
        if (not self.state['in_selling_phase'] 
            and self.state['total_shares'] > 0
            and self.state['cost_basis'] > 0
            and (nav - self.state['cost_basis']) / self.state['cost_basis'] >= self.profit_target_pct):
            
            self.state['in_selling_phase'] = True
            self.state['sell_reference_price'] = nav  # 设置卖出参考点
            actions.append(f"盈利达到{self.profit_target_pct:.0%}，开始卖出阶段")
        
        # 4. 卖出阶段处理
        if self.state['in_selling_phase'] and self.state['total_shares'] > 0:
            # 检查是否应该卖出（盈利超过目标百分比）
            if self.state['sell_reference_price'] is not None:
                profit_pct = (nav - self.state['sell_reference_price']) / self.state['sell_reference_price']
                
                if profit_pct >= self.profit_target_pct:
                    trade_unit = min(self.state['total_shares'], self.get_trade_unit(sell_mode=True))
                    actions.append(f"相对于卖出点盈利{profit_pct:.2%}，触发卖出")
                    self.state['sell_reference_price'] = nav  # 更新卖出参考点
                    return 'sell', actions
            
            # 如果已经卖出所有持仓，结束卖出阶段
            if self.state['total_shares'] <= 0:
                self.state['in_selling_phase'] = False
                actions.append("持仓已全部卖出，结束卖出阶段")
        
        return None, actions


class FourPctStrategy(BaseStrategy):
    """4%定投策略"""
    def __init__(self, initial_cash=100000, 
                 buy_threshold=0.04, 
                 profit_threshold=0.15, 
                 drawdown_threshold=0.04):
        """
        初始化4%定投策略
        
        参数:
        initial_cash (float): 初始投资额，默认为100,000元
        buy_threshold (float): 加仓阈值（下跌百分比），默认为4%
        profit_threshold (float): 止盈阈值（超过成本的百分比），默认为15%
        drawdown_threshold (float): 止盈回撤阈值，默认为4%
        """
        super().__init__(
            initial_cash=initial_cash,
            buy_threshold=buy_threshold,
            profit_threshold=profit_threshold,
            drawdown_threshold=drawdown_threshold
        )
        self.strategy_name = "4% DCA (Dollar-Cost Averaging) strategy"
        
        # 初始化特定参数
        self.buy_threshold = buy_threshold
        self.profit_threshold = profit_threshold
        self.drawdown_threshold = drawdown_threshold
        
    def reset(self):
        """重置策略状态"""
        super().reset()
        # 重置特定状态变量
        self.state.update({
            'reference_price': None,
            'high_water_mark': 0,
            'in_profit_taking': False
        })
    
    def on_data(self, date, nav, context):
        """
        处理每日数据，生成交易信号
        """
        actions = []
        
        # 1. 首日建仓逻辑
        if context['first_day']:
            if self.state['cash'] > 0:
                actions.append("建仓")
                self.state['reference_price'] = nav
                return 'buy', actions
            else:
                return None, actions
        
        # 2. 检查加仓条件（不在止盈期且净值下跌超过阈值）
        if (not self.state['in_profit_taking'] 
            and self.state['reference_price'] is not None 
            and nav <= self.state['reference_price'] * (1 - self.buy_threshold)):
            
            # 只加仓现金充足的情况
            trade_unit = self.get_trade_unit()
            if self.state['cash'] >= trade_unit:
                actions.append(f"加仓")
                self.state['reference_price'] = nav  # 更新参考价格
                return 'buy', actions
            else:
                actions.append("资金不足")
        
        # 3. 检查是否进入止盈期
        if (not self.state['in_profit_taking'] 
            and self.state['total_shares'] > 0 
            and nav >= self.state['cost_basis'] * (1 + self.profit_threshold)):
            
            self.state['in_profit_taking'] = True
            self.state['high_water_mark'] = nav
            actions.append("进入止盈期")
        
        # 4. 止盈期处理
        if self.state['in_profit_taking'] and self.state['total_shares'] > 0:
            # 更新最高水位
            if nav > self.state['high_water_mark']:
                self.state['high_water_mark'] = nav
            
            # 检查回撤是否超过阈值
            if nav <= self.state['high_water_mark'] * (1 - self.drawdown_threshold):
                actions.append(f"止盈")
                self.state['in_profit_taking'] = False
                self.state['reference_price'] = None  # 重置参考点
                return 'sell', actions
        
        return None, actions


class Backtester:
    """回测引擎"""
    def __init__(self, strategy):
        """
        初始化回测引擎
        
        参数:
        strategy: 策略实例，需继承自BaseStrategy
        """
        self.strategy = strategy
        self.data = None
        self.daily_records = []
        self.trade_records = []
        self.performance_metrics = None
        self.results_df = None
    
    def load_data(self, csv_path):
        """
        从CSV文件加载基金净值数据
        
        参数:
        csv_path (str): CSV文件路径，需包含'date'和'nav'列
        """
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        self.data = df.sort_values('date').reset_index(drop=True)
        return self
    
    def execute_trade(self, date, nav, trade_type, amount=None, shares=None):
        """
        执行交易操作（支持分批买入/卖出）
        
        参数:
        date: 交易日期
        nav: 基金净值
        trade_type: 交易类型 ('buy'或'sell')
        amount: 交易金额（用于买入或卖出指定金额）
        shares: 交易份额（用于卖出指定份额）
        
        返回:
        trade_record: 交易记录字典
        
        注意: 
        - 买入操作优先使用amount参数
        - 卖出操作优先使用shares参数
        """
        state = self.strategy.state
        
        # 预处理参数
        if trade_type == 'buy':
            # 买入逻辑
            if amount is None and shares is not None:
                amount = shares * nav  # 如果提供份额则计算金额
            
            if amount is None or amount <= 0:
                raise ValueError("买入操作需要指定有效的交易金额或份额")
                
            if amount > state['cash']:
                raise ValueError(f"尝试买入{amount:.2f}元，但现金只有{state['cash']:.2f}元")
                
            shares_to_trade = amount / nav
        
        elif trade_type == 'sell':
            # 卖出逻辑
            if shares is None:
                if amount is not None:
                    shares = amount / nav  # 如果提供金额则计算份额
                else:
                    # 默认全部卖出
                    shares = state['total_shares']
                    amount = state['total_shares'] * nav
            
            if shares <= 0 or shares > state['total_shares']:
                available_shares = state['total_shares']
                raise ValueError(
                    f"尝试卖出{shares:.2f}份，但持有份额只有{available_shares:.2f}份"
                )
            
            shares_to_trade = shares
            amount = shares_to_trade * nav
        
        else:
            raise ValueError(f"无效的交易类型: {trade_type}")
        
        # 执行交易并更新状态
        if trade_type == 'buy':
            # 买入：减少现金，增加份额
            state['cash'] -= amount
            state['total_shares'] += shares_to_trade
            state['total_invested'] += amount
            
            # 更新成本基准（基于最新持仓）
            if state['total_shares'] > 0:
                state['cost_basis'] = state['total_invested'] / state['total_shares']
            else:
                state['cost_basis'] = 0
                
            action_desc = f"买入 {shares_to_trade:.2f}份"
        
        else:  # sell
            # 计算卖出部分占总投资的比例
            sold_ratio = shares_to_trade / state['total_shares'] if state['total_shares'] > 0 else 0
            
            # 卖出：增加现金，减少份额和相应投资成本
            state['cash'] += amount
            state['total_shares'] -= shares_to_trade
            state['total_invested'] -= state['total_invested'] * sold_ratio
            
            # 更新成本基准
            if state['total_shares'] > 0:
                # 保留剩余持仓的成本（不需要重新计算）
                state['cost_basis'] = state['cost_basis']
            else:
                state['cost_basis'] = 0
                state['total_invested'] = 0
                
            action_desc = f"卖出 {shares_to_trade:.2f}份"
        
        # 创建交易记录
        trade_record = {
            'date': date,
            'type': trade_type,
            'price': nav,
            'shares': shares_to_trade,
            'amount': amount,
            'action': action_desc
        }
        self.trade_records.append(trade_record)
        
        return trade_record

    def execute_sell(self, date, nav):
        """执行全部卖出操作"""
        state = self.strategy.state
        
        if state['total_shares'] <= 0:
            raise ValueError("尝试卖出但持有份额为0")
            
        # 计算收益
        amount = state['total_shares'] * nav
        
        # 更新状态
        state['cash'] += amount
        shares_sold = state['total_shares']
        state['total_shares'] = 0
        state['total_invested'] = 0
        state['cost_basis'] = 0
        
        # 记录交易
        trade_record = {
            'date': date, 
            'type': 'sell',
            'price': nav,
            'shares': shares_sold,
            'amount': amount,
            'action': f"卖出全部 {shares_sold:.2f}份"
        }
        self.trade_records.append(trade_record)
        
        return trade_record

    def record_daily_status(self, date, nav, action=""):
        """记录每日持仓状态"""
        state = self.strategy.state
        
        asset_value = state['total_shares'] * nav
        total_assets = state['cash'] + asset_value
        
        self.daily_records.append({
            'date': date,
            'nav': nav,
            'cash': state['cash'],
            'shares': state['total_shares'],
            'cost_basis': state['cost_basis'] if state['total_shares'] > 0 else None,
            'asset_value': asset_value,
            'total_assets': total_assets,
            'in_profit_taking': state['in_profit_taking'],
            'action': action
        })
    
    def __create_input(self,i):
        #获取当日数据
        row = self.data.iloc[i]
        date = row['date']
        nav = row['nav']
        
        # 创建上下文
        context = {
            'trade_day_idx':i,
            'previous_row': self.data.iloc[i-1],
            'current_row': row
        }

        return date, nav, context

    def run_backtest(self):
        """运行策略回测"""
        if not hasattr(self, 'data') or self.data.empty:
            raise ValueError("请先加载数据")
        
        # 初始化策略状态
        self.strategy.reset()
        self.daily_records = [] #每天记录
        self.trade_records = [] #交易记录
        
        # 遍历剩余交易日
        for i in range(0, len(self.data)):
            # 获取进行决策的输入数据
            date, nav, context = self.__create_input(i)
            
            # 获取策略信号
            trade, action_info = self.strategy.on_data(date, nav, context)
            action_desc = ", ".join(action_info)
            
            # 执行交易
            if trade['trade_type'] is not None:
                self.execute_trade(date, nav, trade_type=trade['trade_type'],amount=trade['amount'],shares=trade['share'])
            
            # 记录状态
            self.record_daily_status(date, nav, action=action_desc)
        
        # 最后一天清算剩余持仓
        if self.strategy.state['total_shares'] > 0:
            last_row = self.data.iloc[-1]
            last_date = last_row['date']
            last_nav = last_row['nav']
            self.execute_sell(last_date, last_nav)
            
            # 更新最后一天状态
            self.daily_records[-1]['action'] = self.daily_records[-1].get('action', "") + ", 清算"
        
        # 创建结果DataFrame
        self.results_df = pd.DataFrame(self.daily_records)
        self.trades_df = pd.DataFrame(self.trade_records)
        
        # 计算业绩指标
        self.calculate_performance()
        
        return self
    
    def calculate_performance(self):
        """计算策略表现指标"""
        if self.results_df is None or self.results_df.empty:
            raise ValueError("请先运行回测")
        
        # 最终资产
        final_value = self.results_df.iloc[-1]['total_assets']
        
        # 总收益率
        total_return = (final_value - self.strategy.initial_cash) / self.strategy.initial_cash
        
        # 计算年化收益率
        start_date = self.results_df['date'].min()
        end_date = self.results_df['date'].max()
        days = (end_date - start_date).days
        years = max(days / 365.25, 0.1)  # 防止除数为0
        annualized_return = (1 + total_return) ** (1/years) - 1
        
        # 计算最大回撤
        peak = self.results_df['total_assets'].cummax()
        drawdown = (self.results_df['total_assets'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # 计算交易指标
        buys = self.trades_df[self.trades_df['type'] == 'buy']
        sells = self.trades_df[self.trades_df['type'] == 'sell']
        
        self.performance_metrics = {
            'strategy': self.strategy.strategy_name,
            'initial_cash': self.strategy.initial_cash,
            'final_value': final_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades_df),
            'num_buys': len(buys),
            'num_sells': len(sells),
            'total_buy_amount': buys['amount'].sum(),
            'total_sell_amount': sells['amount'].sum()
        }
        
        return self.performance_metrics
    
    def get_results(self):
        """获取回测结果"""
        return {
            'daily_results': self.results_df,
            'trades': self.trades_df,
            'performance': self.performance_metrics
        }

    def plot_performance(self, save_path=None):
        """可视化策略表现"""
        if self.results_df is None:
            raise ValueError("请先运行回测")
            
        df = self.results_df.copy()
        
        plt.figure(figsize=(14, 12))
        
        # 1. 净值与交易点
        plt.subplot(3, 1, 1)
        plt.plot(df['date'], df['nav'], label='NAV', color='blue')
        
        # 标记交易点
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            buy_trades = self.trades_df[self.trades_df['type'] == 'buy']
            sell_trades = self.trades_df[self.trades_df['type'] == 'sell']
            
            plt.scatter(buy_trades['date'], buy_trades['price'], 
                        color='red', marker='^', s=80, label='Buy points')
            plt.scatter(sell_trades['date'], sell_trades['price'], 
                        color='green', marker='v', s=80, label='Sell points')
        
        plt.title(f'{self.strategy.strategy_name} - Fund NAV Trend and Trading Points')
        plt.ylabel('NAV')
        plt.legend()
        plt.grid(True)
        
        # 2. 资产价值变化
        plt.subplot(3, 1, 2)
        plt.plot(df['date'], df['total_assets'], 
                 label='Total asset', color='purple', linewidth=2)
        plt.axhline(y=self.strategy.initial_cash, color='gray', linestyle='--', label='Initial principal')
        
        # 标记策略阶段
        if 'in_profit_taking' in df.columns:
            for i, row in enumerate(df.itertuples()):
                if row.in_profit_taking:
                    plt.axvspan(df['date'].iloc[i-1] if i > 0 else row.date, 
                               row.date, color='lightyellow', alpha=0.3, label='Profit-taking period' if i == 0 else "")
        
        plt.title('Total Asset Value Change')
        plt.ylabel('Asset value (CNY)')
        plt.legend()
        plt.grid(True)
        
        # 3. 持仓数量变化
        plt.subplot(3, 1, 3)
        plt.bar(df['date'], df['shares'], label='Holding shares', color='orange', alpha=0.7)
        
        # 标记买入点
        if hasattr(self, 'trades_df') and not self.trades_df.empty:
            buy_trades = self.trades_df[self.trades_df['type'] == 'buy']
            plt.scatter(buy_trades['date'], [0] * len(buy_trades), 
                        color='red', marker='^', s=60, label='Buy')
        
        plt.title('Holding Shares Change')
        plt.ylabel('Holding shares')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            
        plt.show()

    def generate_report(self):
        """生成策略报告"""
        if self.performance_metrics is None:
            self.calculate_performance()
            
        metrics = self.performance_metrics
        
        report = f"""
        ======= {self.strategy.strategy_name} 回测报告 =======
        
        回测期间: {self.results_df['date'].min().strftime('%Y-%m-%d')} 至 {self.results_df['date'].max().strftime('%Y-%m-%d')}
        初始投资额: ¥{metrics['initial_cash']:,.2f}
        最终资产价值: ¥{metrics['final_value']:,.2f}
        ----------------------------------
        总收益率: {metrics['total_return'] * 100:.2f}%
        年化收益率: {metrics['annualized_return'] * 100:.2f}%
        最大回撤: {abs(metrics['max_drawdown']) * 100:.2f}%
        ----------------------------------
        交易统计:
          总交易次数: {metrics['num_trades']}
          买入操作: {metrics['num_buys']} 次 (总计¥{metrics['total_buy_amount']:,.2f})
          卖出操作: {metrics['num_sells']} 次 (总计¥{metrics['total_sell_amount']:,.2f})
        """
        
        # 添加额外分析
        if metrics['final_value'] > metrics['initial_cash'] * 1.5:
            report += "\n策略表现优异，大幅超越初始投资"
        elif metrics['final_value'] > metrics['initial_cash']:
            report += "\n策略表现良好，取得正收益"
        else:
            report += "\n策略表现欠佳，未能实现正收益"
            
        # 添加风险提示
        report += "\n\n注：回测基于历史数据，不代表未来表现"
        
        return report.strip()


def test_strategy(df):
        # 创建策略
    # strategy = FourPctStrategy(
    #     initial_cash=100000,
    #     buy_threshold=0.04,
    #     profit_threshold=0.15,
    #     drawdown_threshold=0.04
    # )
    strategy = FixedPercentStrategy_cost_basis()
    
    # 创建回测引擎
    backtester = Backtester(strategy)
    
    # 创建DataFrame
    # df = pd.read_csv('/app/Timing_problem/rlData.csv', parse_dates=['Date'])
    # df.rename(columns={'Date': 'date', 'Close': 'nav'}, inplace=True)
    
    # 使用模拟数据运行测试
    backtester.data = df
    backtester.run_backtest()
    
    # 获取结果
    results = backtester.get_results()
    
    # 打印报告
    print(backtester.generate_report())
    
    # 可视化结果
    backtester.plot_performance(save_path='4pct_strategy_performance.png')
    
    # 保存结果到CSV
    results['daily_results'].to_csv('daily_results.csv', index=False)
    results['trades'].to_csv('trade_records.csv', index=False)

import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        action='store',
        type=str,
        default='test',
        choices=['test', 'plot'],
    )

    return parser

import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.signal import find_peaks
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

    return minima_dates  # ⚡️ 返回



# 示例用法
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    df = get_zz500()

    if args.mode == 'test':
        test_strategy(df)
    elif args.mode == 'plot':
        df['high_ret'] = (df['high']-df['open'])/df['open']
        print(df.head())
        
        date_list=plot_ma20_with_extrema(
            df,
            price_col="nav",
            date_col="date",
            out_path="ma20_advanced.png",
            extrema_distance=7,   # 至少相隔 7 个交易日
            prominence=10,        # 峰值显著性阈值；根据数据大小酌情调整
        )
        print(date_list)








