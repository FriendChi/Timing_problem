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
            'previous_row': self.data.iloc[:i+1],
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
        if self.trades_df.empty:
            raise ValueError("没有交易记录")
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
        self.results_df = pd.merge(self.results_df, self.data, on='date', how='left')

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

def test_strategy(df,strategy):
    # 创建回测引擎
    backtester = Backtester(strategy)
    
    
    # 创建DataFrame
    # df = pd.read_csv('/app/Timing_problem/rlData.csv', parse_dates=['Date'])
    # df.rename(columns={'Date': 'date', 'Close': 'nav'}, inplace=True)
    
    # 使用模拟数据运行测试
    backtester.data = df
    backtester.run_backtest()
    
    # 打印报告
    print(backtester.generate_report())
    
    # 可视化结果
    backtester.plot_performance(save_path='4pct_strategy_performance.png')

    # 获取结果
    results = backtester.get_results()

    # 保存结果到CSV
    results['daily_results'].to_csv('daily_results.csv', index=False)
    results['trades'].to_csv('trade_records.csv', index=False)

    return results