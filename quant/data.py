def get_zz500():

    # 登录系统
    lg = bs.login()

    # 查询历史K线数据（示例为日线）
    rs = bs.query_history_k_data_plus(
        code="sh.000905",  # 中证500指数代码
        fields="date,code,open,high,low,close,volume,amount,pctChg",
        start_date="2017-01-01",
        end_date="2024-01-01",
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

    out["MA20"] = out['nav'].rolling(20).mean()
    for i in [10,12,13,18,20,26]:
      out[f"EMA{str(i)}"] = out['nav'].ewm(span=i, adjust=False).mean()  # i日指数移动平均线
    
    # out["MA20_smooth"] = out["MA20"].rolling(7, center=True).mean()

    # # 4. 合并行情数据和估值数据
    # df = pd.merge(
    #     df, 
    #     valuation_df,
    #     on='date',
    #     how='left'  # 保留所有交易日数据，非交易日估值可能为NaN
    # )

    # 查看数据
    print(out.head())
    return out