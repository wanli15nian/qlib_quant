#encoding=gbk
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay

class StockReturnAnalysis:
    def __init__(self, stock_data):
        """
        ��ʼ���࣬�����Ʊ���ݵ�DataFrame������Ӧ��������(Date)����ҹ������(RET.COa)������������(RET.OCa)�С�
        """
        self.stock_data = stock_data

    def calculate_abnormal_frequency(self):
        """
        ��������������ת���쳣Ƶ�ʡ�
        """
        # �����ҹ�����ʺ�����������
        self.stock_data['RET.COa'] = self.stock_data.groupby('Date')['RET.COa'].diff().fillna(0)
        self.stock_data['RET.OCa'] = self.stock_data.groupby('Date')['RET.OCa'].diff().fillna(0)
        
        # �������������ת�Ľ�����
        self.stock_data['Positive_Intraday_Reversal'] = (self.stock_data['RET.COa'] < 0) & (self.stock_data['RET.OCa'] > 0)
        
        # ����ÿ���µ�����������תƵ��
        self.stock_data['Month_Year'] = self.stock_data['Date'].dt.to_period('M')
        monthly_data = self.stock_data.groupby(['Month_Year', 'Positive_Intraday_Reversal']).size().unstack(fill_value=0)
        
        # ����ÿ���µ�����������תƵ��
        monthly_data['Frequency'] = monthly_data['True'] / monthly_data['True'].sum(axis=1)
        
        # �����ȥ12���µ�����������תƵ�ʵ�ƽ��ֵ
        months = monthly_data.index.to_perioddelta()
        past_12_months = monthly_data.iloc[:-1 * (months.max() - months.min())]  # ȷ����ȥ12���µ�����
        average_frequency_12_months = past_12_months['Frequency'].mean()
        
        # �����쳣Ƶ��
        monthly_data['Abnormal_Frequency'] = monthly_data['Frequency'] - average_frequency_12_months
        
        return monthly_data

# ʾ���÷�
# ����stock_df��һ��DataFrame����������(Date)����ҹ������(RET.COa)������������(RET.OCa)��
# stock_analysis = StockReturnAnalysis(stock_df)
# abn_frequency = stock_analysis.calculate_abnormal_frequency()

