#encoding=gbk

class StockMarketAnalysis:
    def __init__(self, stock_data):
        """
        ��ʼ�����������չ�Ʊ���ݡ�
        :param stock_data: DataFrame, ������Ʊ��ҹ�����ʺ����������ʵ����ݡ�
        """
        self.stock_data = stock_data

    def calculate_reverse_intraday_reversal_frequency(self):
        """
        ���㷴��������ת��Ƶ�ʡ�
        :return: Series, ÿֻ��Ʊ�ķ���������תƵ�ʡ�
        """
        # ��ҹ�����ʺ�����������
        overnight_returns = self.stock_data['RET.CO.a']
        intraday_returns = self.stock_data['RET.OC.d']

        # ���㷴��������ת��Ƶ��
        reverse_intraday_reversal = (
            (overnight_returns > 0) & (intraday_returns < 0)
        ).cumsum()

        # ����ÿֻ��Ʊ�ķ���������תƵ��
        reverse_intraday_reversal_frequency = reverse_intraday_reversal / self.stock_data['T']

        return reverse_intraday_reversal_frequency

# ʾ��ʹ��
# ����stock_data��һ��������Ʊ��ҹ�����ʺ����������ʵ�DataFrame
# stock_data = pd.read_csv('stock_data.csv')  # ��������
# analysis = StockMarketAnalysis(stock_data)
# frequencies = analysis.calculate_reverse_intraday_reversal_frequency()
# print(frequencies)
