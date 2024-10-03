#encoding=gbk
import numpy as np

class ReversalFrequency:
    def __init__(self, overnight_returns, daytime_returns):
        """
        ��ʼ��
        :param overnight_returns: ��ҹ�������б�
        :param daytime_returns: �����������б�
        """
        self.overnight_returns = np.array(overnight_returns)
        self.daytime_returns = np.array(daytime_returns)

    def calc_reversal_frequency(self):
        """
        ���㷴��������ת��Ƶ��
        :return: ����������ת��Ƶ��
        """
        T = len(self.overnight_returns)
        indicator = (self.overnight_returns > 0) & (self.daytime_returns < 0)
        return np.sum(indicator) / T

# ʾ���÷�
# ��ҹ������
overnight_returns = [0.01, -0.02, 0.015, -0.007, 0.03, -0.012, 0.025]

# ����������
daytime_returns = [-0.008, 0.01, -0.005, 0.004, -0.02, 0.011, -0.015]

# ���㷴��������ת��Ƶ��
reversal_freq_calculator = ReversalFrequency(overnight_returns, daytime_returns)
frequency = reversal_freq_calculator.calc_reversal_frequency()

print(f"����������ת��Ƶ��: {frequency}")
