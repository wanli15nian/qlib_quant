## LARA 

This repository contains the code to replicate the results of our proposed **LARA** on *Qlib platform*. The project supports the reproduction of two open source datasets: ``Alpha158``, ``Alpha360``

### Getting Started

1. Install qlib and prepare the data as their [suggestions](https://github.com/microsoft/qlib).

   ```bash
   pip install numpy
   pip install --upgrade cython
   
   git clone https://github.com/microsoft/qlib.git && cd qlib
   python setup.py install
   
   python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
   ```

2. Install dependencies:

   ```bash
   pip install hnswlib==0.4.0
   pip install metric-learn==0.6.2
   ```

3. For training

   ```bash
   python main.py --dataset $dataset --train 
   ```
   
   Parameter configuration:
   
   - ``$dataset``: Dataset: ``[Alpha158, Alpha360]``

4. For testing

   ```bash
   python main.py --dataset $dataset
   ```

   Parameter configuration:

   - ``$dataset``: Dataset: ``[Alpha158, Alpha360]``

### Resources 

- Our code is publicly available at https://tinyurl.com/LARA-KDD2022. 

- Data is publicly available at https://github.com/microsoft/qlib. You should preprocess data according to their [instructions](https://github.com/microsoft/qlib).

- Additional two pages of supplemental material in included in ``LARA_KDD_2022_sup.pdf``.