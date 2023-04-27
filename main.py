# This is a sample Python script.
import csv
import pandas as pd
import numpy as np

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
# import seaborn as sns
#
# # Preprocesado y modelado
# # ==============================================================================
# from scipy.stats import pearsonr
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.metrics import mean_squared_error
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# plt.rcParams['image.cmap'] = "bwr"
# #plt.rcParams['figure.dpi'] = "100"
# plt.rcParams['savefig.bbox'] = "tight"
# style.use('ggplot') or plt.style.use('ggplot')
#
# # Configuración warnings
# # ==============================================================================
# import warnings
# warnings.filterwarnings('ignore')
#
#








# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vineyard = []
    location = []
    case_price = []
    age = []
    acres = []
    varieties = []
    domestic = []
    visitors = []
    buses = []
    employees = []
    awards = []
    sales = []
    income = []

    with open('vineyards-modified.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # Skip header row
                line_count += 1
            else:
                vineyard.append(row[0])
                location.append(row[1])
                case_price.append(float(row[2]))
                age.append(int(row[3]))
                acres.append(float(row[4]))
                varieties.append(int(row[5]))
                domestic.append(int(row[6]))
                visitors.append(float(row[7]))
                buses.append(float(row[8]))
                employees.append(int(row[9]))
                awards.append(float(row[10]))
                sales.append(float(row[11]))
                income.append(float(row[12]))
                line_count += 1

  # print("Vineyard:", vineyard)
  # print("Location:", location)
  # print("CasePrice:", case_price)
  # print("Age:", age)
  # print("Acres:", acres)
  # print("Varieties:", varieties)
  # print("Domestic:", domestic)
  # print("Visitors:", visitors)
  # print("Buses:", buses)
  # print("Employees:", employees)
  # print("Awards:", awards)
  # print("Sales:", sales)
  # print("Income:", income)

    df = pd.DataFrame(list(zip(vineyard, location, case_price, age, acres, varieties, domestic, visitors, buses, employees, awards, sales, income)), columns= ['vineyard', 'location', 'case_price', 'age', 'acres', 'varieties', 'domestic', 'visitors', 'buses', 'employees', 'awards', 'sales', 'income'])
    # print(df.to_string())
    case_price_mean = df['case_price'].mean()