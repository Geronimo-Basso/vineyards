# This is a sample Python script.
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
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

    #---------- Media ----------#

    print("-----Media-----")
    case_price_mean = df['case_price'].mean()
    age_mean = df['age'].mean()
    acres_mean = df['acres'].mean()
    varieties_mean = int(df['varieties'].mean())
    domestic_mean = int(df['domestic'].mean())
    visitors_mean = int(df['visitors'].mean())
    buses_mean = df['buses'].mean()
    employees_mean = int(df['employees'].mean())
    awards_mean = df['awards'].mean()
    sales_mean = df['sales'].mean()
    income_mean = df['income'].mean()


    print(case_price_mean)
    print(age_mean)
    print(acres_mean)
    print(varieties_mean)
    print(domestic_mean)
    print(visitors_mean)
    print(buses_mean)
    print(employees_mean)
    print(awards_mean)
    print(sales_mean)
    print(income_mean)


    #---------- Mediana ----------#
    print("----- Mediana ------")
    case_price_median = df['case_price'].median()
    age_median = df['age'].median()
    acres_median = df['acres'].median()
    varieties_median = int(df['varieties'].median())
    domestic_median = int(df['domestic'].median())
    visitors_median = int(df['visitors'].median())
    buses_median = df['buses'].median()
    employees_median = int(df['employees'].median())
    awards_median = df['awards'].median()
    sales_median = df['sales'].median()
    income_median = df['income'].median()

    print(case_price_median)
    print(age_median)
    print(acres_median)
    print(varieties_median)
    print(domestic_median)
    print(visitors_median)
    print(buses_median)
    print(employees_median)
    print(awards_median)
    print(sales_median)
    print(income_median)

    #---------- Rango ----------#
    print("------Rango-----")
    case_price_range = df['case_price'].max() - df['case_price'].min()
    age_range = df['age'].max() - df['age'].min()
    acres_range = df['acres'].max() - df['acres'].min()
    varieties_range = int(df['varieties'].max() - df['varieties'].min())
    domestic_range = int(df['domestic'].max() - df['domestic'].min())
    visitors_range = int(df['visitors'].max() - df['visitors'].min())
    buses_range = df['buses'].max() - df['buses'].min()
    employees_range = int(df['employees'].max() - df['employees'].min())
    awards_range = df['awards'].max() - df['awards'].min()
    sales_range = df['sales'].max() - df['sales'].min()
    income_range = df['income'].max() - df['income'].min()

    print(case_price_range)
    print(age_range)
    print(acres_range)
    print(varieties_range)
    print(domestic_range)
    print(visitors_range)
    print(buses_range)
    print(employees_range)
    print(awards_range)
    print(sales_range)
    print(income_range)

    #---------- Desviacion tipica ----------#
    print("-----Desviacion tipica-----")

    case_price_std = df['case_price'].std()
    age_std = df['age'].std()
    acres_std = df['acres'].std()
    varieties_std = int(df['varieties'].std())
    domestic_std = int(df['domestic'].std())
    visitors_std = int(df['visitors'].std())
    buses_std = df['buses'].std()
    employees_std = int(df['employees'].std())
    awards_std = df['awards'].std()
    sales_std = df['sales'].std()
    income_std = df['income'].std()

    print(case_price_std)
    print(age_std)
    print(acres_std)
    print(varieties_std)
    print(domestic_std)
    print(visitors_std)
    print(buses_std)
    print(employees_std)
    print(awards_std)
    print(sales_std)
    print(income_std)

    #---------- Varianza ----------#
    print("-----Varianza-----")
    case_price_var = df['case_price'].var()
    age_var = df['age'].var()
    acres_var = df['acres'].var()
    varieties_var = int(df['varieties'].var())
    domestic_var = int(df['domestic'].var())
    visitors_var = int(df['visitors'].var())
    buses_var = df['buses'].var()
    employees_var = int(df['employees'].var())
    awards_var = df['awards'].var()
    sales_var = df['sales'].var()
    income_var = df['income'].var()

    print(case_price_var)
    print(age_var)
    print(acres_var)
    print(varieties_var)
    print(domestic_var)
    print(visitors_var)
    print(buses_var)
    print(employees_var)
    print(awards_var)
    print(sales_var)
    print(income_var)

    #---------- Graficos ----------#

    #--------- Histogramas ----------#
    # df.hist(column='case_price', bins=5)
    # df.hist(column='age', bins=5)
    # df.hist(column='acres', bins=5)
    # df.hist(column='varieties', bins=5)
    # df.hist(column='domestic', bins=2)
    # df.hist(column='visitors', bins=5)
    # df.hist(column='buses', bins=5)
    # df.hist(column='employees', bins=5)
    # df.hist(column='awards', bins=5)
    # df.hist(column='sales', bins=5)
    # df.hist(column='income', bins=5)

    #--------- Caja y bigotes ----------#
    df.boxplot(column='case_price')
    plt.xlabel('case_price')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of case prices')
    plt.show()

    df.boxplot(column='age')
    plt.xlabel('age')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of age')
    plt.show()

    df.boxplot(column='acres')
    plt.xlabel('acres')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of acres')
    plt.show()

    df.boxplot(column='varieties')
    plt.xlabel('varieties')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of varieties')
    plt.show()

    df.boxplot(column='buses')
    plt.xlabel('buses')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of buses')
    plt.show()

    df.boxplot(column='employees')
    plt.xlabel('employees')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of employees')
    plt.show()

    df.boxplot(column='awards')
    plt.xlabel('awards')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of awards')
    plt.show()

    df.boxplot(column='sales')
    plt.xlabel('sales')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of sales')
    plt.show()

    df.boxplot(column='income')
    plt.xlabel('income')
    plt.ylabel('Values')
    plt.title('Box-and-Whisker Plot of income')
    plt.show()

    #--------- Scatter ----------#
    df.plot.scatter(x='acres', y='employees')
    plt.show()
    df.plot.scatter(x='acres', y='income')
    plt.show()
    df.plot.scatter(x='case_price', y='awards')
    plt.show()
    df.plot.scatter(x='varieties', y='sales')
    plt.show()
    df.plot.scatter(x='age', y='awards')
    plt.show()
    df.plot.scatter(x='location', y='income')
    plt.show()
    df.plot.scatter(x='income', y='sales')
    plt.show()
    df.plot.scatter(x='employees', y='income')
    plt.show()

    #--------- Correlacion ----------#
    print("-----Correlacion-----")
    print(df.corr(numeric_only=True))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='Blues')