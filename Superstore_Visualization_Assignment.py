# Databricks notebook source
# MAGIC %md
# MAGIC ##Assignment 3: Data Visualization with Azure Databricks
# MAGIC
# MAGIC Sana Mehrab Beigi
# MAGIC
# MAGIC Data:[]([url](url)) Global Superstore Orders

# COMMAND ----------

# MAGIC %md
# MAGIC - Row ID: Integer
# MAGIC - Order ID: String
# MAGIC - Order Date: String
# MAGIC - Ship Date: String
# MAGIC - Ship Mode: String
# MAGIC - Customer ID: String
# MAGIC - Customer Name: String
# MAGIC - Segment: String
# MAGIC - City: String
# MAGIC - State: String
# MAGIC - Country: String
# MAGIC - Postal Code: Float
# MAGIC - Market: String
# MAGIC - Region: String
# MAGIC - Product ID: String
# MAGIC - Category: String
# MAGIC - Sub-Category: String
# MAGIC - Product Name: String
# MAGIC - Sales: Float
# MAGIC - Quantity: Integer
# MAGIC - Discount: Float
# MAGIC - Profit: Float
# MAGIC - Shipping Cost: Float
# MAGIC - Order Priority: String

# COMMAND ----------

# MAGIC %md
# MAGIC ##1. Data Preparation

# COMMAND ----------

# Import standard visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pyspark.sql.functions import to_date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import Databricks visualization utilities
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configure matplotlib for Databricks
%matplotlib inline

# Set plot styling
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid")

# COMMAND ----------

# MAGIC %md
# MAGIC 1.1 Loading data and converting date columns to proper timestamp format
# MAGIC
# MAGIC Since I uploaded the file to the Filestore and read it using `inferSchema=True`, Spark automatically detected and assigned appropriate data types. This also ensured that date columns were correctly interpreted as timestamp formats without requiring manual conversion. Additionally, I utilized the following options to handle special cases in the data:
# MAGIC
# MAGIC quote='"' → Treats text enclosed in double quotes ("") as a single value.
# MAGIC
# MAGIC escape='"' → Correctly processes embedded quotes (e.g., "" is interpreted as ").
# MAGIC
# MAGIC These configurations were necessary because some product names contained double quotes, which Spark initially misinterpreted as delimiters, incorrectly shifting values to the next column.

# COMMAND ----------

df = spark.read.csv("/FileStore/tables/GlobalSuperstoreOrders.csv", 
                    header=True, 
                    inferSchema=True, 
                    multiLine=True, 
                    quote='"', 
                    escape='"')


df = df.withColumn("Order Date", to_date("Order Date", "MM/dd/yyyy")) \
       .withColumn("Ship Date", to_date("Ship Date", "MM/dd/yyyy"))


display(df)
print(f"Number of records: {df.count()}\n")

print("Schema:")
df.printSchema()

# COMMAND ----------

num_rows = df.count()
num_cols = len(df.columns)

print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC 1.2 Handle missing values appropriately
# MAGIC
# MAGIC Since only U.S. locations in the dataset had postal codes, I chose to retain this valuable geographic information. To avoid losing these entries while maintaining dataset integrity, I replaced missing postal codes with 0 for non-U.S. records.

# COMMAND ----------

missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
display(missing_values)

# COMMAND ----------

df = df.fillna({'Postal Code': '0'})
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 1.3 Create derived columns that will be useful for your analysis (e.g., order year, month, day of week, profit margin, etc.)

# COMMAND ----------

from pyspark.sql.functions import year, month, dayofweek

#Order Year, Month, Day of Week
df = df.withColumn("Order_Year", year("Order Date")) \
       .withColumn("Order_Month", month("Order Date")) \
       .withColumn("Order_DayOfWeek", dayofweek("Order Date"))  # 1 = Sunday, 7 = Saturday


#Profit Margin: how profitable the orders are relative to sales.
from pyspark.sql.functions import col

df = df.withColumn("Profit_Margin", (col("Profit") / col("Sales")).cast("double"))

#Shipping Duration: Calculate how long it took to ship an order (in days)
from pyspark.sql.functions import datediff

df = df.withColumn("Shipping_Duration", datediff("Ship Date", "Order Date"))

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###2. Visualization Creation
# MAGIC
# MAGIC Converting to Pandas for Visualization
# MAGIC
# MAGIC For more complex visualizations, it's often easier to convert PySpark DataFrames to Pandas:

# COMMAND ----------

# Convert a sample to Pandas for visualization
pandas_df = df.toPandas()

# Check the resulting DataFrame
pandas_df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.A. Temporal Analysis 
# MAGIC
# MAGIC Create visualizations showing how sales and profits vary by:
# MAGIC
# MAGIC Month of year(built-in Databricks visualizations)
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import sum

sales_over_month = df.groupBy("Order_Month") \
                     .agg(sum("Sales").alias("Total_Sales"),
                          sum("Profit").alias("Total_Profit")) \
                     .orderBy("Order_Month")

display(sales_over_month)

# COMMAND ----------

# MAGIC %md
# MAGIC Year-over-year trends(seaborn)

# COMMAND ----------

plt.figure(figsize=(10, 5))
sns.lineplot(data=pandas_df, x="Order_Year", y="Sales", label="Sales")
sns.lineplot(data=pandas_df, x="Order_Year", y="Profit", label="Profit")
plt.title("Year-over-Year Trends")
plt.xlabel("Year")
plt.ylabel("Amount")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Day of week(seaborn)

# COMMAND ----------

plt.figure(figsize=(8, 5))
sns.boxplot(data=pandas_df, x="Order_DayOfWeek", y="Sales")
plt.title("Sales by Day of Week")
plt.xlabel("Day of Week (1=Sun, 7=Sat)")
plt.ylabel("Sales")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Create a heatmap showing monthly sales patterns across different years(seaborn)

# COMMAND ----------

sales_pivot = pandas_df.pivot_table(index="Order_Year", columns="Order_Month", values="Sales", aggfunc="sum")
plt.figure(figsize=(12, 6))
sns.heatmap(sales_pivot, cmap="YlGnBu", annot=True, fmt=".0f")
plt.title("Monthly Sales Patterns Across Years")
plt.xlabel("Month")
plt.ylabel("Year")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Analyze the relationship between order date and ship date (delivery time)(MatplotLib)

# COMMAND ----------

plt.figure(figsize=(8, 5))
sns.histplot(pandas_df["Shipping_Duration"], bins=20, kde=True)
plt.title("Shipping Duration Distribution")
plt.xlabel("Days to Deliver")
plt.ylabel("Number of Orders")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Visualize sales trends by different product categories over time(plotly)

# COMMAND ----------

fig = px.line(pandas_df, x="Order Date", y="Sales", color="Category", title="Sales by Category Over Time")
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC 2.B. Financial Analysis
# MAGIC
# MAGIC Visualize the distribution of sales, profits, and profit margins(Seaborn)

# COMMAND ----------

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
sns.histplot(pandas_df['Sales'], kde=True, bins=30)
plt.title("Sales Distribution")

plt.subplot(1, 3, 2)
sns.histplot(pandas_df['Profit'], kde=True, bins=30)
plt.title("Profit Distribution")

plt.subplot(1, 3, 3)
sns.histplot(pandas_df['Profit_Margin'], kde=True, bins=30)
plt.title("Profit Margin Distribution")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Relationship between Discount and Profit(Matplotlib)

# COMMAND ----------

plt.figure(figsize=(8, 5))
plt.scatter(pandas_df['Discount'], pandas_df['Profit'], alpha=0.5)
plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC How Shipping Cost Affects Profitability(Plotly)

# COMMAND ----------

fig = px.scatter(pandas_df, x="Shipping Cost", y="Profit", 
                 color="Category", 
                 title="Shipping Cost vs Profit by Category",
                 size="Sales", 
                 hover_data=['Order Priority'])
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Factors Correlating with Profit(Databricks Built-in Heatmap)

# COMMAND ----------

# Calculate correlations
correlation_df = pandas_df[['Sales', 'Profit', 'Profit_Margin', 'Discount', 'Shipping Cost', 'Quantity', 'Shipping_Duration']].corr()

# Factors Correlating with Profit
profit_correlation = correlation_df[['Profit']]

# Visualize the correlation with a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(profit_correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Factors Correlating with Profit")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Compare Order Priorities by Sales and Profit(Seaborn)

# COMMAND ----------

plt.figure(figsize=(10, 5))
sns.barplot(data=pandas_df, x="Order Priority", y="Sales", estimator=np.sum, errorbar=None, color='blue', label='Sales')
sns.barplot(data=pandas_df, x="Order Priority", y="Profit", estimator=np.sum, errorbar=None, color='orange', label='Profit')
plt.title("Total Sales and Profit by Order Priority")
plt.ylabel("Value")
plt.legend()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 2.C. Geographic Analysis`
# MAGIC
# MAGIC Sales and Profits by Country or Region - Plotly Choropleth (Best for mapping across countries)

# COMMAND ----------

country_summary = pandas_df.groupby("Country")[["Sales", "Profit"]].sum().reset_index()

fig = px.choropleth(country_summary, 
                    locations="Country", 
                    locationmode="country names", 
                    color="Sales",
                    hover_name="Country", 
                    color_continuous_scale="Viridis",
                    title="Global Sales by Country")
fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Most Profitable and Unprofitable Markets - Matplotlib Horizontal Bar Chart (Clear for ranked comparisons)

# COMMAND ----------

market_profit = pandas_df.groupby("Market")["Profit"].sum().sort_values()

plt.figure(figsize=(10, 6))
market_profit.plot(kind='barh', color=['red' if p < 0 else 'green' for p in market_profit])
plt.title("Total Profit by Market")
plt.xlabel("Profit")
plt.ylabel("Market")
plt.grid(True, axis='x')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Product Preferences by Region - Seaborn Heatmap (Best for dense comparisons)

# COMMAND ----------

product_region = pandas_df.groupby(["Region", "Sub-Category"])["Sales"].sum().unstack().fillna(0)

plt.figure(figsize=(14, 6))
sns.heatmap(product_region, cmap="Blues", annot=False, linewidths=0.5)
plt.title("Sales by Sub-Category and Region")
plt.xlabel("Product Sub-Category")
plt.ylabel("Region")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Shipping Costs Across Regions - Databricks Built-in Visualization (Simple, clean comparison chart)

# COMMAND ----------

region_shipping = pandas_df.groupby("Region")[["Shipping Cost"]].mean().reset_index()
display(region_shipping)

# COMMAND ----------

# MAGIC %md
# MAGIC D. Product and Customer Analysis Dashboard
# MAGIC
# MAGIC Create a comprehensive dashboard with at least 4 complementary visualizations that tell a cohesive story about product performance and customer segments
# MAGIC Include visualizations of:
# MAGIC
# MAGIC Sales and profits by product category and sub-category (Plotly Treemap (Great for hierarchical breakdown))
# MAGIC
# MAGIC Customer segment performance(Seaborn Bar Plot)
# MAGIC
# MAGIC Product profitability analysis(Databricks Built-in Table + Bar Chart)
# MAGIC
# MAGIC Customer buying patterns(Plotly Sunburst (Shows how different segments buy products))

# COMMAND ----------

category_data = pandas_df.groupby(["Category", "Sub-Category"])[["Sales", "Profit"]].sum().reset_index()

fig1 = px.treemap(category_data, 
                  path=["Category", "Sub-Category"], 
                  values="Sales", 
                  color="Profit",
                  color_continuous_scale='RdYlGn',
                  title="Sales and Profit by Category and Sub-Category")
fig1.show()


# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

segment_perf = pandas_df.groupby("Segment")[["Sales", "Profit"]].sum().reset_index()

plt.figure(figsize=(10, 5))
sns.barplot(data=segment_perf, x="Segment", y="Sales", color="skyblue", label="Sales")
sns.barplot(data=segment_perf, x="Segment", y="Profit", color="green", label="Profit")
plt.title("Customer Segment Performance")
plt.legend()
plt.xlabel("Segment")
plt.ylabel("Amount")
plt.show()


# COMMAND ----------

product_profit = pandas_df.groupby("Product Name")[["Sales", "Profit"]].sum().sort_values("Profit", ascending=False).reset_index()

# Top 10 most profitable
top_10_profitable = product_profit.head(10)
display(top_10_profitable)

# Also optionally:
# Bottom 10 least profitable
bottom_10_profitable = product_profit.tail(10)
display(bottom_10_profitable)


# COMMAND ----------

fig4 = px.sunburst(pandas_df, 
                   path=["Segment", "Category", "Sub-Category"], 
                   values="Sales",
                   title="Customer Buying Patterns by Segment and Product Hierarchy")
fig4.show()


# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC show them in one dashboard

# COMMAND ----------

sunburst_data = pandas_df.groupby(["Category", "Sub-Category", "Segment", "Product Name"]).agg({
    "Sales": "sum",
    "Profit": "sum"
}).reset_index()

# Sunburst by Sales (you can switch to Profit too)
fig = px.sunburst(
    sunburst_data,
    path=["Category", "Sub-Category", "Segment", "Product Name"],
    values="Sales",
    color="Profit",
    color_continuous_scale="Viridis",
    title="🧩 Sales, Profit, and Customer Performance Drill-Down"
)

fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
fig.show()


# COMMAND ----------

