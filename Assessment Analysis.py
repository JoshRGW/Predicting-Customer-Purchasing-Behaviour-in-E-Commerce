#!/usr/bin/env python
# coding: utf-8

# # 1 Data Preparation

# In[ ]:


# As an initial step, I import the necessary modules that will be utilized in this notebook:


# In[141]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# After loading the data, I provide basic information about the dataframe,
# including variable types, the count and percentage of null values relative to the total number of entries.


# In[4]:


# Specify the file path
file_path = 'International E-Commerce Dataset.csv'

# Read the datafile
ecommerce_data = pd.read_csv(file_path, encoding="ISO-8859-1", dtype={'CustomerID': str, 'invoice': str})
print('Dataframe dimensions:', ecommerce_data.shape)


# In[5]:


# Convert the 'invoiceDate' column to datetime format
ecommerce_data['invoiceDate'] = pd.to_datetime(ecommerce_data['invoiceDate'])


# In[6]:


# Create a summary of column information including data types and number of null values.
column_info_summary = pd.DataFrame(ecommerce_data.dtypes).T.rename(index={0:'column type'})

# Create DataFrame for null values counts.
null_values_counts = pd.DataFrame(ecommerce_data.isnull().sum()).T.rename(index={0:'null values (no)'})

# Concatenate null values counts DataFrame with column_info_summary DataFrame.
column_info_summary = pd.concat([column_info_summary, null_values_counts], axis=0)

# Create DataFrame for null values percentages.
null_values_percentages = pd.DataFrame(ecommerce_data.isnull().sum() / 
                                       ecommerce_data.shape[0] * 100).T.rename(index={0:'null values (%)'})

# Concatenate null values percentages DataFrame with column_info_summary DataFrame.
column_info_summary = pd.concat([column_info_summary, null_values_percentages], axis=0)

# Display the summary of column information.
display(column_info_summary)

# Display the first 5 rows of the DataFrame.
display(ecommerce_data.head())


# In[7]:


# When examining the null values within the dataframe, it's notable that approximately 25% of the entries lack an assigned customer.
# Given the unavailability of data to impute values for these users,these entries are deemed irrelevant for the current analysis.
# Consequently, they are removed from the dataframe.


# In[8]:


# Drop rows with null values in the 'CustomerID' column
ecommerce_data.dropna(axis=0, subset=['CustomerID'], inplace=True)

# Display the dimensions of the dataframe after dropping null values
print('Dataframe dimensions:', ecommerce_data.shape)

# Create a summary of column information: data types and null values
column_info_summary = pd.DataFrame(ecommerce_data.dtypes).T.rename(index={0: 'column type'})
null_values_counts = pd.DataFrame(ecommerce_data.isnull().sum()).T.rename(index={0: 'null values (no)'})
null_values_percentages = pd.DataFrame(ecommerce_data.isnull().sum() / 
                                       len(ecommerce_data) * 100).T.rename(index={0: 'null values (%)'})

# Concatenate the column_info_summary DataFrame with null_values_counts and null_values_percentages DataFrames
column_info_summary = pd.concat([column_info_summary, null_values_counts, null_values_percentages], axis=0)

# Display the summary of column information
display(column_info_summary)


# In[9]:


# By removing these entries, we achieve a dataframe with complete data for all variables.
# Subsequently, I perform a check for duplicate entries and eliminate them.


# In[10]:


# Print the number of duplicate entries in the DataFrame and drop them

print('Duplicate Entries: {}'.format(ecommerce_data.duplicated().sum()))
ecommerce_data.drop_duplicates(inplace=True)


# # 2 Exploring the content of variables

# In[ ]:


# This dataframe contains 8 variables:
# 1. invoice: A 6-digit integral number uniquely assigned to each transaction. If it starts with 'c', it's a cancellation.
# 2. stockID: A 5-digit integral number uniquely assigned to each distinct product.
# 3. itemDescription: The name of the product.
# 4. quantity: The quantity of each product per transaction.
# 5. invoiceDate: The date and time of each transaction.
# 6. unitPrice: The price per unit in sterling.
# 7. CustomerID: A 5-digit integral number uniquely assigned to each customer.
# 8. residence: The country where each customer resides.


# In[ ]:


# 2.1 residence


# In[12]:


# Let's inspect the countries of residence associated with the orders placed.

# Count the occurrences of unique combinations of CustomerID, invoice, and residence
temp_df = ecommerce_data[['CustomerID', 'invoice', 'residence']].groupby(['CustomerID', 'invoice', 'residence']).count()

# Reset the index of the temporary DataFrame
temp_df = temp_df.reset_index(drop=False)

# Count the occurrences of each country in the residence column
countries_count = temp_df['residence'].value_counts()

# Print the number of unique countries found in the DataFrame
print('Number of unique countries in the dataframe: {}'.format(len(countries_count)))


# In[14]:


# Visualizing the results using a chloropleth map to provide a geographic representation of the data.

# Creating a choropleth map with the Robinson projection and various colorscale
data = dict(type = 'choropleth',
    locations = countries_count.index,  # Locations representing countries
    locationmode = 'country names',  # Using country names for location mode
    z = countries_count,  # Using country counts as values to be mapped
    text = countries_count.index,  # Text to display on hover (country names)
    colorbar = {'title': 'Number of Orders'},  # Title for color bar
    colorscale = [[0, 'rgb(255, 255, 255)'],    # White
        [0.01, 'rgb(0, 0, 0)'],       # Black
        [0.02, 'rgb(255, 0, 0)'],     # Red
        [0.03, 'rgb(0, 255, 0)'],     # Green
        [0.05, 'rgb(0, 0, 255)'],     # Blue
        [0.10, 'rgb(255, 255, 0)'],   # Yellow
        [0.20, 'rgb(255, 0, 255)'],   # Magenta
        [1, 'rgb(0, 255, 255)']],     # Cyan
    reversescale = False)  # Reversing the color scale

layout = dict(title='Number of Orders by Country',  # Title of the plot
geo=dict(showframe=True,  # Displaying frame around the map
projection={'type': 'robinson'}))  # Robinson projection
    
# Creating the choropleth map figure
choromap = go.Figure(data=[data], layout=layout)

# Displaying the choropleth map
iplot(choromap, validate=False)


# In[15]:


# The choropleth map reveals a significant dominance of orders originating from the UK within the dataset.


# # 2.2 Customers and Products Analysis.

# In[ ]:


# With a dataframe containing 400,000 entries, let's explore the number of unique users and products in the dataset.


# In[17]:


# Count the number of products based on stockID, transactions based on invoice, and customers based on CustomerID
pd.DataFrame([{'products': len(ecommerce_data['stockID'].value_counts()),    
               'transactions': len(ecommerce_data['invoice'].value_counts()),
               'customers': len(ecommerce_data['CustomerID'].value_counts()),  
              }], columns = ['products', 'transactions', 'customers'], index = ['quantity'])


# In[18]:


# Analysis of customers and Product Statistics:

# The dataset comprises 4,372 unique users who purchased 3,684 distinct products. 
# In total, there were 22,188 transactions recorded.


# In[19]:


# Calculating the Number of Products Purchased per Transaction:

# Group the data by 'CustomerID' and 'invoice' and count the occurrences of each group
temp_df = ecommerce_data.groupby(by = ['CustomerID', 'invoice'], as_index=False)['invoiceDate'].count()

# Rename the column to indicate the number of products per transaction
product_count_per_basket = temp_df.rename(columns={'invoiceDate': 'Number of products'})

# Display the first five rows of the DataFrame, sorted by 'CustomerID'
product_count_per_basket[:5].sort_values('CustomerID')


# In[20]:


# Observations from the first lines of the list:

# - Presence of entries with the 'C' prefix in the invoiceID variable, indicating canceled transactions.
# - Existence of single-purchase users, such as customer number 12346.
# - Presence of frequent users who make large orders consistently.


# # 2.2.1 Cancelled Orders

# In[21]:


# Firstly, I determine the count of transactions corresponding to cancelled orders:

# Create a new column indicating if the order is cancelled
product_count_per_basket['cancelled orders'] = product_count_per_basket['invoice'].apply(lambda x: int('C' in x))

# Display the first five rows of the updated DataFrame
display(product_count_per_basket[:5])

# Calculate the total number and percentage of cancelled orders
total_cancelled_orders = product_count_per_basket['cancelled orders'].sum()
total_orders = product_count_per_basket.shape[0]
print('Number of Cancelled Orders: {}/{} ({:.2f}%)'.format(total_cancelled_orders, 
                                                           total_orders, 
                                                           total_cancelled_orders / total_orders * 100))


# In[22]:


# We observe a significant number of cancellations, accounting for approximately 16% of all transactions. 
# Let's examine the initial entries of the dataframe to gain more insight:

# Sort out the `ecommerce_data` DataFrame by the 'CustomerID' column in ascending order
# Display the first 5 rows of the sorted DataFrame.
display(ecommerce_data.sort_values('CustomerID')[:5])


# In[23]:


# Upon inspecting these few lines, it becomes evident that for cancelled orders, corresponding entries exist in the dataframe,
# often mirroring the original transactions with differences primarily in the "quantity" and "invoiceDate" variables.

# To verify if this pattern persists across all entries, I intend to locate transactions indicating negative quantities. 
# Subsequently, I will ascertain whether there consistently exists an order reflecting the same quantity (albeit positive), 
# accompanied by identical details such as CustomerID, itemDescription, and unitPrice.


# In[24]:


# Create a DataFrame containing transactions with negative quantities, indicating cancellations, and select relevant columns
df_check = ecommerce_data[ecommerce_data['quantity'] < 0][['CustomerID', 'quantity', 'stockID', 
                                                           'itemDescription', 'unitPrice']]

# Iterate over each row in the DataFrame containing cancellations
for index, col in df_check.iterrows():
    
# Check if there exists a corresponding transaction meeting specific criteria
    if ecommerce_data[(ecommerce_data['CustomerID'] == col[0]) & 
                      (ecommerce_data['quantity'] == -col[1]) & 
                      (ecommerce_data['itemDescription'] == col[2])].shape[0] == 0:
        
# If no corresponding transaction is found, print the details of the transaction with negative quantity
        print(df_check.loc[index])
        
# Print a visual indicator that the hypothesis is not fulfilled
        print(15 * '-' + '>' + ' HYPOTHESIS NOT FULFILLED')
        
# Break out of the loop since the hypothesis is not fulfilled
        break


# In[25]:


# The initial hypothesis is disproved due to the presence of a 'Discount' entry.
# I'll reevaluate the hypothesis, this time excluding the 'Discount' entries:

df_check = ecommerce_data[(ecommerce_data['quantity'] < 0) & (ecommerce_data['itemDescription'] != 'Discount')][
                                 ['CustomerID','quantity','stockID',
                                  'itemDescription','unitPrice']]

for index, col in  df_check.iterrows():
    if ecommerce_data[(ecommerce_data['CustomerID'] == col[0]) &
                      (ecommerce_data['quantity'] == -col[1]) & 
                      (ecommerce_data['itemDescription'] == col[2])].shape[0] == 0: 
        print(index, df_check.loc[index])
        print(15 * '-' + '>' + ' HYPOTHESIS NOT FULFILLED')
        break


# In[26]:


# Once again, the initial hypothesis is not confirmed. Cancellations do not consistently correlate with prior orders.

# I proceed to create a new variable in the dataframe indicating whether a portion of the order has been cancelled.
# Some cancellations without counterparts may be due to orders placed before December 2022 (the database entry point).
# Below, I conduct a survey of cancel orders and verify the presence of counterparts:


# In[27]:


# Create a deep copy of the original dataframe.
ecommerce_data_cleaned = ecommerce_data.copy(deep=True)

# Initialize a new column to track the number of cancelled orders.
ecommerce_data_cleaned['cancelled_orders_count'] = 0

# Initialize empty lists to store indices of entries to remove and entries with doubtful cancellations.
entries_to_remove = []
doubtful_entries = []

# Iterate over each row in the dataframe.
for index, col in ecommerce_data.iterrows():
    
    # Skip rows where the quantity is greater than zero or the item description is 'Discount'
    if (col['quantity'] > 0) or (col['itemDescription'] == 'Discount'):
        continue
    
    # Filter rows to find previous orders matching the current cancelled item.
    previous_orders = ecommerce_data[
        (ecommerce_data['CustomerID'] == col['CustomerID']) &
        (ecommerce_data['stockID'] == col['stockID']) &
        (ecommerce_data['invoiceDate'] < col['invoiceDate']) &
        (ecommerce_data['quantity'] > 0)
    ].copy()
    
    # Handle different scenarios based on the number of matching previous orders.
    if previous_orders.shape[0] == 0:
        # Add the current entry to 'doubtful_entries' if no previous orders are found.
        doubtful_entries.append(index)
    elif previous_orders.shape[0] == 1:
        # Update the 'cancelled_orders_count' in 'ecommerce_data_cleaned'.
        # Add the current entry to 'entries_to_remove if only one previous order is found.
        matching_order_index = previous_orders.index[0]
        ecommerce_data_cleaned.loc[matching_order_index, 'cancelled_orders_count'] = -col['quantity']
        entries_to_remove.append(index)
    elif previous_orders.shape[0] > 1:
        # Sort previous orders in descending order and select the first one that can cover the cancelled quantity
        previous_orders.sort_index(axis=0, ascending=False, inplace=True)
        for ind, val in previous_orders.iterrows():
            if val['quantity'] < -col['quantity']:
                ecommerce_data_cleaned.loc[ind, 'cancelled_orders_count'] = -col['quantity']
                entries_to_remove.append(index)

print("entries_to_remove: {}".format(len(entries_to_remove)))
print("doubtful_entries: {}".format(len(doubtful_entries)))


# In[28]:


# The indices of the corresponding cancel orders are respectively stored in the doubtful_entries 
# and entries_to_remove lists. The sizes of these lists are: 5748 To be removed and 1226 Doubtful

# In the above function, I handled two cases:

# 1. When a cancel order exists without a counterpart.
# 2. When there's at least one counterpart with the exact same quantity.


# In[29]:


# Among these entries, the lines listed in the doubtful_entries list correspond to entries indicating a cancellation
# without a preceding order. These entries represent approximately 1.4% and 0.2% of the total dataframe entries, respectively.

# Now, let's Filter out indices that do not exist in the dataframe 
# And check the number of entries corresponding to cancellations that haven't been deleted with the previous filter.


# In[30]:


# Filter out indices that do not exist in the dataframe
valid_entries_to_remove = [idx for idx in entries_to_remove if idx in ecommerce_data_cleaned.index]

# Drop entries marked for deletion from the dataframe
ecommerce_data_cleaned.drop(valid_entries_to_remove, axis=0, inplace=True)

# Filter remaining entries with negative quantity and exclude entries with stock code 'D'
remaining_entries = ecommerce_data_cleaned[(ecommerce_data_cleaned['quantity'] < 0) & 
                                           (ecommerce_data_cleaned['stockID'] != 'D')]

# Print the number of remaining entries
print("Number of remaining entries after deletion: {}".format(remaining_entries.shape[0]))


# In[31]:


# Display the first five remaining entries
remaining_entries[:5]


# In[32]:


# If we examine the purchases made by the customer associated with one of the entries listed above,
# specifically those related to the same product as the cancellation, several observations become apparent.


# In[33]:


# Filter the DataFrame to select rows where the 'CustomerID' is equal to 14409 
# and the 'stockID' is equal to '22834'

ecommerce_data_cleaned[(ecommerce_data_cleaned['CustomerID'] == 14409) & (ecommerce_data_cleaned['stockID'] == '22834')]


# # 2.2.2 StockID

# In[34]:


# Above we observed that the quantity canceled exceeds the total quantity purchased previously.


# In[35]:


# Check for special transaction indicators in the stockID variable, such as 'D' for Discount.
# Find the unique set of codes consisting only of letters:

list_special_codes = ecommerce_data_cleaned[ecommerce_data_cleaned['stockID'].str.contains('^[a-zA-Z]+',
                                                                                           regex=True)]['stockID'].unique()
list_special_codes


# In[36]:


# Iterate over each special product code in the list of special codes.
for code in list_special_codes:
    # Print the product code and its corresponding item description.
    # Use string formatting to align the output for better readability.
    print("{:<15} -> {:<30}".format(code, ecommerce_data_cleaned[ecommerce_data_cleaned['stockID']
                                                                 == code]['itemDescription'].unique()[0]))


# In[37]:


# Upon inspection, it's evident that there are various types of special transactions, 
# such as those related to postage or bank charges.


# # 2.2.3 Cart Price

# In[38]:


# Here, I'm creating a new variable to represent the total price of each purchase:

ecommerce_data_cleaned['TotalPrice'] = ecommerce_data_cleaned['unitPrice'] * (ecommerce_data_cleaned['quantity']
                                                                              - ecommerce_data_cleaned['cancelled_orders_count'])
ecommerce_data_cleaned.sort_values('CustomerID')[:5]


# In[39]:


# Each row in the dataframe represents a single product purchase, potentially part of a larger order.
# To calculate the total order price, we need to aggregate all purchases made during a single order.

# Calculate the sum of purchases for each customer and order
temp_df = ecommerce_data_cleaned.groupby(by=['CustomerID', 'invoice'], as_index=False)['TotalPrice'].sum()
order_totals = temp_df.rename(columns={'TotalPrice': 'Cart Price'})

# Convert invoiceDate to integer for calculation
ecommerce_data_cleaned['invoice_date_int'] = ecommerce_data_cleaned['invoiceDate'].astype('int64')

# Calculate the average invoice date per customer and order
temp_df = ecommerce_data_cleaned.groupby(by=['CustomerID', 'invoice'], as_index=False)['invoice_date_int'].mean()
ecommerce_data_cleaned.drop('invoice_date_int', axis=1, inplace=True)

# Convert invoiceDate back to datetime format
order_totals.loc[:, 'invoiceDate'] = pd.to_datetime(temp_df['invoice_date_int'])

# Select entries with positive Cart Price (total price of purchases)
order_totals = order_totals[order_totals['Cart Price'] > 0]

# Sort the resulting dataframe by CustomerID and display the first 6 rows
order_totals.sort_values('CustomerID')[:6]


# In[40]:


# Determine the distribution of purchases based on total prices to gain insight into the types of orders in the dataset.


# In[41]:


# Counting purchases
price_range = [0, 50, 100, 200, 500, 1000, 5000, 50000]
count_price = []

# Iterate over price ranges and count the number of purchases falling within each range
for i, price in enumerate(price_range):
    if i == 0: continue
    val = order_totals[(order_totals['Cart Price'] < price) &
                       (order_totals['Cart Price'] > price_range[i-1])]['Cart Price'].count()
    count_price.append(val)

# Visualizing the number of purchases by amount
plt.rc('font', weight='bold')
f, ax = plt.subplots(figsize=(11, 6))
colors = ['#FF5733', '#33FF57', '#3388FF', '#9B33FF', '#EE82EE', '#4169E1', '#B22222']
labels = [ '{}<.<{}'.format(price_range[i-1], s) for i,s in enumerate(price_range) if i != 0]
sizes  = count_price
explode = [0.0 if sizes[i] < 100 else 0.0 for i in range(len(sizes))]
ax.pie(sizes, explode = explode, labels=labels, colors = colors,
       autopct = lambda x:'{:1.0f}%'.format(x) if x > 1 else '',
       shadow = False, startangle=0)
ax.axis('equal')
f.text(0.5, 1.01, "Distribution of Order Prices", ha='center', fontsize = 18);


# In[42]:


# It's evident that the majority of orders consist of relatively high-value purchases,
# with approximately 65% of purchases exceeding £200 in value.


# # 3 Insights on Product Categories

# In[43]:


# Each product in the dataframe is uniquely identified by the stockID variable,
# while a brief description of the product is provided in the itemDescription variable. 
# I aim to utilize the information from the itemDescription variable to categorize the products into distinct categories.


# # 3.1 Product Descriptions

# In[44]:


# In the initial phase, I extract relevant information from the itemDescription variable using the following function:


# In[45]:


# Define a function to determine if a part-of-speech tag represents a noun
is_noun = lambda pos: pos[:2] == 'NN'

def keywords_inventory(dataframe, column='itemDescription'):
    # Initialize a stemmer for word stemming
    stemmer = nltk.stem.SnowballStemmer("english")
    # Initialize dictionaries to store word roots, selected keywords, and their counts
    keywords_roots = dict()  # Collect the words/roots
    keywords_select = dict()  # Association: root <-> keyword
    category_keys = []  # List to store selected keywords
    count_keywords = dict()  # Dictionary to count the occurrences of each keyword root
    icount = 0
    
    # Iterate over each entry in the specified column of the dataframe
    for s in dataframe[column]:
        # Skip null entries
        if pd.isnull(s):
            continue
        # Convert the text to lowercase
        lines = s.lower()
        # Tokenize the text
        tokenized = nltk.word_tokenize(lines)
        # Extract nouns from the tokenized text
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]
        
        # Process each noun extracted
        for t in nouns:
            t = t.lower()  # Convert the noun to lowercase
            root = stemmer.stem(t)  # Stem the noun to its root
            # Update dictionaries with the root and its corresponding keyword
            if root in keywords_roots:
                keywords_roots[root].add(t)
                count_keywords[root] += 1
            else:
                keywords_roots[root] = {t}
                count_keywords[root] = 1
    
    # Process each root in the keywords_roots dictionary
    for s in keywords_roots.keys():
        # If multiple keywords share the same root
        if len(keywords_roots[s]) > 1:  
            min_lenght = 1000
            # Find the shortest keyword and use it as the representative keyword for the root
            for k in keywords_roots[s]:
                if len(k) < min_lenght:
                    key = k
                    min_lenght = len(k)
            category_keys.append(key)  # Add the representative keyword to the list of selected keywords
            keywords_select[s] = key  # Associate the root with the representative keyword
        else:
            # If only one keyword corresponds to the root, add it directly to the selected keywords list
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]
                   
    # Print the number of unique keywords found in the specified column
    print("Number of keywords in variable '{}': {}".format(column, len(category_keys)))
    # Return the selected keywords, their roots, associations, and counts
    return category_keys, keywords_roots, keywords_select, count_keywords


# In[46]:


# This function processes a DataFrame by analyzing the content of the itemDescription column. It performs the following tasks:

# - Extracts names (proper, common) from product descriptions
# - Derives the root of each name and aggregates the set of names associated with this root
# - Counts the occurrences of each root in the DataFrame
# - Selects the shortest name as the keyword for each root (prioritizing singular forms)

# The initial step involves retrieving the list of products.


# In[47]:


# Create a DataFrame containing unique values from the 'itemDescription' column of the initial DataFrame
# Rename the column to 'itemDescription'

unique_desc_df = pd.DataFrame(ecommerce_data_cleaned['itemDescription'].unique()).rename(columns={0: 'itemDescription'})


# In[48]:


import nltk
nltk.download('punkt')


# In[49]:


import nltk
nltk.download('averaged_perceptron_tagger')


# In[50]:


# With the list of unique product descriptions created,
# I now apply the previously defined function to analyze these descriptions:

keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(unique_desc_df)


# In[51]:


# This function returns three variables:

# - keywords: a list of extracted keywords
# - keywords_roots: a dictionary where keys are the keyword roots and values are lists of words associated with those roots
# - count_keywords: a dictionary listing the number of times each word is used

# To sort the keywords based on their occurrences, the count_keywords dictionary is converted into a list.


# In[52]:


# Create an empty list to store the products along with their counts
product_list = []

# Iterate over the items (keyword, count) in the count_keywords dictionary
for k, v in count_keywords.items():
    # Append a list containing the keyword and its count to product_list
    product_list.append([keywords_select[k], v])

# Sort the product_list based on the count of occurrences of each keyword
# Sorting is done in descending order of counts
product_list.sort(key=lambda x: x[1], reverse=True)


# In[53]:


# Sort the list of products based on their occurrences in descending order
sorted_list = sorted(product_list, key=lambda x: x[1], reverse=True)

# Set up the plot with a smaller size and fewer words
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(10, 15))  # Adjust width and height as needed

# Extract the frequencies and corresponding words for the top 50 most common words
y_axis = [i[1] for i in sorted_list[:50]]
x_axis = [k for k, i in enumerate(sorted_list[:50])]
x_label = [i[0] for i in sorted_list[:50]]

# Set labels and font sizes
plt.xticks(fontsize=15)
plt.yticks(fontsize=13)
plt.yticks(x_axis, x_label)
plt.xlabel('Frequency', fontsize=18, labelpad=10)

# Create the horizontal bar chart
ax.barh(x_axis, y_axis, align='center')
ax = plt.gca()
ax.invert_yaxis()

# Set title with a black background and white text
plt.title('Word Frequency', bbox={'facecolor': 'k', 'pad': 5}, color='w', fontsize=25)

# Display the plot
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


# # 3.2 Defining Product Categories

# In[54]:


# The list obtained contains over 1400 keywords, with the most frequent ones appearing in over 200 products.
# However, upon examining the content, I noticed that some names are useless, 
# while others don't carry valuable information.
# Therefore, I discard these words from the analysis that follows,
# I also decide to consider only the words that appear more than 13 times.


# In[55]:


product_list = []

# Iterate over each key-value pair in the count_keywords dictionary
for k, v in count_keywords.items():
    # Retrieve the corresponding word using the keywords_select dictionary
    word = keywords_select[k]
    
    # Check if the word is in a predefined list of words to be excluded (such as colors)
    if word in ['pink', 'blue', 'tag', 'green', 'orange']:
        continue
    
    # Check if the length of the word is less than 3 characters or if its frequency of occurrence (v) is less than 13
    if len(word) < 3 or v < 13:
        continue
    
    # Check if the word contains certain characters like '+' or '/', and if so, skip the word
    if ('+' in word) or ('/' in word):
        continue
    
    # If the word passes all the conditions, add it to the product_list along with its frequency of occurrence
    product_list.append([word, v])

# Sort the product_list in descending order based on the frequency of occurrence
product_list.sort(key=lambda x: x[1], reverse=True)

# Print the number of words retained in the list, labeled as 'retained words'
print('Retained words:', len(product_list))


# # 3.2.1 Data Encoding

# In[56]:


# Now, I'll utilize these keywords to create product groups. 
# First, I define the X matrix
# Where the a_i,j coefficient is 1 if the description of the product i contains the word j, and 0 otherwise.


# In[57]:


# Extracting unique product descriptions
product_descriptions = ecommerce_data_cleaned['itemDescription'].unique()

# Creating an empty DataFrame to represent the presence of keywords in product descriptions
X = pd.DataFrame()

# Looping through the product_list to populate the DataFrame
for keyword, occurrence in product_list:
    # Mapping whether each product description contains the current keyword
    X[keyword] = list(map(lambda x: int(keyword.upper() in x), product_descriptions))


# In[58]:


# The X matrix represents the presence of keywords in product descriptions using one-hot encoding.
# Additionally, to create more balanced groups, I incorporate the price range of the products.
# Thus, I add 6 extra columns to the matrix to indicate the price range of the products.


# In[59]:


# Define the price threshold ranges
price_thresholds = [0, 1, 2, 3, 5, 10]

# Initialize a list to store column labels
price_label_columns = []

# Create column labels based on the price threshold ranges
for i in range(len(price_thresholds)):
    if i == len(price_thresholds) - 1:
        column_label = '.>{}'.format(price_thresholds[i])
    else:
        column_label = '{}<.<{}'.format(price_thresholds[i], price_thresholds[i + 1])
    price_label_columns.append(column_label)
    X.loc[:, column_label] = 0

# Iterate over product descriptions to determine their average price and assign labels based on price ranges
for i, product in enumerate(product_descriptions):
    average_price = ecommerce_data_cleaned[ecommerce_data_cleaned['itemDescription'] == product]['unitPrice'].mean()
    j = 0
    while average_price > price_thresholds[j]:
        j += 1
        if j == len(price_thresholds):
            break
    X.loc[i, price_label_columns[j - 1]] = 1


# In[60]:


# To determine suitable price ranges, I examine the distribution of products across different price groups:

# Print the header for the price range analysis table
print("{:<8} {:<20} \n".format('Range', 'Number of Products') + 20*'-')

# Iterate over each price threshold to count the number of products falling into each range
for i in range(len(price_thresholds)):
    if i == len(price_thresholds)-1:
        column_label = '.>{}'.format(price_thresholds[i])
    else:
        column_label = '{}<.<{}'.format(price_thresholds[i],price_thresholds[i+1])    

    # Print the count of products in each price range
    print("{:<10}  {:<20}".format(column_label, X.loc[:, column_label].sum()))


# # 3.2.2 Creating Clusters for Products

# In[61]:


# In this section, I will group the products into different classes. 
# In the case of matrices with binary encoding, the most suitable metric for the calculation of distances is the Hamming's metric. 
# Note that the kmeans method of sklearn uses a Euclidean distance that can be used, but it is not the best choice in the case of categorical variables. 
# However, in order to use the Hamming's metric, we need to use the kmodes package which is not available on the current platform. 
# Hence, I use the kmeans method even if this is not the best choice.

# In order to define (approximately) the number of clusters that best represents the data, I use the silhouette score:


# In[62]:


# Convert the dataframe X into a numpy array
matrix = X.to_numpy()

# Iterate through different numbers of clusters
for n_clusters in range(3, 10):
    # Initialize KMeans Model with k-means++ initialization
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
    # Fit KMeans to the data
    kmeans.fit(matrix)
    # Predict the cluster labels for each data point
    cluster_labels = kmeans.predict(matrix)
    # Calculate the average silhouette score
    avg_silhouette = silhouette_score(matrix, cluster_labels)
    # Print the silhouette score for the current number of clusters
    print("For n_clusters =", n_clusters, "Average silhouette score:", avg_silhouette)


# In[63]:


# In practice, the scores obtained above can be considered equivalent since, depending on the run,
# scores of 0.1±0.05 will be obtained for all clusters with n_clusters > 3
# (we obtain slightly lower scores for the first cluster).
# On the other hand, I found that beyond 5 clusters, some clusters contained very few elements.
# I therefore choose to separate the dataset into 5 clusters.
# In order to ensure a good classification at every run of the notebook,
# I iterate until we obtain the best possible silhouette score, which is, in the present case, around 0.15:


# In[64]:


# Set the initial number of clusters to 5 and the initial average silhouette score to -1
n_clusters = 5
avg_silhouette = -1

# Iterate until the average silhouette score is greater than or equal to 0.145
while avg_silhouette < 0.145:
    # Initialize KMeans model with k-means++ initialization
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    # Fit KMeans to the data
    kmeans.fit(matrix)
    # Predict the cluster labels for each data point
    cluster_labels = kmeans.predict(matrix)
    # Calculate the average silhouette score
    avg_silhouette = silhouette_score(matrix, cluster_labels)
    
    # Print the silhouette score for the current number of clusters
    print("For n_clusters =", n_clusters, "Average silhouette score:", avg_silhouette)


# # 3.2.3 Characterising the content of clusters

# In[66]:


# Check the number of elements in every class
pd.Series(cluster_labels).value_counts()


# # (a) Silhouette Intra-cluster Score

# In[67]:


# To assess the quality of the classification, 
# we visualize the silhouette scores of each element in the different clusters. 
# This is depicted in the following figure, sourced from the sklearn documentation.


# In[68]:


def plot_silhouette(n_clusters, x_limit, matrix_size, silhouette_values, cluster_labels):
    plt.rcParams["patch.force_edgecolor"] = True
    plt.style.use('fivethirtyeight')
    mpl.rc('patch', edgecolor='dimgray', linewidth=1)
    # Create the plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    ax.set_xlim([x_limit[0], x_limit[1]])
    ax.set_ylim([0, matrix_size + (n_clusters + 1) * 10])
    y_lower = 10
    for i in range(n_clusters):
        # Get silhouette values for the current cluster
        cluster_silhouette_values = silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        cluster_size = cluster_silhouette_values.shape[0]
        y_upper = y_lower + cluster_size
        cmap = cm.get_cmap("Spectral")
        color = cmap(float(i) / n_clusters)        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_values,
                           facecolor=color, edgecolor=color, alpha=0.8)
        # Label clusters
        ax.text(-0.03, y_lower + 0.5 * cluster_size, str(i), color='red', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round, pad=0.3'))
        # Update y_lower for the next plot
        y_lower = y_upper + 10  


# In[69]:


# Calculate individual silhouette scores (silhouette_values) for each data point
silhouette_values = silhouette_samples(matrix, cluster_labels)

# Plot the silhouette scores for each data point in the clusters
plot_silhouette(n_clusters, [-0.07, 0.33], len(X), silhouette_values, cluster_labels)


# # (b) Word Cloud

# In[70]:


# Now we can visualize the type of objects represented by each cluster. 
# To gain a comprehensive overview of their contents, we identify the most frequent keywords in each cluster.


# In[71]:


# Extracting product descriptions and keywords
sorted_list = pd.DataFrame(product_descriptions)
word_lists = [word for (word, cluster_frequency) in product_list]

# Initialize list of dictionaries to store occurrences of keywords in each cluster
cluster_frequency = [dict() for _ in range(n_clusters)]

# Iterate through each cluster
for i in range(n_clusters):
    # Extract products belonging to the current cluster
    cluster_lists = sorted_list.loc[cluster_labels == i]
    # Iterate through each keyword
    for word in word_lists:
        # Skip certain irrelevant keywords
        if word in ['art', 'set', 'heart', 'pink', 'blue', 'tag']:
            continue
        # Calculate occurrence of keyword in the current cluster
        cluster_frequency[i][word] = sum(cluster_lists.loc[:, 0].str.contains(word.upper()))


# In[72]:


# Outputting the result as word clouds
    
# Define a function to generate a random color for the word cloud
def random_color_func(word = None, font_size = None, position = None,
                      orientation = None, font_path = None, random_state = None):
    h = int(360.0 * tone / 255.0)  # Calculate hue value for color
    s = int(100.0 * 255.0 / 255.0)  # Calculate saturation value for color
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)  # Calculate lightness value for color
    return "hsl({}, {}%, {}%)".format(h, s, l)

# Define a function to create a word cloud
def make_wordcloud(sorted_list, increment):
    ax = fig.add_subplot(4, 2, increment)  # Add subplot to the figure
    words = dict()  # Initialize dictionary to store word frequencies
    trunc_occurrences = sorted_list[0:150]  # Truncate the word list to 150 words
    for s in trunc_occurrences:
        words[s[0]] = s[1]  # Store word and its frequency in the dictionary
    # Create the word cloud with specified parameters
    wordcloud = WordCloud(width = 1000, height = 400, background_color = 'lightgrey',
                          max_words = 1628, relative_scaling = 1,
                          color_func = random_color_func,
                          normalize_plurals = False)
    wordcloud.generate_from_frequencies(words)  # Generate word cloud from word frequencies
    ax.imshow(wordcloud, interpolation="bilinear")  # Display the word cloud on the subplot
    ax.axis('off')  # Turn off axis labels
    plt.title('Cluster nº{}'.format(increment - 1))  # Set title for the subplot

# Create a figure to hold the word clouds
fig = plt.figure(1, figsize=(14, 14))
color = [0, 160, 130, 95, 280, 40, 330, 110, 25]  # Define color values for word clouds
for i in range(n_clusters):
    cluster_occurrences = cluster_frequency[i]  # Get word occurrences for current cluster
    tone = color[i]  # Define the color tone for the word cloud
    sorted_list = []  # Initialize list to store word occurrences
    for key, value in cluster_occurrences.items():
        sorted_list.append([key, value])  # Append word and its frequency to the list
    sorted_list.sort(key=lambda x: x[1], reverse=True)  # Sort word list by frequency
    make_wordcloud(sorted_list, i + 1)  # Generate word cloud for the current cluster


# In[73]:


# From this representation, we can observe that some clusters contain objects associated with gifts, 
# (keywords: Christmas, packaging, card, etc.). 
# Another cluster may contain luxury items and jewelry (keywords: necklace, bracelet, lace, silver, etc.). 
# However, it's also evident that many words appear in multiple clusters, making it difficult to clearly distinguish them.


# # (c) Principal Component Analysis (PCA)

# In[74]:


# In order to assess the distinctiveness of these clusters, 
# I analyze their composition. Due to the high dimensionality of the initial matrix, 
# I opt to perform PCA:


# In[75]:


# Initialize and fit PCA model
pca = PCA()
pca.fit(matrix)

# Transform the input matrix using PCA
transformed_matrix = pca.transform(matrix)


# In[77]:


# Next, i check for the amount of variance explained by each component:

# Create a plot to visualize the explained variance of principal components
figure, axis = plt.subplots(figsize=(14, 5))

# Set the font scale for seaborn
sns.set(font_scale = 1)

# Plot the cumulative explained variance
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where = 'mid',
         label ='cumulative explained variance')

# Plot the individual explained variance
sns.barplot(x = np.arange(1, matrix.shape[1] + 1), y = pca.explained_variance_ratio_, alpha = 0.5, color = 'red',
            label = 'Individual explained variance')

# Set the limit for the x-axis
plt.xlim(0, 100)

# Set the labels for the x-axis ticks
axis.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in axis.get_xticklabels()])

# Set the label for the y-axis
plt.ylabel('Explained variance', fontsize = 14)

# Set the label for the x-axis
plt.xlabel('Principal components', fontsize = 14)

# Add a legend to the plot
plt.legend(loc = 'upper left', fontsize=13);


# In[78]:


# The number of components required to explain the data is crucial:
# over 100 components are needed to account for 90% of the variance. 
# However, for practical purposes, a limited number of components are retained 
# since this decomposition is primarily for data visualization.


# In[79]:


# Initialize PCA with 50 components
pca = PCA(n_components=50)

# Fit PCA to the matrix and transform it to 9-dimensional space
matrix_9D = pca.fit_transform(matrix)

# Convert the transformed matrix to a DataFrame
df_9D = pd.DataFrame(matrix_9D)

# Add a 'cluster' column to the DataFrame with the cluster labels
df_9D['cluster'] = pd.Series(cluster_labels)


# In[80]:


# Import necessary libraries
import matplotlib.patches as mpatches

# Set seaborn styles
sns.set_style("white")
sns.set_context("notebook", font_scale = 1, rc = {"lines.linewidth": 2.5})

# Define color map for cluster labels
CLUSTER_COLOR_MAP = {0: 'r', 1: 'gold', 2: 'b', 3: 'k', 4: 'c', 5: 'g'}
label_color = [CLUSTER_COLOR_MAP[l] for l in df_9D['cluster']]

# Create figure
figure = plt.figure(figsize=(15, 8))
increment = 0

# Scatter plot for PCA components
for ix in range(4):
    for iy in range(ix + 1, 4):
        increment += 1
        axis = figure.add_subplot(2, 3, increment)
        axis.scatter(df_9D[ix], df_9D[iy], c = label_color, alpha = 0.4)
        plt.ylabel('PCA {}'.format(iy + 1), fontsize = 12)
        plt.xlabel('PCA {}'.format(ix + 1), fontsize = 12)
        axis.yaxis.grid(color = 'lightgray', linestyle = ':')
        axis.xaxis.grid(color = 'lightgray', linestyle = ':')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        
        if increment == 9:
            break
    if increment == 9:
        break

# Set legend
legend_patches = [mpatches.Patch(color = CLUSTER_COLOR_MAP[i], label=str(i)) for i in range(5)]
plt.legend(handles = legend_patches, bbox_to_anchor = (1.1, 0.97), title = 'Cluster', facecolor = 'lightgrey',
           shadow = True, frameon = True, framealpha = 1, fontsize = 13, bbox_transform = plt.gcf().transFigure)

plt.show()


# # 4 Customer Categories

# # 4.1 Formatting Data

# In[81]:


# In the previous section, the different products were grouped into five clusters. 
# In order to prepare the rest of the analysis, a first step consists of introducing this information into the dataframe. 
# To do this, we create the categorical variable 'product_cluster_category' where we indicate the cluster of each product.


# In[82]:


# Create a dictionary to map product descriptions to their respective cluster labels
product_cluster_mapping = dict()

# Iterate over product descriptions and cluster labels, and populate the dictionary
for key, val in zip(product_descriptions, cluster_labels):
    product_cluster_mapping[key] = val 

# Map the 'product_cluster_category' column in the dataframe using the created dictionary
ecommerce_data_cleaned['product_cluster_category'] = ecommerce_data_cleaned.loc[:, 'itemDescription'].map(product_cluster_mapping)


# # 4.1.1 Grouping products

# In[83]:


# In this step, we create the categ_N variables (with N∈[0:4]), that contain the amount spent in each product category.


# In[84]:


# Iterate over each cluster category
for i in range(5):
    # Create the column_label for the category
    column_label = 'categ_{}'.format(i)
    
    # Filter the DataFrame for the current cluster category
    cluster_category_data = ecommerce_data_cleaned[ecommerce_data_cleaned['product_cluster_category'] == i]
    
    # Calculate the total price for each product in the category
    category_prices = cluster_category_data['unitPrice'] * (cluster_category_data['quantity'] - 
                                                            cluster_category_data['cancelled_orders_count'])
    
    # Set negative prices to zero
    category_prices = category_prices.apply(lambda x: x if x > 0 else 0)
    
    # Assign the calculated prices to the corresponding category column
    ecommerce_data_cleaned.loc[:, column_label] = category_prices
    
    # Fill missing values with zero in the category column
    ecommerce_data_cleaned[column_label].fillna(0, inplace=True)

# Display the selected columns for the first five rows of the DataFrame
ecommerce_data_cleaned[['invoice', 'itemDescription', 'product_cluster_category', 
                        'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']][:5]


# In[85]:


# Consolidating Order Information.
# Up to this point, information pertaining to a single order was spread across multiple lines in the dataframe, with one line per product. 
# To simplify analysis, I consolidate the information related to each order into a single entry.
# I create a new dataframe that includes, for each order, the total basket amount, as well as its distribution across the 5 product categories.


# In[86]:


# Calculate total purchases per customer and order
temp_df = ecommerce_data_cleaned.groupby(by = ['CustomerID', 'invoice'], as_index = False)['TotalPrice'].sum()
order_totals = temp_df.rename(columns = {'TotalPrice': 'Cart Price'})

# Calculate percentage of order price per product category
for i in range(5):
    column_label = 'categ_{}'.format(i)
    temp_df = ecommerce_data_cleaned.groupby(by = ['CustomerID', 'invoice'], as_index = False)[column_label].sum()
    order_totals[column_label] = temp_df[column_label].values

# Extract order date
ecommerce_data_cleaned['invoice_date_int'] = ecommerce_data_cleaned['invoiceDate'].astype('int64')
temp_df = ecommerce_data_cleaned.groupby(by = ['CustomerID', 'invoice'], as_index = False)['invoice_date_int'].mean()
ecommerce_data_cleaned.drop('invoice_date_int', axis = 1, inplace = True)
order_totals['invoiceDate'] = pd.to_datetime(temp_df['invoice_date_int'])

# Select significant entries
order_totals = order_totals[order_totals['Cart Price'] > 0]
order_totals.sort_values('CustomerID', ascending = True)[:5]


# # 4.1.2 Separation of data over time

# In[87]:


# The dataframe order_totals contains information for a period of 12 months.
# Later, one of the objectives will be to develop a model capable of characterizing 
# and anticipating the habits of the customers visiting the site and this, from
# their first visit. 
# In order to be able to test the model in a realistic way,
# I split the data set by retaining the first 10 months to develop the model and the following two months to test it


# In[88]:


# Print the range of dates in the 'invoiceDate' column
print(order_totals['invoiceDate'].min(), '->',  order_totals['invoiceDate'].max())


# In[89]:


# Convert date to numpy datetime64 object
date_threshold = np.datetime64('2023-10-01')

# Select data for training set before October 1, 2023
training_set = order_totals[order_totals['invoiceDate'] < date_threshold]

# Select data for test set on or after October 1, 2023
test_set = order_totals[order_totals['invoiceDate'] >= date_threshold]

# Update order_totals with the training set
order_totals = training_set.copy(deep=True)


# # 4.1.3 Consumer Order Combinations

# In[90]:


# Group entries by user to analyze their purchasing behavior.
# Calculate the number of purchases, minimum, maximum, average purchase amounts,
# and total amount spent during all visits for each user.


# In[91]:


# Calculate the number of transactions and statistics on the cart amount per user
transactions_per_user = order_totals.groupby(by=['CustomerID'])['Cart Price'].agg(['count', 'min', 'max', 'mean', 'sum'])

# Calculate the percentage of each product category in the total amount spent for each user
for i in range(5):
    column_label = 'categ_{}'.format(i)
    transactions_per_user.loc[:, column_label] = order_totals.groupby(by=['CustomerID'])[column_label].sum() / \
                                                transactions_per_user['sum'] * 100

# Reset the index of transactions_per_user dataframe
transactions_per_user.reset_index(drop=False, inplace=True)

# Calculate the sum of purchases in the 'categ_0' category for each user
order_totals.groupby(by=['CustomerID'])['categ_0'].sum()

# Sort transactions_per_user dataframe by CustomerID in ascending order
transactions_per_user.sort_values('CustomerID', ascending=True)[:5]


# In[92]:


# Finally, defining two additional variables:

# - FirstPurchase: Number of days elapsed since the first purchase
# - LastPurchase: Number of days since the last purchase


# In[93]:


# Get the last date in the dataset
latest_date = order_totals['invoiceDate'].max().date()

# DataFrame for the first registration date of each customer
first_purchase_date = pd.DataFrame(order_totals.groupby(by = ['CustomerID'])['invoiceDate'].min())

# DataFrame for the last purchase date of each customer
last_purchase_date = pd.DataFrame(order_totals.groupby(by = ['CustomerID'])['invoiceDate'].max())

# Calculate the number of days since the last purchase for each customer
days_since_first_purchase = first_purchase_date.applymap(lambda x: (latest_date - x.date()).days)

# Calculate the number of days since the first purchase for each customer
days_since_last_purchase = last_purchase_date.applymap(lambda x: (latest_date - x.date()).days)

# Assign the calculated values to the transactions_per_user DataFrame
transactions_per_user.loc[:, 'LastPurchase'] = days_since_last_purchase.reset_index(drop=False)['invoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = days_since_first_purchase.reset_index(drop=False)['invoiceDate']

# Display the first 5 rows of the transactions_per_user DataFrame
transactions_per_user[:5]


# In[94]:


# Identifying customers who made only one purchase
# One-time customers are of special interest as they may need targeted retention strategies
# Here, I determine the proportion of customers who made only one purchase, 
# which accounts for approximately one-third of the total customer base.


# In[95]:


# Count the number of customers who made only one purchase
num_customers_single_purchase = transactions_per_user[transactions_per_user['count'] == 1].shape[0]

# Total number of customers
total_num_customers = transactions_per_user.shape[0]

# Print the number and percentage of customers with a single purchase
print("Number of customers with a single purchase: {:<2}/{:<5} ({:<2.2f}%)".format(num_customers_single_purchase, 
                                                                                   total_num_customers, 
                                                                                   num_customers_single_purchase / 
                                                                                   total_num_customers * 100))


# # 4.2 Creation of Customer Categories

# # 4.2.1 Data Encoding

# In[96]:


# The dataframe transactions_per_user summarizes all commands made by each client.
# Each entry in this dataframe represents a particular client.
# We use this information to characterize different types of customers
# and retain only a subset of variables for further analysis.


# In[97]:


# Define the list of columns to select from transactions_per_user dataframe
selected_columns = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']

# Create a deep copy of transactions_per_user dataframe
customer_summary = transactions_per_user.copy(deep = True)

# Convert the selected columns to a NumPy array
matrix = customer_summary[selected_columns].values


# In[98]:


# Standardizing Data

# Before continuing analysis, i standardize the selected variables to ensure they contribute equally,
# regardless of their original scales.


# In[99]:


# Initialize StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data
scaler.fit(matrix)

# Print the mean values of the variables
print('Variable Mean Values: \n' + 90*'-' + '\n' , scaler.mean_)

# Standardize the data matrix
scaled_matrix = scaler.transform(matrix)


# In[100]:


# Before creating customer clusters, 
# it's beneficial to define a lower-dimensional base to describe the scaled_matrix. 
# This base will be used to visualize the clusters and assess their separation quality. 
# Hence, I'll conduct Principal Component Analysis (PCA) as a preliminary step:


# In[101]:


# Initialize PCA
pca = PCA()

# Fit PCA to the scaled matrix
pca.fit(scaled_matrix)

# Transform the scaled matrix using the learned PCA
pca_samples = pca.transform(scaled_matrix)


# In[102]:


# Visualizing the variance explained by each principal component.


# In[103]:


# Create a plot to visualize the explained variance of principal components
figure, axis = plt.subplots(figsize=(14, 5))

# Set the font scale for seaborn
sns.set(font_scale = 1)

# Plot the cumulative explained variance
plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label ='cumulative explained variance')

# Plot the individual explained variance
sns.barplot(x = np.arange(1, matrix.shape[1] + 1), y = pca.explained_variance_ratio_, alpha = 0.5, color = 'red',
            label ='Individual explained variance')

# Set the limit for the x-axis
plt.xlim(0, 10)

# Set the labels for the x-axis ticks
axis.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in axis.get_xticklabels()])

# Set the label for the y-axis
plt.ylabel('Explained variance', fontsize=14)

# Set the label for the x-axis
plt.xlabel('Principal components', fontsize=14)

# Add a legend to the plot
plt.legend(loc ='best', fontsize=13);


# # 4.2.2 Creating Customer Categories

# In[104]:


# At this stage, I define clusters of clients using the k-means algorithm from scikit-learn,
# based on the standardized matrix defined earlier. The number of clusters is chosen based on
# the silhouette score, and it is found that the best score is obtained with 11 clusters.


# In[105]:


# Set the number of clusters
n_clusters = 11

# Initialize the KMeans algorithm with k-means++ initialization
kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init = 100)

# Fit the KMeans model to the standardized matrix
kmeans.fit(scaled_matrix)

# Predict the clusters for each sample in the standardized matrix
customer_clusters = kmeans.predict(scaled_matrix)

# Calculate the average silhouette score for the clustering
avg_silhouette = silhouette_score(scaled_matrix, customer_clusters)

# Print the silhouette score
print('Silhouette Score: {:<.3f}'.format(avg_silhouette))


# # (a) Report via PCA

# In[106]:


# There is a disparity in the sizes of different groups created. 
# Hence, I will now analyze the content of these clusters to validate (or not) this separation.

# Using the result of PCA:


# In[107]:


# Perform PCA with 6 components
pca = PCA(n_components = 6)
matrix_3D = pca.fit_transform(scaled_matrix)

# Create a DataFrame from the transformed matrix
df_9D = pd.DataFrame(matrix_3D)

# Add a column for cluster labels
df_9D['cluster'] = pd.Series(customer_clusters)


# In[108]:


# Import necessary libraries
import matplotlib.patches as mpatches

# Set seaborn style and context
sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

# Define color map for clusters
CLUSTER_COLOR_MAP = {0: 'r', 1: 'tan', 2: 'b', 3: 'k', 4: 'c', 5: 'g', 6: 'deeppink', 7: 'skyblue', 
                     8: 'darkcyan', 9: 'orange', 10: 'yellow', 11: 'tomato', 12: 'seagreen'}

# Assign colors to cluster labels
label_color = [CLUSTER_COLOR_MAP[l] for l in df_9D['cluster']]

# Create figure for plotting
figure = plt.figure(figsize=(12, 10))
increment = 0

# Iterate over pairs of principal components for plotting
for ix in range(6):
    for iy in range(ix + 1, 6):
        increment += 1
        axis = figure.add_subplot(4, 3, increment)
        
        # Scatter plot of PCA components colored by cluster label
        axis.scatter(df_9D[ix], df_9D[iy], c=label_color, alpha=0.5) 
        
        # Set axis labels and grid
        plt.ylabel('PCA {}'.format(iy + 1), fontsize=12)
        plt.xlabel('PCA {}'.format(ix + 1), fontsize=12)
        axis.yaxis.grid(color='lightgray', linestyle=':')
        axis.xaxis.grid(color='lightgray', linestyle=':')
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        
        if increment == 12:
            break
    if increment == 12:
        break

# Set legend for clusters
comp_handler = []
for i in range(n_clusters):
    comp_handler.append(mpatches.Patch(color=CLUSTER_COLOR_MAP[i], label=i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9), title='Cluster', facecolor='lightgrey',
           shadow=True, frameon=True, framealpha=1, fontsize=13, bbox_transform=plt.gcf().transFigure)

plt.tight_layout()


# In[109]:


# From this representation, 
# it's evident that the first principal component effectively separates the smallest clusters from the rest. 
# Generally, there's always a representation where two clusters appear distinct.


# # (b) Intra-cluster silhouette score

# In[110]:


# Similar to product categories, 
# another approach to assess the quality of separation is by examining the silhouette scores within individual clusters.


# In[111]:


# Compute silhouette scores for each sample in the scaled matrix using the customer clusters
silhouette_values = silhouette_samples(scaled_matrix, customer_clusters)

# Define individual silhouette scores
silhouette_values = silhouette_samples(scaled_matrix, customer_clusters)

# Plot the silhouette graph
plot_silhouette(n_clusters, [-0.15, 0.55], len(scaled_matrix), silhouette_values, customer_clusters)


# # (c) Customers' Morphotype

# In[112]:


# Now, having confirmed that the various clusters are distinct,
# I proceed to examine the purchasing habits of customers within each cluster. 
# To begin this analysis, 
# I augment the customer_summary dataframe with a new column indicating the cluster assignment for each customer.


# In[113]:


# Assign the cluster labels to each customer in the customer summary dataframe
customer_summary.loc[:, 'cluster'] = customer_clusters


# In[114]:


# Calculate the average values for each cluster in the customer summary dataframe
# This includes metrics such as average cart price, number of visits, and total spent
# Additionally, determine the size of each cluster (number of customers)


# In[115]:


# Initialize an empty DataFrame to store merged data
cluster_stats_df = pd.DataFrame()

# Iterate over each cluster
for i in range(n_clusters):
    # Calculate the mean values for the current cluster and convert to DataFrame
    test = pd.DataFrame(customer_summary[customer_summary['cluster'] == i].mean())
    # Transpose the DataFrame and set the index to 'cluster'
    test = test.T.set_index('cluster', drop=True)
    # Add a column 'size' representing the number of clients in the current cluster
    test['size'] = customer_summary[customer_summary['cluster'] == i].shape[0]
    # Concatenate the current cluster data with the merged DataFrame
    cluster_stats_df = pd.concat([cluster_stats_df, test])

# Drop the 'CustomerID' column from the merged DataFrame
cluster_stats_df.drop('CustomerID', axis=1, inplace=True)

# Print the total number of customers across all clusters
print('Number of Customers:', cluster_stats_df['size'].sum())

# Sort the merged DataFrame by the 'sum' column
cluster_stats_df = cluster_stats_df.sort_values('sum')


# In[116]:


# In conclusion, I restructure the dataframe by organizing the clusters according to two criteria: 
# firstly, based on the expenditure within each product category, and secondly, by the total amount spent.


# In[117]:


# Initialize an empty list to store the indexes of clusters meeting the condition
selected_indexes = []

# Iterate through each product category
for i in range(5):
    # Get the column name corresponding to the product category
    column = 'categ_{}'.format(i)
    # Find the index of the cluster with spending greater than 45 in the current product category
    selected_indexes.append(cluster_stats_df[cluster_stats_df[column] > 45].index.values[0])

# Reorder the list of selected indexes
reordered_indexes = selected_indexes + [s for s in cluster_stats_df.index if s not in selected_indexes]

# Reindex the dataframe using the reordered index list
cluster_stats_df = cluster_stats_df.reindex(index=reordered_indexes)

# Reset the index of the dataframe
cluster_stats_df = cluster_stats_df.reset_index(drop=False)

# Display the selected columns of the dataframe
display(cluster_stats_df[['cluster', 'count', 'min', 'max', 'mean', 'sum', 'categ_0',
                          'categ_1', 'categ_2', 'categ_3', 'categ_4', 'size']])


# # (d) Customers Morphology

# In[118]:


# Finally, I've created a visualization representing the various customer morphologies.


# In[119]:


# Function to scale data to a specific range
def _scale_data(data, ranges):
    (x1, x2) = ranges[0]
    d = data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

# Class to create Radar Charts
class RadarChart():
    def __init__(self, figure, location, sizes, variables, ranges, n_ordinate_levels=6):
        # Define angles for each variable
        angles = np.arange(0, 360, 360./len(variables))
        
        # Extract location and sizes
        ix, iy = location[:]
        size_x, size_y = sizes[:]
        
        # Add polar axes to the figure
        axes = [figure.add_axes([ix, iy, size_x, size_y], polar=True, label="axes{}".format(i)) for i in range(len(variables))]

        # Set variable labels along the radial axis
        _, text = axes[0].set_thetagrids(angles, labels=variables)
        
        # Rotate labels based on their position
        for txt, angle in zip(text, angles):
            if angle > -1 and angle < 181:
                txt.set_rotation(angle - 90)
            else:
                txt.set_rotation(angle - 270)
        
        # Hide extra axes and grid lines
        for axis in axes[1:]:
            axis.patch.set_visible(False)
            axis.xaxis.set_visible(False)
            axis.grid("off")
        
        # Set radial grids and labels
        for i, axis in enumerate(axes):
            grid = np.linspace(*ranges[i], num=n_ordinate_levels)
            num_labels = len(grid)
            grid_label = [""] + ["{:.0f}".format(x) for x in grid[1:]]
            axis.set_rgrids(grid, labels=grid_label, angle=angles[i])
            axis.set_ylim(*ranges[i])
        
        # Convert angles to radians and store ranges and axes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.axis = axes[0]
                
    def plot(self, data, *args, **kw):
        # Scale data and plot on the radar chart
        sdata = _scale_data(data, self.ranges)
        self.axis.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        # Scale data and fill the radar chart area
        sdata = _scale_data(data, self.ranges)
        self.axis.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        # Add legend to the radar chart
        self.axis.legend(*args, **kw)
        
    def title(self, title, *args, **kw):
        # Add title to the radar chart
        self.axis.text(0.9, 1, title, transform=self.axis.transAxes, *args, **kw)


# In[120]:


# The provided code snippet allows for a comprehensive overview of the contents within each cluster.


# In[121]:


figure = plt.figure(figsize=(10,12))

attributes = ['count', 'mean', 'sum', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
# Define the attributes to be plotted on the radar chart

ranges = [[0.01, 10], [0.01, 1500], [0.01, 10000], [0.01, 75], [0.01, 75], [0.01, 75], [0.01, 75], [0.01, 75]]
# Define the ranges for each attribute, specifying the minimum and maximum values

index  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# Define the index for clusters

n_groups = n_clusters ; i_cols = 3
i_rows = n_groups//i_cols
size_x, size_y = (1/i_cols), (1/i_rows)

for ind in range(n_clusters):
    # Iterate through each cluster
    ix = ind%3 ; iy = i_rows - ind//3
    pos_x = ix*(size_x + 0.05) ; pos_y = iy*(size_y + 0.05)            
    location = [pos_x, pos_y]  ; sizes = [size_x, size_y] 
    # Calculate the position and size of each radar chart
    
    data = np.array(cluster_stats_df.loc[index[ind], attributes])    
    # Extract data for the current cluster
    
    radar = RadarChart(figure, location, sizes, attributes, ranges)
    # Create a radar chart object
    
    radar.plot(data, color = 'b', linewidth=2.0)
    # Plot the data on the radar chart
    
    radar.fill(data, alpha = 0.2, color = 'b')
    # Fill the area under the radar chart
    
    radar.title(title = 'cluster nº{}'.format(index[ind]), color = 'r')
    # Set the title for the radar chart
    ind += 1


# In[122]:


# The analysis reveals distinct patterns among clusters. 
# For instance, the initial five clusters exhibit a significant preference for purchases in specific product categories. 
# In contrast, other clusters showcase variations from the average basket compositions, 
# mean expenditure, total expenditure by clients, or total visit counts.


# # 5 Customer Classification

# In[123]:


# In this section, the goal is to develop a classifier capable of categorizing customers into the various client segments,
# identified in the preceding section. The aim is to enable this classification from the very first visit. 
# To achieve this objective, I will evaluate multiple classifiers available in scikit-learn. To streamline their utilization, 
# I will define a class that facilitates the integration of various functionalities common to these classifiers.


# In[124]:


class Class_Fit(object):
    def __init__(self, clf, params=None):
        # Initialize classifier with optional parameters
        if params:            
            self.clf = clf(**params)
        else:
            self.clf = clf()

    def train(self, x_train, y_train):
        # Train the classifier
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        # Make predictions using the trained classifier
        return self.clf.predict(x)
    
    def grid_search(self, parameters, Kfold):
        # Perform grid search cross-validation
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=Kfold)
        
    def grid_fit(self, X, Y):
        # Fit the grid search to the data
        self.grid.fit(X, Y)
        
    def grid_predict(self, X, Y):
        # Make predictions using the grid search optimized classifier
        self.predictions = self.grid.predict(X)
        print("Accuracy: {:.2f} %".format(100*metrics.accuracy_score(Y, self.predictions)))


# In[125]:


# The objective is to determine the customer class at their initial visit. 
# Therefore, only the cart content variables are considered, 
# excluding factors such as visit frequency or price variations over time.


# In[126]:


# Define the columns to be used as features for classification
columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']

# Extract the features (X) and the target variable (Y) from the customer summary data
X = customer_summary[columns]  # Features
Y = customer_summary['cluster']  # Target variable


# In[127]:


# Finally, I split the dataset into training and testing sets:


# In[128]:


# Split the dataset into training and testing sets
# X_train: Training features
# X_test: Testing features
# Y_train: Training labels
# Y_test: Testing labels
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8)


# # 5.1 Support Vector Machine Classifier (SVC)

# In[129]:


# Instantiate the Class_Fit class to use the SVC classifier
# Call the grid_search() method to search for optimal hyperparameters using cross-validation

# Parameters:
# - hyperparameters: the hyperparameters for which optimal values will be sought
# - Kfold: the number of folds to be used for cross-validation


# In[130]:


# Instantiate the Class_Fit class with LinearSVC classifier
svc = Class_Fit(clf=svm.LinearSVC)

# Perform grid search to find optimal hyperparameters for LinearSVC
# Parameters:
# - parameters: a dictionary containing the hyperparameters to be searched over
# - Kfold: the number of folds for cross-validation
svc.grid_search(parameters=[{'C': np.logspace(-2, 2, 10)}], Kfold=5)


# In[131]:


# After creating this instance, I proceed to train the classifier with the training data.


# In[132]:


# fitting the classifier to the training data
svc.grid_fit(X = X_train, Y = Y_train)


# In[133]:


# Evaluating the classifier's performance on the test data.
svc.grid_predict(X_test, Y_test)


# # 5.1.1 Confusion Matrix

# In[134]:


# The accuracy of the classifier's results appears to be acceptable. However, 
# it's essential to note that there was an imbalance in the sizes of the different classes defined earlier. 
# One class comprises approximately 40% of the clients, indicating potential bias. 
# Therefore, it's insightful to examine the relationship between the predicted and actual values across the various classes. 
# This analysis is facilitated by confusion matrices. To visualize them, I utilize code from the scikit-learn documentation.


# In[135]:


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    # Plotting the confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)
    
    # Adding text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    # Adding labels and adjusting layout
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[136]:


# I Create the following representation from the plot_confusion_matrix function:

# Define class names for the confusion matrix
class_names = [i for i in range(11)]

# Compute the confusion matrix based on the predictions
cnf_matrix = confusion_matrix(Y_test, svc.predictions) 

# Set options for printing
np.set_printoptions(precision=2)

# Create a figure with a specified size
plt.figure(figsize = (8,8))

# Plot the confusion matrix
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix')


# # 5.1.2 Learning curve

# In[137]:


# A common method to assess model performance and detect issues like overfitting or underfitting
# This curve helps to understand if the model would benefit from more data
# Using the code from the scikit-learn documentation


# In[138]:


# Function to plot the learning curve of a given estimator
# It generates a plot showing the training and cross-validation scores over different training set sizes
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10)):
    # Create a new figure for the plot
    plt.figure()
    # Set the title of the plot
    plt.title(title)
    # Set the y-axis limits if specified
    if ylim is not None:
        plt.ylim(*ylim)
    # Label the x-axis
    plt.xlabel("Training examples")
    # Label the y-axis
    plt.ylabel("Score")
    # Generate the learning curve using the provided estimator, data, and parameters
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    # Compute the mean and standard deviation of training scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    # Compute the mean and standard deviation of test scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # Add grid lines to the plot
    plt.grid()

    # Fill the area between the training scores with a light red color
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    # Fill the area between the test scores with a light green color
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # Plot the mean training scores as circles connected by a red line
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    # Plot the mean test scores as circles connected by a green line
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    # Add a legend to the plot and position it at the best location
    plt.legend(loc="best")
    # Return the plot object
    return plt


# In[139]:


# Generating the learning curve for the SVC classifier
g = plot_learning_curve(svc.grid.best_estimator_,
                        "SVC learning curves", X_train, Y_train, ylim=[1.01, 0.6],
                        cv=5, train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                          0.6, 0.7, 0.8, 0.9, 1])


# In[142]:


# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, svc.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, svc.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, svc.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, svc.predictions)}")


# In[143]:


# On this curve, we observe convergence between the training and cross-validation curves as the sample size increases, 
# indicating low variance in the model and no overfitting issues.
# Additionally, the accuracy of the training curve remains high, indicating low bias and no underfitting of the data.


# # 5.2 Logistic Regression

# In[144]:


# In this section, I analyse the logistic regression classifier. Similar to previous steps, 
# I instantiate the Class_Fit class, fit the model to the training data, and evaluate its predictions against the actual values.


# In[145]:


# Create an instance of the Class_Fit class for logistic regression classifier
lr = Class_Fit(clf=linear_model.LogisticRegression)

# Perform grid search to find the best hyperparameters for logistic regression
lr.grid_search(parameters=[{'C': np.logspace(-2, 2, 20)}], Kfold=5)

# Fit the logistic regression model to the training data
lr.grid_fit(X=X_train, Y=Y_train)

# Make predictions using the fitted logistic regression model on the test data
lr.grid_predict(X_test, Y_test)


# In[146]:


# Then, I plot the learning curve to assess the quality of the model:
g = plot_learning_curve(lr.grid.best_estimator_, "Logistic Regression learning curves", X_train, Y_train,
                        ylim = [1.01, 0.7], cv = 5, 
                        train_sizes = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[147]:


# Predict the labels for the test data using the trained Logistic Regression classifier.
lr.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, lr.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, lr.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, lr.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, lr.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, lr.predictions)}")


# # 5.3 k-Nearest Neighbours

# In[149]:


# This code initializes the KNeighborsClassifier and performs grid search to find the best hyperparameter.
knn = Class_Fit(clf = neighbors.KNeighborsClassifier)
knn.grid_search(parameters = [{'n_neighbors': np.arange(1, 50, 1)}], Kfold = 5)

# Fits the model to the training data.
knn.grid_fit(X = X_train, Y = Y_train)

# Makes predictions on the test data.
knn.grid_predict(X_test, Y_test)


# In[150]:


# This code generates the learning curve for the KNeighborsClassifier to evaluate its performance.
g = plot_learning_curve(knn.grid.best_estimator_, "Nearest Neighbors learning curves", X_train, Y_train,
                        ylim=[1.01, 0.7], cv=5, 
                        train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[151]:


# Predict the labels for the test data using the trained KNeighborsClassifier classifier.
knn.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, knn.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, knn.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, knn.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, knn.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, knn.predictions)}")


# # 5.4 Decision Tree

# In[153]:


# Create an instance of the Class_Fit class with DecisionTreeClassifier as the classifier.
tr = Class_Fit(clf = tree.DecisionTreeClassifier)

# Perform a grid search to find the best parameters for the Decision Tree classifier.
tr.grid_search(parameters = [{'criterion': ['entropy', 'gini'], 'max_features': ['sqrt', 'log2']}], Kfold=5)

# Fit the Decision Tree classifier to the training data.
tr.grid_fit(X = X_train, Y = Y_train)

# Predict the labels for the test data using the trained Decision Tree classifier.
tr.grid_predict(X_test, Y_test)


# In[154]:


# Plot the learning curve for the Decision Tree classifier to evaluate its performance.
g = plot_learning_curve(tr.grid.best_estimator_, "Decision tree learning curves", X_train, Y_train,
                        ylim=[1.01, 0.7], cv=5, 
                        train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[155]:


# Predict the labels for the test data using the trained Decision Tree classifier.
tr.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, tr.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, tr.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, tr.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, tr.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, tr.predictions)}")


# # 5.5 Random Forest

# In[157]:


# Create an instance of the Class_Fit class with the Random Forest classifier.
rf = Class_Fit(clf = ensemble.RandomForestClassifier)

# Define the parameter grid for grid search.
param_grid = {'criterion': ['entropy', 'gini'], 'n_estimators': [20, 40, 60, 80, 100],
              'max_features': ['sqrt', 'log2']}

# Perform grid search to find the best parameters for the Random Forest classifier.
rf.grid_search(parameters = param_grid, Kfold = 5)

# Fit the Random Forest classifier to the training data.
rf.grid_fit(X = X_train, Y = Y_train)

# Predict using the trained Random Forest classifier on the test data.
rf.grid_predict(X_test, Y_test)


# In[158]:


# Plot the learning curve for the Random Forest classifier.
g = plot_learning_curve(rf.grid.best_estimator_, "Random Forest learning curves", X_train, Y_train,
                        ylim=[1.01, 0.7], cv=5, 
                        train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[159]:


# Predict the labels for the test data using the trained Random Forest classifier.
rf.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, rf.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, rf.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, rf.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, rf.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, rf.predictions)}")


# # 5.6 AdaBoost Classifier

# In[161]:


# Create an instance of the AdaBoost classifier.
ada = Class_Fit(clf=AdaBoostClassifier)

# Define the parameter grid for grid search.
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

# Perform grid search to find the best parameters.
ada.grid_search(parameters=param_grid, Kfold=5)

# Fit the classifier to the training data.
ada.grid_fit(X=X_train, Y=Y_train)

# Predict the labels for the test data.
ada.grid_predict(X_test, Y_test)


# In[162]:


# Generate the learning curves for the AdaBoost classifier.
g = plot_learning_curve(ada.grid.best_estimator_, "AdaBoost learning curves", X_train, Y_train,
                        ylim=[1.01, 0.4], cv=5, 
                        train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[163]:


# Predict the labels for the test data using the trained AdaBoost classifier.
ada.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, ada.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, ada.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, ada.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, ada.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, ada.predictions)}")


# # 5.7 Gradient Boosting Classifier

# In[165]:


# Create an instance of the Gradient Boosting classifier.
gb = Class_Fit(clf = ensemble.GradientBoostingClassifier)

# Define the parameter grid for grid search.
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}

# Perform grid search to find the best parameters.
gb.grid_search(parameters = param_grid, Kfold = 5)

# Fit the Gradient Boosting classifier to the training data.
gb.grid_fit(X = X_train, Y = Y_train)

# Predict the labels for the test data using the fitted model.
gb.grid_predict(X_test, Y_test)


# In[166]:


# Plot the learning curve for the Gradient Boosting classifier.
g = plot_learning_curve(gb.grid.best_estimator_, "Gradient Boosting learning curves", X_train, Y_train,
                        ylim=[1.01, 0.7], cv=5, 
                        train_sizes=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


# In[167]:


# Predict the labels for the test data using the trained Gradient Boosting classifier.
gb.grid_predict(X_test, Y_test)

# Print the confusion matrix
conf_matrix = confusion_matrix(Y_test, gb.predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Print the f1 score, precision, recall, and support
print(f"F1 Score: {f1_score(Y_test, gb.predictions, average='weighted')}")
print(f"Precision: {precision_score(Y_test, gb.predictions, average='weighted')}")
print(f"Recall: {recall_score(Y_test, gb.predictions, average='weighted')}")
print(f"Support:\n{classification_report(Y_test, gb.predictions)}")


# # 5.8 Voting Consensus

# In[168]:


# In this phase, i aim to enhance the classification model
# by aggregating the outcomes of the various classifiers discussed earlier. 
# The approach involves selecting the customer category based on the majority vote from multiple classifiers. 
# To implement this, I employ the VotingClassifier method provided by the sklearn package. 
# Initially, I fine-tune the parameters of the different classifiers using the previously identified optimal parameters.


# In[169]:


# Instantiate Random Forest Classifier with the best parameters obtained from grid search
rf_best  = ensemble.RandomForestClassifier(**rf.grid.best_params_)

# Instantiate Gradient Boosting Classifier with the best parameters obtained from grid search
gb_best  = ensemble.GradientBoostingClassifier(**gb.grid.best_params_)

# Instantiate Support Vector Machine Classifier with the best parameters obtained from grid search
svc_best = svm.LinearSVC(**svc.grid.best_params_)

# Instantiate Decision Tree Classifier with the best parameters obtained from grid search
tr_best  = tree.DecisionTreeClassifier(**tr.grid.best_params_)

# Instantiate K-Nearest Neighbors Classifier with the best parameters obtained from grid search
knn_best = neighbors.KNeighborsClassifier(**knn.grid.best_params_)

# Instantiate Logistic Regression Classifier with the best parameters obtained from grid search
lr_best  = linear_model.LogisticRegression(**lr.grid.best_params_)


# In[170]:


# Next, I create a classifier that combines the predictions from multiple classifiers.
votingC = ensemble.VotingClassifier(estimators=[('rf', rf_best),('gb', gb_best),
                                                ('knn', knn_best)], voting='soft')


# In[171]:


# Fit the VotingClassifier to the training data
votingC = votingC.fit(X_train, Y_train)


# In[172]:


# Finally, generate predictions using the trained model.
predictions = votingC.predict(X_test)
print("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y_test, predictions)))


# In[185]:


# It's worth noting that in defining the votingC classifier, 
# I opted to include only a subset of the classifiers defined earlier, 
# specifically the Random Forest, k-Nearest Neighbors, and Gradient Boosting classifiers. 
# This selection was made based on their performance in the subsequent classification tasks.


# # 6. Testing Prediction

# In[174]:


# In the preceding section, several classifiers were trained to classify customers. 
# Up to this stage, the analysis utilized data from the initial 10 months. 
# Now, in this section, the model is tested using the data from the last two months of the dataset, 
# which has been stored in the `test_set` dataframe.


# In[175]:


order_totals = test_set.copy(deep = True)


# In[176]:


# Initially, I grouped and reformatted this data using the same procedure as applied to the training set.
# However, I adjust the data to consider the time difference between the two datasets and normalize
# the variables count and sum to maintain consistency with the training set.


# In[177]:


# Grouping transactions per user based on CustomerID and 
# calculating various statistics such as count, minimum, maximum, mean, and sum of Cart Price.
transactions_per_user = order_totals.groupby(by=['CustomerID'])['Cart Price'].agg(['count','min','max','mean','sum'])

# Calculating the percentage of total Cart Price spent by each customer for each of the five categories.
for i in range(5):
    column_label = 'categ_{}'.format(i)
    transactions_per_user.loc[:,column_label] = order_totals.groupby(by=['CustomerID'])[column_label].sum() /\
                                            transactions_per_user['sum']*100

# Resetting the index to ensure the DataFrame is correctly formatted.
transactions_per_user.reset_index(drop = False, inplace = True)

# Calculating the sum of purchases in category 0 for each customer.
order_totals.groupby(by=['CustomerID'])['categ_0'].sum()

# Correcting the time range by multiplying the count of transactions and 
# the sum by a factor of 5 to account for the difference in time between datasets.
transactions_per_user['count'] = 5 * transactions_per_user['count']
transactions_per_user['sum']   = transactions_per_user['count'] * transactions_per_user['mean']

# Sorting the DataFrame by CustomerID in ascending order and displaying the first five rows.
transactions_per_user.sort_values('CustomerID', ascending = True)[:5]


# In[178]:


# Next, I convert the dataframe into a matrix and select only the variables that describe the category to which consumers belong. 
# At this stage, I apply the normalization method that was previously used on the training set.


# In[179]:


# Define the selected columns to be used in the matrix
selected_columns = ['count','min','max','mean','categ_0','categ_1','categ_2','categ_3','categ_4']

# Create a matrix from the selected columns in the transactions_per_user dataframe
matrix_test = transactions_per_user[selected_columns].to_numpy()

# Apply the same scaling transformation used on the training set to the test set matrix
scaled_test_matrix = scaler.transform(matrix_test)


# In[180]:


# Each row in this matrix represents a consumer's purchasing behavior. 
# At this stage, we use these behaviors to determine the category each consumer belongs to. 
# These categories were established in Section 4. 
# It's important to note that this step isn't the classification stage itself. 
# Here, we're preparing the test data by assigning categories to customers based on their purchasing habits. 
# However, this assignment is made using data collected over a 2-month period (via the 'count', 'min', 'max', and 'sum' variables). 
# The classifier defined in Section 5 uses a narrower set of variables, which are derived from the customer's first purchase.

# We utilize the available data over a 2-month period to assign categories to customers. 
# Then, we test the classifier by comparing its predictions with these categories. 
# To assign categories to customers, I utilize the instance of the kmeans method introduced in section 4. 
# The predict method of this instance calculates the distance of each customer from the centroids of the 11 customer classes, 
# and the smallest distance determines the customer's category.


# In[181]:


# Predicts the category labels for the test data using the K-means clustering model trained on the scaled test matrix.
Y = kmeans.predict(scaled_test_matrix)


# In[182]:


# To prepare for the execution of the classifier, we need to select the features on which it will operate.
columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4' ]
X = transactions_per_user[columns]


# In[186]:


# Now, i can proceed to evaluate the predictions made by the various classifiers trained in Section 5.


# In[183]:


# Define a list of tuples containing classifiers and their labels
classifiers = [(svc, 'Support Vector Machine'),
                (lr, 'Logistic Regression'),
                (knn, 'k-Nearest Neighbors'),
                (tr, 'Decision Tree'),
                (rf, 'Random Forest'),
                (gb, 'Gradient Boosting')]

# Iterate over each classifier and its label
for clf, label in classifiers:
    # Print the label
    print(30 * '_', '\n{}'.format(label))
    # Use the classifier to make predictions on the data
    clf.grid_predict(X, Y)


# In[184]:


# Finally, as mentioned in Section 5.8,
# we can enhance the classifier's performance by combining their predictions. 
# Here, I opted to blend predictions from Random Forest, Gradient Boosting, 
# and k-Nearest Neighbors as it results in a slight improvement in prediction accuracy.

predictions = votingC.predict(X)
print("Precision: {:.2f} % ".format(100*metrics.accuracy_score(Y, predictions)))


# # 7. Conclusion

# In[ ]:


# The analysis outlined in this notebook revolves around a dataset containing records of purchases made on an E-commerce platform
# spanning a year. Each entry in the dataset represents a product purchase by a specific customer on a particular date. 
# In total, the database comprises approximately 4000 clients. 
# Leveraging the provided data, the objective is to develop a classifier capable of predicting the type of purchase a customer 
# will make and the number of visits they will make within a year, starting from their initial visit to the E-commerce platform.


# In[ ]:


# The initial phase of this project involved categorizing the various products available on the website, 
# resulting in the creation of five primary product categories. Subsequently, 
# I conducted a customer classification based on their purchasing patterns over a span of 10 months. 
# Customers were classified into 11 major categories considering their typical product preferences, visit frequency, and 
# total expenditure during this period. With these categories defined, 
# the focus shifted to training multiple classifiers aimed at categorizing consumers into one of the 11 segments from their 
# initial purchase. 
# To achieve this, the classifier relies on five variables, namely:


# In[ ]:


# mean : amount of the basket of the current purchase
# categ_N with  N∈[0:4]: percentage spent in product category with index  N


# In[ ]:


# Finally, the effectiveness of the various classifiers' predictions was evaluated using data from the last two months of the dataset. 
# This evaluation involved a two-step process: 

# initially, all data from the two-month period was utilized to determine each client's category assignment, 
# followed by a comparison of classifier predictions with these assigned categories. 
# The analysis revealed that 75% of clients were correctly categorized, 
# indicating acceptable performance of the classifier despite potential limitations in the current model. 
# Notably, an unaddressed bias concerns the influence of seasonal purchasing patterns, 
# where buying behaviors may vary depending on the time of year (e.g., during holidays like Christmas). 
# This seasonal effect could lead to differences between categories defined over a 10-month period and those inferred from 
# the last two months. 
# To mitigate such biases, it would be advantageous to collect data over a longer timeframe.


# In[ ]:




