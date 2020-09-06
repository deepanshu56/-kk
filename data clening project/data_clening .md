# Data Cleaning: Airbnb Listings¶


```python
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import math
import pylab
import scipy.stats as stats
%matplotlib inline
```

# lests understand the data 


```python
cols = [
    'id',
    'host_id',
    'zipcode',
    'property_type',
    'room_type',
    'accommodates',
    'bedrooms',
    'beds',
    'bed_type',
    'price',
    'number_of_reviews',
    'review_scores_rating',
    'host_listing_count',
    'availability_30',
    'minimum_nights',
    'bathrooms'
]

data = pd.read_csv('listings.csv', usecols=cols)
```


```python
data.shape
```




    (27392, 16)




```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>zipcode</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>availability_30</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>host_listing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1069266</td>
      <td>5867023</td>
      <td>10022-4175</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>$160.00</td>
      <td>3</td>
      <td>21</td>
      <td>62</td>
      <td>86.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1846722</td>
      <td>2631556</td>
      <td>NaN</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>10</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>$105.00</td>
      <td>1</td>
      <td>28</td>
      <td>22</td>
      <td>85.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2061725</td>
      <td>4601412</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>$58.00</td>
      <td>3</td>
      <td>4</td>
      <td>35</td>
      <td>98.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44974</td>
      <td>198425</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>$185.00</td>
      <td>10</td>
      <td>1</td>
      <td>26</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4701675</td>
      <td>22590025</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>$195.00</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68914</td>
      <td>343302</td>
      <td>11231</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>$165.00</td>
      <td>2</td>
      <td>11</td>
      <td>16</td>
      <td>96.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4832596</td>
      <td>4148973</td>
      <td>11207</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>$80.00</td>
      <td>1</td>
      <td>29</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2562510</td>
      <td>13119459</td>
      <td>10013</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>$120.00</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>NaN</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3005360</td>
      <td>4421803</td>
      <td>10003</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>$150.00</td>
      <td>1</td>
      <td>30</td>
      <td>14</td>
      <td>96.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2431607</td>
      <td>4973668</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>$40.00</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>94.0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(data['zipcode'][data.zipcode.isnull()])
```




    162




```python
len(data.columns)
```




    16




```python
# check the number of missing values in each individual column

for col in data.columns:
    print (col + ', Number of Missing Values:', len(data[col][data[col].isnull()]))
```

    id, Number of Missing Values: 0
    host_id, Number of Missing Values: 0
    zipcode, Number of Missing Values: 162
    property_type, Number of Missing Values: 6
    room_type, Number of Missing Values: 0
    accommodates, Number of Missing Values: 0
    bathrooms, Number of Missing Values: 463
    bedrooms, Number of Missing Values: 140
    beds, Number of Missing Values: 98
    bed_type, Number of Missing Values: 0
    price, Number of Missing Values: 0
    minimum_nights, Number of Missing Values: 0
    availability_30, Number of Missing Values: 0
    number_of_reviews, Number of Missing Values: 0
    review_scores_rating, Number of Missing Values: 8657
    host_listing_count, Number of Missing Values: 0
    

# Remove NaN values from dataframe except review_scores_rating¶


```python
original = len(data)
data = data.dropna(how='any', subset=['zipcode', 'property_type', 'bedrooms', 'beds', 'bathrooms'])
print('Number of NaN values removed:', original - len(data))
```

    Number of NaN values removed: 769
    

# 2. Convert formatting for price from dollar1.00 into a float of 1.00¶

$1.00 ----> 1.00


```python
data['price'] = (data['price'].str.replace(r'[^-+\d.]', '').astype(float))

```

# 3. Drop any invalid values¶


```python
print ('Number of Accommodates 0:', len(data[data['accommodates'] == 0]))
print ('Number of Bedrooms 0:', len(data[data['bedrooms'] == 0]))
print ('Number of Beds 0:', len(data[data['beds'] == 0]))
print ('Number of Listings with Price $0.00:', len(data[data['price'] == 0.00]))

data = data[data['accommodates'] != 0]
data = data[data['bedrooms'] != 0]
data = data[data['beds'] != 0]
data = data[data['price'] != 0.00]
```

    Number of Accommodates 0: 0
    Number of Bedrooms 0: 2321
    Number of Beds 0: 0
    Number of Listings with Price $0.00: 0
    

# 4. Convert Zipcode to 5 digits¶


```python
data['zipcode'] = data['zipcode'].str.replace(r'-\d+', '')

```


```python
type(str.replace)
```




    method_descriptor




```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>zipcode</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>availability_30</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>host_listing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1069266</td>
      <td>5867023</td>
      <td>10022</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>160.0</td>
      <td>3</td>
      <td>21</td>
      <td>62</td>
      <td>86.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2061725</td>
      <td>4601412</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>58.0</td>
      <td>3</td>
      <td>4</td>
      <td>35</td>
      <td>98.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44974</td>
      <td>198425</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>185.0</td>
      <td>10</td>
      <td>1</td>
      <td>26</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4701675</td>
      <td>22590025</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>195.0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68914</td>
      <td>343302</td>
      <td>11231</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>165.0</td>
      <td>2</td>
      <td>11</td>
      <td>16</td>
      <td>96.0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.isnull().sum()
```




    id                         0
    host_id                    0
    zipcode                    0
    property_type              0
    room_type                  0
    accommodates               0
    bathrooms                  0
    bedrooms                   0
    beds                       0
    bed_type                   0
    price                      0
    minimum_nights             0
    availability_30            0
    number_of_reviews          0
    review_scores_rating    7712
    host_listing_count         0
    dtype: int64



# 5. Let's explore distribution of accommodates¶


```python
print('Number of Unique Accomodation: ', len(np.unique(data['beds'])))
```

    Number of Unique Accomodation:  15
    


```python
print('Number of Unique Accomodation: ', np.unique(data['accommodates']))
for i in range(1, 17):
    print('Accommodation {}:'.format(i), len(data[data['accommodates'] == i]))
```

    Number of Unique Accomodation:  [ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16]
    Accommodation 1: 2643
    Accommodation 2: 11400
    Accommodation 3: 2909
    Accommodation 4: 4278
    Accommodation 5: 982
    Accommodation 6: 1214
    Accommodation 7: 217
    Accommodation 8: 333
    Accommodation 9: 57
    Accommodation 10: 119
    Accommodation 11: 15
    Accommodation 12: 43
    Accommodation 13: 4
    Accommodation 14: 14
    Accommodation 15: 5
    Accommodation 16: 69
    


```python
data.groupby('accommodates').agg('count')['id']
```




    accommodates
    1      2643
    2     11400
    3      2909
    4      4278
    5       982
    6      1214
    7       217
    8       333
    9        57
    10      119
    11       15
    12       43
    13        4
    14       14
    15        5
    16       69
    Name: id, dtype: int64



# Visualize distribution of price, accommdations, beds, and review_scores_rating respectively¶


```python
plt.hist(data['accommodates'], bins=50)
plt.title("Histogram of Accommodations")
plt.xlabel("Number of Accommodations")
plt.ylabel("Frequency")
plt.show()
```


![png](output_25_0.png)


We see that a majority of listings have accomodations for 1-4 people. 1 bed typically accomodates 2 individuals, so let's plot beds instead to analyze how many of the listings are single bedroom listings.


```python
# explore distribution of beds

print('Number of Unique Beds: ', np.unique(data['beds']))
for i in range(1, 17):
    print('Beds {}:'.format(i), len(data[data['beds'] == i]))
```

    Number of Unique Beds:  [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 16.]
    Beds 1: 16002
    Beds 2: 5418
    Beds 3: 1770
    Beds 4: 610
    Beds 5: 243
    Beds 6: 117
    Beds 7: 41
    Beds 8: 22
    Beds 9: 3
    Beds 10: 20
    Beds 11: 4
    Beds 12: 9
    Beds 13: 1
    Beds 14: 15
    Beds 15: 0
    Beds 16: 27
    


```python
# Visualize the distribution of beds
plt.hist(data['beds'], bins=50)
plt.title("Histogram of Beds")
plt.xlabel("Bed Count")
plt.ylabel("Frequency")
plt.show()
```


![png](output_28_0.png)



```python
# visualize distribution of review scores ratings
plt.hist(data['review_scores_rating'][~data['review_scores_rating'].isnull()])
plt.title("Histogram of Review Scores Ratings")
plt.xlabel("Review Score")
plt.ylabel("Frequency")
plt.show()
```


![png](output_29_0.png)



```python
idx_vals = data['review_scores_rating'][data['number_of_reviews'] == 0].index.values.tolist()
data.loc[idx_vals, 'review_scores_rating'] = data['review_scores_rating'][data['number_of_reviews'] == 0].replace(np.nan, 'No Reviews')
```


```python
data.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>zipcode</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>availability_30</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>host_listing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1069266</td>
      <td>5867023</td>
      <td>10022</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>160.0</td>
      <td>3</td>
      <td>21</td>
      <td>62</td>
      <td>86.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2061725</td>
      <td>4601412</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>58.0</td>
      <td>3</td>
      <td>4</td>
      <td>35</td>
      <td>98.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44974</td>
      <td>198425</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>185.0</td>
      <td>10</td>
      <td>1</td>
      <td>26</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4701675</td>
      <td>22590025</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>195.0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68914</td>
      <td>343302</td>
      <td>11231</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>165.0</td>
      <td>2</td>
      <td>11</td>
      <td>16</td>
      <td>96.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3005360</td>
      <td>4421803</td>
      <td>10003</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>150.0</td>
      <td>1</td>
      <td>30</td>
      <td>14</td>
      <td>96.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2431607</td>
      <td>4973668</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>40.0</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>94.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>234327</td>
      <td>652642</td>
      <td>10018</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>Real Bed</td>
      <td>80.0</td>
      <td>3</td>
      <td>30</td>
      <td>7</td>
      <td>80.0</td>
      <td>25</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2000287</td>
      <td>10303472</td>
      <td>11213</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>110.0</td>
      <td>1</td>
      <td>29</td>
      <td>26</td>
      <td>91.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2525956</td>
      <td>7365834</td>
      <td>10019</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>189.0</td>
      <td>19</td>
      <td>30</td>
      <td>7</td>
      <td>86.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>15</th>
      <td>809929</td>
      <td>4260203</td>
      <td>10014</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>200.0</td>
      <td>2</td>
      <td>4</td>
      <td>8</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>753622</td>
      <td>869880</td>
      <td>10003</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>95.0</td>
      <td>5</td>
      <td>30</td>
      <td>6</td>
      <td>90.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1781041</td>
      <td>9348914</td>
      <td>10040</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>125.0</td>
      <td>1</td>
      <td>30</td>
      <td>2</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>4060165</td>
      <td>1860033</td>
      <td>10033</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>35.0</td>
      <td>7</td>
      <td>30</td>
      <td>6</td>
      <td>73.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>4323940</td>
      <td>4344968</td>
      <td>11238</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>165.0</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>80.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1832976</td>
      <td>724988</td>
      <td>10038</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>100.0</td>
      <td>2</td>
      <td>0</td>
      <td>30</td>
      <td>90.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>4218098</td>
      <td>21892444</td>
      <td>10027</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>110.0</td>
      <td>2</td>
      <td>6</td>
      <td>2</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4053471</td>
      <td>3967335</td>
      <td>11222</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>65.0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>95.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2596813</td>
      <td>13298061</td>
      <td>10025</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>150.0</td>
      <td>2</td>
      <td>12</td>
      <td>11</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>312664</td>
      <td>5926</td>
      <td>10030</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>10</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>Real Bed</td>
      <td>255.0</td>
      <td>1</td>
      <td>3</td>
      <td>33</td>
      <td>88.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>28</th>
      <td>26229</td>
      <td>64206</td>
      <td>10003</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>299.0</td>
      <td>5</td>
      <td>29</td>
      <td>37</td>
      <td>92.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3414741</td>
      <td>10451037</td>
      <td>10038</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>365.0</td>
      <td>2</td>
      <td>30</td>
      <td>1</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1006877</td>
      <td>72062</td>
      <td>10009</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>225.0</td>
      <td>1</td>
      <td>29</td>
      <td>13</td>
      <td>95.0</td>
      <td>10</td>
    </tr>
    <tr>
      <th>33</th>
      <td>798391</td>
      <td>3381805</td>
      <td>10031</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>90.0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>91.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>3235070</td>
      <td>14337132</td>
      <td>10026</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>125.0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2677222</td>
      <td>13707331</td>
      <td>10027</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Futon</td>
      <td>109.0</td>
      <td>2</td>
      <td>30</td>
      <td>20</td>
      <td>93.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>39</th>
      <td>2407196</td>
      <td>5905011</td>
      <td>10030</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>89.0</td>
      <td>2</td>
      <td>0</td>
      <td>27</td>
      <td>97.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>3574160</td>
      <td>4111729</td>
      <td>10128</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>250.0</td>
      <td>1</td>
      <td>25</td>
      <td>4</td>
      <td>95.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>45</th>
      <td>783964</td>
      <td>2347382</td>
      <td>11101</td>
      <td>House</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>66.0</td>
      <td>2</td>
      <td>26</td>
      <td>33</td>
      <td>92.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>47</th>
      <td>2769628</td>
      <td>3464208</td>
      <td>10031</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>70.0</td>
      <td>4</td>
      <td>14</td>
      <td>19</td>
      <td>88.0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>48</th>
      <td>2568017</td>
      <td>9113389</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>35.0</td>
      <td>1</td>
      <td>22</td>
      <td>10</td>
      <td>80.0</td>
      <td>12</td>
    </tr>
    <tr>
      <th>50</th>
      <td>3595080</td>
      <td>1606560</td>
      <td>10014</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>125.0</td>
      <td>3</td>
      <td>30</td>
      <td>3</td>
      <td>93.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>53</th>
      <td>995373</td>
      <td>5467851</td>
      <td>10028</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>5</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>199.0</td>
      <td>3</td>
      <td>28</td>
      <td>37</td>
      <td>93.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>3593884</td>
      <td>3371804</td>
      <td>11206</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>140.0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>100.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>59</th>
      <td>1906230</td>
      <td>6654717</td>
      <td>11211</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>5</td>
      <td>1.5</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>275.0</td>
      <td>7</td>
      <td>0</td>
      <td>3</td>
      <td>99.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>857948</td>
      <td>4431637</td>
      <td>10025</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>1</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>150.0</td>
      <td>2</td>
      <td>30</td>
      <td>1</td>
      <td>94.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>403666</td>
      <td>101689</td>
      <td>11215</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>139.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>93.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1973503</td>
      <td>616927</td>
      <td>10007</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>275.0</td>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>95.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>63</th>
      <td>843499</td>
      <td>350553</td>
      <td>10027</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>60.0</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>80.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>64</th>
      <td>111308</td>
      <td>97476</td>
      <td>11211</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>60.0</td>
      <td>4</td>
      <td>18</td>
      <td>71</td>
      <td>94.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>2056723</td>
      <td>9215509</td>
      <td>11205</td>
      <td>House</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>150.0</td>
      <td>1</td>
      <td>30</td>
      <td>14</td>
      <td>94.0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2842403</td>
      <td>14538812</td>
      <td>10030</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>60.0</td>
      <td>2</td>
      <td>27</td>
      <td>16</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>68</th>
      <td>4332961</td>
      <td>1153352</td>
      <td>10002</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>145.0</td>
      <td>2</td>
      <td>29</td>
      <td>4</td>
      <td>95.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>69</th>
      <td>1846580</td>
      <td>9646158</td>
      <td>11215</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>65.0</td>
      <td>2</td>
      <td>30</td>
      <td>27</td>
      <td>88.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>67145</td>
      <td>330660</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>125.0</td>
      <td>3</td>
      <td>17</td>
      <td>17</td>
      <td>94.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>67272</td>
      <td>331589</td>
      <td>11217</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>130.0</td>
      <td>2</td>
      <td>0</td>
      <td>41</td>
      <td>96.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>3344643</td>
      <td>921046</td>
      <td>10012</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>225.0</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>100.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74</th>
      <td>1509665</td>
      <td>4132505</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>219.0</td>
      <td>1</td>
      <td>0</td>
      <td>17</td>
      <td>97.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>75</th>
      <td>2351967</td>
      <td>12010916</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>209.0</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>87.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>511887</td>
      <td>1264213</td>
      <td>10002</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>250.0</td>
      <td>5</td>
      <td>29</td>
      <td>2</td>
      <td>97.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove inconsistent NaN values
data = data[~data['review_scores_rating'].isnull()]
```


```python
# ensure all zipcodes are of length 5
data = data[data['zipcode'].map(len) == 5]
```


```python
data = data[data['zipcode'].apply(len) == 5]
```

# Convert review_scores_rating into different buckets¶


```python
def convert_scores_buckets(val):
    if val == 'No Reviews':
        return 'No Reviews'
    elif val >= 95.0:
        return '95-100'
    elif val >= 90.0 and val < 95.0:
        return '90-94'
    elif val >= 85.0 and val < 90.0:
        return '85-89'
    elif val >= 80.0 and val < 85.0:
        return '80-84'
    elif val >= 70.0 and val < 80.0:
        return '70-79'
    elif val >= 60.0 and val < 70.0:
        return '60-69'
    elif val >= 50.0 and val < 60.0:
        return '50-59'
    elif val >= 40.0 and val < 50.0:
        return '40-49'
    elif val >= 30.0 and val < 40.0:
        return '30-39'
    elif val >= 20.0 and val < 30.0:
        return '20-29'
    elif val >= 10.0 and val < 20.0:
        return '10-19'
    elif val < 10.0:
        return '0-9'
```


```python
data['review_scores_rating'] = data['review_scores_rating'].apply(convert_scores_buckets)
print ('Unique Values in the Column:', np.unique(data['review_scores_rating']))
```

    Unique Values in the Column: ['20-29' '30-39' '40-49' '50-59' '60-69' '70-79' '80-84' '85-89' '90-94'
     '95-100']
    


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>zipcode</th>
      <th>property_type</th>
      <th>room_type</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>bed_type</th>
      <th>price</th>
      <th>minimum_nights</th>
      <th>availability_30</th>
      <th>number_of_reviews</th>
      <th>review_scores_rating</th>
      <th>host_listing_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1069266</td>
      <td>5867023</td>
      <td>10022</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>160.0</td>
      <td>3</td>
      <td>21</td>
      <td>62</td>
      <td>85-89</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2061725</td>
      <td>4601412</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>58.0</td>
      <td>3</td>
      <td>4</td>
      <td>35</td>
      <td>95-100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44974</td>
      <td>198425</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>185.0</td>
      <td>10</td>
      <td>1</td>
      <td>26</td>
      <td>95-100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4701675</td>
      <td>22590025</td>
      <td>10011</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>195.0</td>
      <td>1</td>
      <td>30</td>
      <td>1</td>
      <td>95-100</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68914</td>
      <td>343302</td>
      <td>11231</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>6</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>Real Bed</td>
      <td>165.0</td>
      <td>2</td>
      <td>11</td>
      <td>16</td>
      <td>95-100</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3005360</td>
      <td>4421803</td>
      <td>10003</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>4</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>150.0</td>
      <td>1</td>
      <td>30</td>
      <td>14</td>
      <td>95-100</td>
      <td>4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2431607</td>
      <td>4973668</td>
      <td>11221</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>40.0</td>
      <td>4</td>
      <td>0</td>
      <td>10</td>
      <td>90-94</td>
      <td>4</td>
    </tr>
    <tr>
      <th>12</th>
      <td>234327</td>
      <td>652642</td>
      <td>10018</td>
      <td>Apartment</td>
      <td>Shared room</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>Real Bed</td>
      <td>80.0</td>
      <td>3</td>
      <td>30</td>
      <td>7</td>
      <td>80-84</td>
      <td>25</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2000287</td>
      <td>10303472</td>
      <td>11213</td>
      <td>Apartment</td>
      <td>Private room</td>
      <td>2</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Real Bed</td>
      <td>110.0</td>
      <td>1</td>
      <td>29</td>
      <td>26</td>
      <td>90-94</td>
      <td>3</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2525956</td>
      <td>7365834</td>
      <td>10019</td>
      <td>Apartment</td>
      <td>Entire home/apt</td>
      <td>3</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>Real Bed</td>
      <td>189.0</td>
      <td>19</td>
      <td>30</td>
      <td>7</td>
      <td>85-89</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



# Encode categorical variables¶


```python
property_dummies = pd.get_dummies(data['property_type'])
room_dummies = pd.get_dummies(data['room_type'])
bed_dummies = pd.get_dummies(data['bed_type'])

```

# Replace the old columns with our new one-hot encoded ones¶



```python
df = pd.concat((data.drop(['property_type', 'room_type', 'bed_type'], axis=1), \
     property_dummies.astype(int), room_dummies.astype(int), bed_dummies.astype(int)), \
     axis=1)

print ('Number of Columns:', len(df.columns))
```

    Number of Columns: 36
    

# Move target predictor 'price' to the end of the dataframe


```python
cols = list(df.columns.values)
idx = cols.index('price')
rearrange_cols = cols[:idx] + cols[idx+1:] + [cols[idx]]
df = df[rearrange_cols]
```

# Convert non-categorical variables to floats and normalize¶


```python
def normalize(col):
    mean = np.mean(col)
    std = np.std(col)
    return col.apply(lambda x: (x - mean) / std)

non_cat_vars = ['accommodates', 'bedrooms', 'beds', 'number_of_reviews', 'host_listing_count', 'availability_30', 'minimum_nights', 'bathrooms']
for col in non_cat_vars:
    df[col] = df[col].astype(float)
    df[col] = normalize(df[col])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>host_id</th>
      <th>zipcode</th>
      <th>accommodates</th>
      <th>bathrooms</th>
      <th>bedrooms</th>
      <th>beds</th>
      <th>minimum_nights</th>
      <th>availability_30</th>
      <th>number_of_reviews</th>
      <th>...</th>
      <th>Villa</th>
      <th>Entire home/apt</th>
      <th>Private room</th>
      <th>Shared room</th>
      <th>Airbed</th>
      <th>Couch</th>
      <th>Futon</th>
      <th>Pull-out Sofa</th>
      <th>Real Bed</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1069266</td>
      <td>5867023</td>
      <td>10022</td>
      <td>-0.550907</td>
      <td>-0.317717</td>
      <td>-0.407726</td>
      <td>-0.490785</td>
      <td>0.215485</td>
      <td>0.337183</td>
      <td>2.204784</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2061725</td>
      <td>4601412</td>
      <td>11221</td>
      <td>-0.550907</td>
      <td>-0.317717</td>
      <td>-0.407726</td>
      <td>0.347334</td>
      <td>0.215485</td>
      <td>-1.067475</td>
      <td>0.937086</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44974</td>
      <td>198425</td>
      <td>10011</td>
      <td>-0.550907</td>
      <td>-0.317717</td>
      <td>-0.407726</td>
      <td>-0.490785</td>
      <td>3.250752</td>
      <td>-1.315356</td>
      <td>0.514519</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>185.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4701675</td>
      <td>22590025</td>
      <td>10011</td>
      <td>-0.550907</td>
      <td>-0.317717</td>
      <td>-0.407726</td>
      <td>0.347334</td>
      <td>-0.651735</td>
      <td>1.080826</td>
      <td>-0.659276</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>195.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>68914</td>
      <td>343302</td>
      <td>11231</td>
      <td>1.654450</td>
      <td>-0.317717</td>
      <td>1.362218</td>
      <td>1.185454</td>
      <td>-0.218125</td>
      <td>-0.489087</td>
      <td>0.045001</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>165.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
# visualize distribution of price (target variable)
plt.hist(df['price'], bins=50)
plt.title("Histogram of Pricing")
plt.xlabel("Pricing (USD) Per Day")
plt.ylabel("Frequency")
plt.show()
```


![png](output_48_0.png)



```python
# Histogram of Pricing indicates Pricing is pretty skewed
# log transform the response 'price'
df['price_log'] = df['price'].apply(lambda x: math.log(x))

plt.hist(df['price_log'], bins=30)
plt.title("Histogram of Pricing Log-Transformed")
plt.xlabel("Pricing (USD) Per Day")
plt.ylabel("Frequency")
plt.show()
```


![png](output_49_0.png)



```python
# read to csv
df.to_csv('output.csv')
```


```python

```
