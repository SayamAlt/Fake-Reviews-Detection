{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "f57dadd2",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\hp\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping corpora\\wordnet.zip.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "from nltk.corpus import stopwords\n",
    "from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.model_selection import train_test_split\n",
    "import string, nltk\n",
    "from nltk import word_tokenize\n",
    "from nltk.stem import PorterStemmer\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "nltk.download('wordnet')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "335d77c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package omw-1.4 to\n",
      "[nltk_data]     C:\\Users\\hp\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping corpora\\omw-1.4.zip.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nltk.download('omw-1.4')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ccecce80",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>category</th>\n",
       "      <th>rating</th>\n",
       "      <th>label</th>\n",
       "      <th>text_</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Love this!  Well made, sturdy, and very comfor...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>love it, a great upgrade from the original.  I...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>This pillow saved my back. I love the look and...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>1.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Missing information on how to use it, but it i...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Home_and_Kitchen_5</td>\n",
       "      <td>5.0</td>\n",
       "      <td>CG</td>\n",
       "      <td>Very nice set. Good quality. We have had the s...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             category  rating label  \\\n",
       "0  Home_and_Kitchen_5     5.0    CG   \n",
       "1  Home_and_Kitchen_5     5.0    CG   \n",
       "2  Home_and_Kitchen_5     5.0    CG   \n",
       "3  Home_and_Kitchen_5     1.0    CG   \n",
       "4  Home_and_Kitchen_5     5.0    CG   \n",
       "\n",
       "                                               text_  \n",
       "0  Love this!  Well made, sturdy, and very comfor...  \n",
       "1  love it, a great upgrade from the original.  I...  \n",
       "2  This pillow saved my back. I love the look and...  \n",
       "3  Missing information on how to use it, but it i...  \n",
       "4  Very nice set. Good quality. We have had the s...  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('fake reviews dataset.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "1718eda3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "category    0\n",
       "rating      0\n",
       "label       0\n",
       "text_       0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0ff175e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 40432 entries, 0 to 40431\n",
      "Data columns (total 4 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   category  40432 non-null  object \n",
      " 1   rating    40432 non-null  float64\n",
      " 2   label     40432 non-null  object \n",
      " 3   text_     40432 non-null  object \n",
      "dtypes: float64(1), object(3)\n",
      "memory usage: 1.2+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7185834d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>40432.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>4.256579</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>1.144354</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>5.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             rating\n",
       "count  40432.000000\n",
       "mean       4.256579\n",
       "std        1.144354\n",
       "min        1.000000\n",
       "25%        4.000000\n",
       "50%        5.000000\n",
       "75%        5.000000\n",
       "max        5.000000"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "30a99867",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "5.0    24559\n",
       "4.0     7965\n",
       "3.0     3786\n",
       "1.0     2155\n",
       "2.0     1967\n",
       "Name: rating, dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['rating'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "57789974",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x576 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(15,8))\n",
    "labels = df['rating'].value_counts().keys()\n",
    "values = df['rating'].value_counts().values\n",
    "explode = (0.1,0,0,0,0)\n",
    "plt.pie(values,labels=labels,explode=explode,shadow=True,autopct='%1.1f%%')\n",
    "plt.title('Proportion of each rating',fontweight='bold',fontsize=25,pad=20,color='crimson')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7361e4ee",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    nopunc = [w for w in text if w not in string.punctuation]\n",
    "    nopunc = ''.join(nopunc)\n",
    "    return  ' '.join([word for word in nopunc.split() if word.lower() not in stopwords.words('english')])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "119ab6a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('Love this!  Well made, sturdy, and very comfortable.  I love it!Very pretty',\n",
       " 'Love Well made sturdy comfortable love itVery pretty')"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['text_'][0], clean_text(df['text_'][0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "662fdd78",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    Love Well made sturdy comfortable love itVery ...\n",
       "1    love great upgrade original Ive mine couple years\n",
       "2              pillow saved back love look feel pillow\n",
       "3          Missing information use great product price\n",
       "4                 nice set Good quality set two months\n",
       "Name: text_, dtype: object"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['text_'].head().apply(clean_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "f7274eaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(40432, 4)"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "09b11a2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df['text_'] = df['text_'].apply(clean_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "62a6f1f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'] = df['text_'].astype(str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "583d2a53",
   "metadata": {},
   "outputs": [],
   "source": [
    "def preprocess(text):\n",
    "    return ' '.join([word for word in word_tokenize(text) if word not in stopwords.words('english') and not word.isdigit() and word not in string.punctuation])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "0ef65bf7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'Very nice set Good quality We set two months'"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "preprocess(df['text_'][4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "89b7578c",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'][:10000] = df['text_'][:10000].apply(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "785cd44a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'][10001:20000] = df['text_'][10001:20000].apply(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "59d7b1d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'][20001:30000] = df['text_'][20001:30000].apply(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "5fc6ca33",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'][30001:40000] = df['text_'][30001:40000].apply(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "2c0eed68",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'][40001:40432] = df['text_'][40001:40432].apply(preprocess)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4d858b63",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['text_'] = df['text_'].str.lower()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "0021efd8",
   "metadata": {},
   "outputs": [],
   "source": [
    "stemmer = PorterStemmer()\n",
    "def stem_words(text):\n",
    "    return ' '.join([stemmer.stem(word) for word in text.split()])\n",
    "df['text_'] = df['text_'].apply(lambda x: stem_words(x))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "b96d3e8a",
   "metadata": {},
   "outputs": [],
   "source": [
    "lemmatizer = WordNetLemmatizer()\n",
    "def lemmatize_words(text):\n",
    "    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])\n",
    "df[\"text_\"] = df[\"text_\"].apply(lambda text: lemmatize_words(text))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "89598dfa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    love well made sturdi comfort i love veri pretti\n",
       "1      love great upgrad origin i 've mine coupl year\n",
       "2        thi pillow save back i love look feel pillow\n",
       "3               miss inform use great product price i\n",
       "4         veri nice set good qualiti we set two month\n",
       "Name: text_, dtype: object"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['text_'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "ea1290cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv('Preprocessed Fake Reviews Detection Dataset.csv')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
