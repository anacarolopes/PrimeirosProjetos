{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h1><center><strong>Projeto 2 - Titanic</strong></center></h1>\n",
    "\n",
    "<h2>Introdução</h2>\n",
    "    \n",
    "<h6><p>Este projeto tem como objetivo propor pontos a serem investigados referentes as chances de sobrevivência dos passageiros do Titanic.</p>\n",
    "    <p>Através da análise de dados iremos obter estatísticas relevantes a investigação proposta.</p>\n",
    "    <p>Através da análise desenvolvida, tentaremos identificar se a taxa de sobrevivência dos passageiros foi influenciada por fatores como:</p>\n",
    "    <ul>\n",
    "        <li>Gênero;</li>\n",
    "        <li>Classe Social;</li>\n",
    "        <li>Idade.</li>\n",
    "    </ul>\n",
    "    <p>Para tanto, as perguntas a serem respondidas na análise são as seguintes:</p>\n",
    "    <ul>\n",
    "        <li><strong>As mulheres tiveram maior chance de sobrevivência do que os homens?</strong></li>\n",
    "        <li><strong>Crianças tiveram uma taxa de sobrevivência maior que as demais faixas etárias?</strong></li>\n",
    "        <li><strong>Os passageiros das primeiras classes tiveram maiores chances de sobrevivência do que os passageiros das segunda e terceira classes?</strong></li>\n",
    "    </ul>\n",
    "    Primeiro, vamos carregar o conjunto dos dados do site <a href=\"https://www.kaggle.com/c/titanic/data\">Kaggle</a> e fazer uma verificação inicial, verificando como os dados estão estruturados:</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
   ],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import warnings\n",
    "import math\n",
    "df_titanic = pd.read_csv('titanic_data.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h4><p>VARIÁVEIS:</p></h4>\n",
    "<h6><p>Survived = Sobreviveu: 0 = Não, 1 = Sim</p>\n",
    "<p>Pclass = Classe: Classe de ingresso 1 = 1º, 2 = 2º, 3 = 3º</p>\n",
    "<p>Sex = Sexo: Sexo do passageiro</p>\n",
    "<p>Age = Idade: Idade em anos</p>\n",
    "<p>Sibsp = Quantidade de irmãos e/ou cônjuges a bordo do Titanic</p>\n",
    "<p>Parch = Quantidade de pais e/ou crianças a bordo do Titanic</p>\n",
    "<p>Ticket = Bilhete: Número do bilhete de embarque</p>\n",
    "<p>Fare = Tarifa: Tarifa paga pelo Passageiro</p>\n",
    "<p>Cabin = Cabine: Número de cabine</p>\n",
    "<p>Embarked = Embarque: Porto de Embarque (C = Cherbourg, Q = Queenstown, S = Southampton)</p>\n",
    "<p>Notas:</p>\n",
    "<p>Pclass = Classe: 1º = Superior 2º = Médio 3º = inferior</p>\n",
    "    <p>Age = Idade: A idade informda é fracionada se for inferior a 1 ano. Se for uma idade estimada, esta será na forma de xx.5</p></br>\n",
    "    Sibsp - definição das relações familiares a seguir:</br>\n",
    "Sibling = Irmão, irmã, meio-irmão, irmandade</br>\n",
    "Spouse = Cônjuge que é definido por marido ou esposa</br></br>\n",
    "Parch - definição das relações familiares a seguir:</br>\n",
    "Parent (Pais) = mãe, pai</br>\n",
    "Child (Criança) = filha, filho, enteada, enteado</br>\n",
    "Algumas crianças viajaram apenas com uma babá, portanto, parch = 0 para elas</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 3,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificando os primeiros registros da planilha\n",
    "df_titanic.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>886</th>\n",
       "      <td>887</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>Montvila, Rev. Juozas</td>\n",
       "      <td>male</td>\n",
       "      <td>27.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>211536</td>\n",
       "      <td>13.00</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>887</th>\n",
       "      <td>888</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Graham, Miss. Margaret Edith</td>\n",
       "      <td>female</td>\n",
       "      <td>19.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>112053</td>\n",
       "      <td>30.00</td>\n",
       "      <td>B42</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>888</th>\n",
       "      <td>889</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Johnston, Miss. Catherine Helen \"Carrie\"</td>\n",
       "      <td>female</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>W./C. 6607</td>\n",
       "      <td>23.45</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>889</th>\n",
       "      <td>890</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Behr, Mr. Karl Howell</td>\n",
       "      <td>male</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>111369</td>\n",
       "      <td>30.00</td>\n",
       "      <td>C148</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>890</th>\n",
       "      <td>891</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Dooley, Mr. Patrick</td>\n",
       "      <td>male</td>\n",
       "      <td>32.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>370376</td>\n",
       "      <td>7.75</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Q</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     PassengerId  Survived  Pclass                                      Name  \\\n",
       "886          887         0       2                     Montvila, Rev. Juozas   \n",
       "887          888         1       1              Graham, Miss. Margaret Edith   \n",
       "888          889         0       3  Johnston, Miss. Catherine Helen \"Carrie\"   \n",
       "889          890         1       1                     Behr, Mr. Karl Howell   \n",
       "890          891         0       3                       Dooley, Mr. Patrick   \n",
       "\n",
       "        Sex   Age  SibSp  Parch      Ticket   Fare Cabin Embarked  \n",
       "886    male  27.0      0      0      211536  13.00   NaN        S  \n",
       "887  female  19.0      0      0      112053  30.00   B42        S  \n",
       "888  female   NaN      1      2  W./C. 6607  23.45   NaN        S  \n",
       "889    male  26.0      0      0      111369  30.00  C148        C  \n",
       "890    male  32.0      0      0      370376   7.75   NaN        Q  "
      ]
     },
     "execution_count": 4,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificando os últimos registros da planilha\n",
    "df_titanic.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Vemos através dos últimos registro que a coluna que contém a idade (Age) dos passageiros apresenta informação \"NaN\" ao invés de trazer um valor. Isto significa que a coluna apresenta dados nulos.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId      int64\n",
       "Survived         int64\n",
       "Pclass           int64\n",
       "Name            object\n",
       "Sex             object\n",
       "Age            float64\n",
       "SibSp            int64\n",
       "Parch            int64\n",
       "Ticket          object\n",
       "Fare           float64\n",
       "Cabin           object\n",
       "Embarked        object\n",
       "dtype: object"
      ]
     },
     "execution_count": 5,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificando o tipo de dado em cada coluna\n",
    "df_titanic.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(891, 12)"
      ]
     },
     "execution_count": 6,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificando o tamanho total do dataset\n",
    "df_titanic.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    891\n",
       "Survived       891\n",
       "Pclass         891\n",
       "Name           891\n",
       "Sex            891\n",
       "Age            714\n",
       "SibSp          891\n",
       "Parch          891\n",
       "Ticket         891\n",
       "Fare           891\n",
       "Cabin          204\n",
       "Embarked       889\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificando o número de linhas em cada coluna\n",
    "df_titanic.count()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Com base nas duas últimas verificações, identificamos que o dataset possui 891 linhas e 12 colunas, ou seja, contém a informação de 891 passageiros. Analisando o número de linhas em cada coluna, identificamos que na coluna \"Age\" há apenas 714 registros dos 891 passageiros, o que significa que 177 passageiros não possuem a informação de sua idade no registro. O mesmo ocorre nas colunas \"Cabin\" e \"Embarked\", mas para as nossas análises, estas duas últimas colunas não serão relevantes.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass     Sex   Age  SibSp  Parch\n",
       "0            1         0       3    male  22.0      1      0\n",
       "1            2         1       1  female  38.0      1      0\n",
       "2            3         1       3  female  26.0      0      0\n",
       "3            4         1       1  female  35.0      1      0\n",
       "4            5         0       3    male  35.0      0      0"
      ]
     },
     "execution_count": 8,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# eliminando as coluna irrelevantes para a análise\n",
    "df_titanic_1 = df_titanic.drop(['Name','Ticket','Cabin','Fare','Embarked'], axis=1)\n",
    "df_titanic_1.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
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
       "      <th>PassengerId</th>\n",
       "      <th>Sobreviveu</th>\n",
       "      <th>Classe</th>\n",
       "      <th>Sexo</th>\n",
       "      <th>Idade</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Sobreviveu  Classe    Sexo  Idade  SibSp  Parch\n",
       "0            1           0       3    male   22.0      1      0\n",
       "1            2           1       1  female   38.0      1      0\n",
       "2            3           1       3  female   26.0      0      0\n",
       "3            4           1       1  female   35.0      1      0\n",
       "4            5           0       3    male   35.0      0      0"
      ]
     },
     "execution_count": 9,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# renomeando as colunas restantes\n",
    "df_titanic_1.columns = ['PassengerId', 'Sobreviveu', 'Classe', 'Sexo', 'Idade', 'SibSp',\n",
    "              'Parch']\n",
    "df_titanic_1.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Após eliminarmos as colunas irrelevantes e renomeá-las para melhor entendimento, vamos tratar agora dos valores nulos que encontramos na coluna de \"Idade\".</br>\n",
    "Sabemos que não podemos simplesmente eliminar a coluna ou as linnhas que contém valores nulos, pois isso eliminará também outros dados relevantes para nossa análise. Nesse caso, podemos então substituir os valores faltantes por uma média de idade.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "29\n"
     ]
    }
   ],
   "source": [
    "# verificando a idade média dos passageiros, arredondando o valor e exibindo o valor encontrado\n",
    "idademedia = df_titanic_1['Idade'].mean()\n",
    "idademedia = math.floor(idademedia)\n",
    "print(idademedia)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>A idade média encontrada foi 29 anos. Agora efetuaremos a substituição dos valores nulos por este.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PassengerId    0\n",
       "Sobreviveu     0\n",
       "Classe         0\n",
       "Sexo           0\n",
       "Idade          0\n",
       "SibSp          0\n",
       "Parch          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# substituindo os valores\n",
    "df_titanic_1.update(df_titanic_1['Idade'].fillna(idademedia))\n",
    "\n",
    "# verificando a somatória dos valores nulos para conferência\n",
    "df_titanic_1.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Após efetuarmos a limpeza do dataset e deixar os dados consistentes, podemos iniciar a análise.</h6></br>\n",
    "<h6>Primeiramente, vamos identificar o percentual de pessoas que sobreviveram e que não sobreviveram.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0,0.5,'')"
      ]
     },
     "execution_count": 12,
     "metadata": {
     },
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "f3bce401cce7c68a0caa03214a146889354788de",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f823cbcff98>"
      ]
     },
     "execution_count": 12,
     "metadata": {
      "image/png": {
       "height": 249,
       "width": 504
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# gráfico para a representação da porcentagem de sobreviventes e não sobreviventes\n",
    "df_titanic_1['Sobreviveu'].value_counts().plot.pie(\n",
    "    colors=('tab:red', 'tab:blue'),\n",
    "    title='Sobreviventes x Não sobreviventes',\n",
    "    fontsize=13,\n",
    "    shadow=True,\n",
    "    startangle=90,\n",
    "    autopct='%1.1f%%',\n",
    "    labels=('Não sobreviventes', 'Sobreviventes'),\n",
    "    figsize=(6, 4)).set_ylabel('')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Do total de 891 passageiros contidos no dataset, observamos que mais da metade não sobreviveu. Esta porcentagem representa 549 passageiros aproximadamente.</h6></br>\n",
    "<h6>A seguir vamos separar os passageiros por \"Sexo\" e verificar a contagem dos dados através do gráfico.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Homens: \n",
      "577\n",
      "Mulheres: \n",
      "314\n"
     ]
    },
    {
     "data": {
      "image/png": "702a6b53c2903a89e8298cdac235a617a243ca48",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f822db0e6a0>"
      ]
     },
     "execution_count": 13,
     "metadata": {
      "image/png": {
       "height": 365,
       "width": 364
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# verificação da quantidade dos passageiros\n",
    "homemData = df_titanic_1[df_titanic_1.Sexo == \"male\"]\n",
    "mulherData = df_titanic_1[df_titanic_1.Sexo == \"female\"]\n",
    "\n",
    "print(\"Homens: \")\n",
    "print(homemData.count()[\"Sexo\"])\n",
    "\n",
    "print(\"Mulheres: \")\n",
    "print(mulherData.count()[\"Sexo\"])\n",
    "\n",
    "# gráfico para a representação da quantidade de acordo com o gênero\n",
    "sns.set_style(\"whitegrid\")\n",
    "g = sns.catplot(x='Sexo', data = df_titanic_1, kind='count')\n",
    "g.despine(left=True)\n",
    "g.set_xlabels(u\"Gênero\")\n",
    "plt.title(u\"Quantidade de passageiros por Sexo\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>A maioria dos passageiros do dataset eram homens.</h6></br>\n",
    "<h6>Verificaremos então a influência do gênero sobre o fator sobrevivência.</h6></b>\n",
    "<h6><strong>As mulheres tiveram maior chance de sobrevivência do que os homens?</strong></h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0,0.5,'Quantidade')"
      ]
     },
     "execution_count": 14,
     "metadata": {
     },
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "d2a96a79f200895aced62287e358a62e4f2e1bf3",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f822db45eb8>"
      ]
     },
     "execution_count": 14,
     "metadata": {
      "image/png": {
       "height": 276,
       "width": 388
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# gráfico que apresenta o número de sobreviventes de acordo com o gênero\n",
    "df_titanic_1.pivot_table(index='Sexo',  values=('Sobreviveu'), aggfunc=np.sum)[['Sobreviveu']].plot(\n",
    "    kind='bar', rot=0, label=('female','male'),\n",
    "    color=('tab:green','tab:red'), stacked=True,\n",
    "    title='Quantidade de Sobreviventes por sexo').set_xlabel('Sexo')\n",
    "plt.ylabel('Quantidade')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Podemos observar através do gráfico que apesar do número de passageiros do sexo masculino superar o número do sexo feminino, o maior índice de sobrevivência foram de passageiros do sexo feminino. Portanto, as mulheres tiveram mais chance de sobrevivência do que os homens.</h6>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Agora identificaremos o fator \"Classe\" sobre o índice de sobrevivência.</h6>\n",
    "<h6><strong>Os passageiros da primeira classe tiveram maiores chances de sobrevivência do que os passageiros das segunda e terceira classes?</strong></h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "e38b95e071b9001fbe6bc78e28a6577fe1a74346",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f822a2c4630>"
      ]
     },
     "execution_count": 15,
     "metadata": {
      "image/png": {
       "height": 276,
       "width": 388
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# gráfico para demonstração do número de sobreviventes e mortos por classe social\n",
    "df_titanic_1['Sobreviveu'].replace({0:'Não', 1:'Sim'}, inplace=True)\n",
    "sns.countplot(x='Sobreviveu', data=df_titanic_1, hue='Classe')\n",
    "plt.title(u'Número de sobreviventes e não sobreviventes por classe')\n",
    "plt.legend(title='Classe')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>O gráfico acima nos mostra que a quantidade de \"Não sobreviventes\" da terceira classe é muito maior do que das primeira e segunda classes. Podemos observar ainda que o número de \"Sobreviventes\" da primeira classe é maior do que os sobreviventes das segunda e terceira. Constatamos então que mais passageiros da primeira classe conseguiram sobreviver. Por outro lado, os passageiros da terceira classe foram os que mais tiveram dificuldade em sobreviver.</h6>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6><p>Passaremos agora a verificar a influência do fator idade sobre a sobrevivência dos passageiros.</p></br>\n",
    "<strong>Crianças tiveram uma taxa de sobrevivência maior que as demais faixas etárias?</strong>\n",
    "</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "76b9cd48d2eb47a443cbbede90b653368126cc73",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f822a16c4a8>"
      ]
     },
     "execution_count": 16,
     "metadata": {
      "image/png": {
       "height": 263,
       "width": 405
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Distribuição por idade (total de passageiros sobreviventes ou não)\n",
    "df_titanic_1['Idade'] = pd.cut(df_titanic_1.Idade, range(0, 81, 13), right=False)\n",
    "df_titanic_1.groupby(['Idade']).size().plot(kind='barh',stacked=True)\n",
    "plt.title('Idade por faixa etária')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6>Observamos que a maioria dos passageiros (sobrevivenes e não sobreviventes) se encontram entre a faixa de 26 a 39 anos de idade.</h6>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "add3524a5918e79b6f9a94519a69c358589b2753",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x7f8229a9f630>"
      ]
     },
     "execution_count": 29,
     "metadata": {
      "image/png": {
       "height": 276,
       "width": 388
      }
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# gráfico para a demonstração de sobreviventes e não sobreviventes em cada faixa etária do gráfico anterior\n",
    "df_titanic_1.groupby(['Idade']).size()\n",
    "sns.countplot(x='Idade', data=df_titanic_1, hue='Sobreviveu')\n",
    "plt.title(u'Número de sobreviventes e não sobreviventes por faixa etária')\n",
    "plt.legend(title='Sobreviveu')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h6> Com base no gráfico acima, verificamos que a faixa onde encontram-se as crianças de 0 a 13 anos é a única que apresenta o número de \"Sobreviventes\" (Sim) maior do que o de \"Não sobreviventes\" (Não). Portanto as crianças de 0 a 13 anos tiveram uma maior facilidade em sobreviver ao naufrágio do que passageiros nas demais faixas etárias.</h6>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "<h2>Conclusão</h2>\n",
    "<h6>Com base na análise de dados apresentada, concluiu-se os fatores como gênero, classe social e idade influenciaram diretamente nos índices de sobreviventes do Titanic. Os passageiros do sexo feminino, da primeira classe e na faixa etária de 0 a 13 anos foram os que obtiveram os maiores índices de sobrevivência ao naufrágio.</br></br>\n",
    "Limitações da análise:</br></br>\n",
    "O Titanic transportava um total de 1300 passageiros (além da tripulação a bordo). Na planilha analisada constavam apenas 891 registros, que representa 64,54% do valor total dos passageiros. Além disso, o fato de que muitos passageiros não tinham a idade registrada no nosso conjunto de dados, pode ter impactado nos resultados que envolviam essa variável de forma isolada ou combinada com outras variáveis. Em qualquer parte desta análise que envolvia, de algum modo, a variável idade foi utilizado uma cópia do dataframe original na qual foram substituídads as idads faltantes por uma média das idades.</h6>\n",
    "\n",
    "<h6>Fontes:</h6>\n",
    "<ul>\n",
    "    <li>https://paulovasconcellos.com.br/como-criar-seu-primeiro-projeto-de-data-science-parte-2-de-2-cb9a2fe05eff</li>\n",
    "    <li>https://medium.com/horadecodar/como-tratar-dados-nulos-no-dataset-4f0470b22d38</li>\n",
    "    <li>https://stackoverflow.com/questions/34193862/pandas-pivot-table-list-of-aggfunc</li>\n",
    "    <li>http://www.dadoscomsalelimao.com</li>\n",
    "</ul>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
   ],
   "source": [
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (system-wide)",
   "language": "python",
   "metadata": {
    "cocalc": {
     "description": "Python 3 programming language",
     "priority": 100,
     "url": "https://www.python.org/"
    }
   },
   "name": "python3",
   "resource_dir": "/ext/jupyter/kernels/python3"
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
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}