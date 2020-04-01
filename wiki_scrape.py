from requests_html import HTMLSession
import numpy as np
import seaborn as sns
from pandas import DataFrame, read_csv
from dateutil.parser import parse, ParserError
import matplotlib.pyplot as plt
import re
wiki_url = 'https://he.wikipedia.org/wiki/' \
           '%D7%94%D7%AA%D7%A4%D7%A8%D7%A6%D7%95%D7%AA_' \
           '%D7%A0%D7%92%D7%99%D7%A3_%D7%94%D7%A7%D7%95%D7%A8%D7%95' \
           '%D7%A0%D7%94_%D7%91%D7%99%D7%A9%D7%A8%D7%90%D7%9C'
covid19data_url = 'https://raw.githubusercontent.com/idandrd/israel-covid19-data/master/IsraelCOVID19.csv'
generic = 'https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_{}'

'(\d{0,3},)?(\d{3},)?'
def get_html_table(element):
    for elem in element.find('table'):
        if 'class' not in elem.attrs:
            return elem

def html_to_python(element):

    def get_number(text):
        end = text.find('(')
        if end == -1:
            end = None
        text = text[:end].replace(',', '')
        number = re.match('\d+', text).group()
        return int(number)

    python_list = []
    rows = element.find('tr')[:-1]
    for elem in rows:
        if 'style' not in elem.attrs:
            row = elem.text.split('\n')
            try:
                python_list.append((parse(row[0]),get_number(row[1])))
            except ParserError:
                pass
    return python_list


def from_wikipedia(country):
    url = 'https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_{}'.format(country.title())
    session = HTMLSession()
    wiki = session.get(url)
    table = get_html_table(wiki.html)
    data = html_to_python(table)
    df = DataFrame(data=data, columns=['Date', 'Total Cases'])
    df['New Cases'] = (df['Total Cases'] - df['Total Cases'].shift()).fillna(1).astype('int64')
    return df

def from_covid19data():
    df = read_csv(covid19data_url)
    df['Date'] = df['Date'].apply(parse)
    return df

def chi_squared(x, y, sigma):
    var = 1/sigma**2
    s = np.sum(var)
    sx = np.sum(x * var)
    sy = np.sum(y * var)
    sxx = np.sum(x**2 * var)
    sxy = np.sum(x * y * var)
    delta = s * sxx - sx**2
    a = (sxx*sy - sx*sxy) / delta
    b = (s*sxy - sx*sy) / delta
    da = (sxx/delta)**(1/2)
    db = (s/delta)**(1/2)
    chi2 = np.sum((y - a- b*x)**2/var)
    return a, da, b, db, chi2

    # plt.scatter(x=x, y=y)
    # print(x[::2])
    # x_ticks = np.arange(start=1,stop=x.size)
    # print(x_ticks)
    # plt.xticks(x[::2],labels=df['day'][::2],rotation=60)
if __name__ == '__main__':
    df = from_covid19data()
    df = from_wikipedia('Israel')

    # print(df)
    df = df[:]
    print(df)
    df['Ln Total Cases'] = np.log(df['Total Cases'])
    x, y = df.index, df['Ln Total Cases']
    sigma = 1/(df['Total Cases'])**(1/2)
    # sigma = np.ones(df['Ln Total Cases'].size)
    a , da, b, db, chi2  = chi_squared(x,y,sigma)
    sns.scatterplot(x=x, y=y)
    degrees_of_freedom = df['Ln Total Cases'].size - 2
    # sns.lineplot(x, b*x + a)
    # sns.lineplot(x,np.exp(a+b*x))
    print(chi2/degrees_of_freedom)
    print(np.exp(b))
    print(np.exp(a))
    print(np.log(10)/b)
    print(np.log(2)/b)


    plt.show()
