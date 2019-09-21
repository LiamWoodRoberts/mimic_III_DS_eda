def sort_by_frequency(df,col):
    '''groups a column and returns a sorted list of most frequent values'''
    sorted_categories = df.groupby(col).count()['TEXT'].sort_values(ascending=False).index
    return sorted_categories

def get_titles(text):
    '''gets unique section titles in a list of report texts'''
    titles = []
    for reports in text:
        titles+=list(set(re.findall(r"\n\n.*:",reports)))
    return [title.strip().rstrip() for title in titles]
    
def get_counts(titles):
    '''gets counts of section titles'''
    counts = {}
    for title in titles:
        if title in counts:
            counts[title] += 1
        else:
            counts[title] = 1
    return counts

def format_counts(count_dict):
    '''formats section titles counts as dataframe'''
    counts = pd.DataFrame()
    counts['Title'] = count_dict.keys()
    counts['Counts'] = count_dict.values()
    counts = counts.sort_values('Counts',ascending=False)
    return counts

def get_count_df(text):
    '''accepts a list of reports (strings) and returns sorted list of section titles'''
    titles = get_titles(text)
    count_dict = get_counts(titles)
    count_df = format_counts(count_dict)
    count_df['%'] = count_df['Counts']/len(text)*100
    return count_df