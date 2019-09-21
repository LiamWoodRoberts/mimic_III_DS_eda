import pandas as pd
import re
from utils import get_count_df

df = pd.read_csv("./discharge_summaries.csv")

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

def replace_line_breaks(report):
    return report.replace('\n','NEW_LINE')

def format_titles(report,titles):
    for title in titles:
        report = report.replace(title,title.upper())
        
    # Make Like Titles The Same
    report = report.replace('PHYSICAL EXAMINATION','PHYSICAL EXAM')
    report = report.replace('MEDICATIONS ON DISCHARGE','DISCHARGE MEDICATIONS')
    report = report.replace("DISCHARGE MEDICATION:","DISCHARGE MEDICATIONS:")
    report = report.replace("BRIEF SUMMARY OF HOSPITAL COURSE",'BRIEF HOSPITAL COURSE')
    report = report.replace("DIAGNOSES","DIAGNOSIS")
    report = report.replace("HOSPITAL COURSE BY SYSTEM","BRIEF HOSPITAL COURSE")
    return report

def get_title_block(report,title):
    block = re.search(f"{title}.*?(NEW_LINENEW_LINE[^(a-z)]*?:)",report)
    if block != None:
        block = block.group(0)
        block = re.sub(r'NEW_LINENEW_LINE[^(a-z)]*:','',block)
        block = block.replace('NEW_LINE','\n')
        return block
    else:
        return 'NOT FOUND'

def run(title):
    # Load Data
    df = pd.read_csv("./discharge_summaries.csv")
    
    # Format Titles
    title_counts = get_count_df(df["TEXT"])
    titles = title_counts["Title"].values[:50]

    series = df["TEXT"].apply(lambda x:format_titles(x,titles))

    # Get Sections
    series = series.apply(lambda x:replace_line_breaks(x))
    series = series.apply(lambda x:get_title_block(x,title.upper()))
    return series

if __name__ == "__main__":
    title = "DISCHARGE MEDICATIONS:"
    series = run(title)
    print(series[:10])