# Compatibility fix for Python 3.10+
import collections
import collections.abc
collections.Mapping = collections.abc.Mapping

import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import networkx as nx
from pyvis.network import Network
import tempfile
import squarify

# Scopus API configuration
SCOPUS_SEARCH_URL = "https://api.elsevier.com/content/search/scopus"
SCOPUS_ABSTRACT_URL = "https://api.elsevier.com/content/abstract/scopus_id/"

# Set page configuration
st.set_page_config(page_title="Scopus Bibliometric Analyzer", layout="wide")

def search_scopus(api_key, query, count=25, start=0):
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    params = {
        "query": query,
        "count": count,
        "start": start,
        "field": "prism:coverDate,author,affiliation,citedby-count,authkeywords,title"
    }
    try:
        # Add timeout to prevent hanging
        response = requests.get(SCOPUS_SEARCH_URL, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        # Check for rate limit headers
        remaining = response.headers.get('X-RateLimit-Remaining')
        if remaining and int(remaining) < 1:
            reset_time = response.headers.get('X-RateLimit-Reset')
            raise Exception(f"API rate limit reached. Resets at timestamp: {reset_time}")
            
        return response.json()
    except requests.Timeout:
        st.error("Request timed out. Please try again.")
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            st.error("API quota exceeded. Please wait before making more requests.")
        elif e.response.status_code == 401:
            st.error("Invalid API key or authentication error.")
        elif e.response.status_code == 403:
            st.error("Authorization error. Please check your API permissions.")
        raise e

def process_scopus_data(results):
    if not results:
        return pd.DataFrame()
        
    records = []
    for item in results:
        try:
            # Validate required fields
            if not isinstance(item, dict):
                continue
                
            # Safely get and validate the date
            cover_date = item.get('prism:coverDate', '')
            if not cover_date or not isinstance(cover_date, str):
                continue
                
            year = cover_date[:4]
            # Validate year is a reasonable number
            try:
                year_int = int(year)
                current_year = pd.Timestamp.now().year
                if not (1900 <= year_int <= current_year + 1):  # Allow 1 year in future for in-press
                    continue
            except (ValueError, TypeError):
                continue
            
            # Clean and process authors - handle None values safely
            authors = []
            for auth in item.get('author', []) or []:  # Use empty list if None
                if isinstance(auth, dict):
                    given_name = (auth.get('given-name') or '').strip()
                    surname = (auth.get('surname') or '').strip()
                    if given_name or surname:
                        authors.append(f"{given_name} {surname}".strip())
                elif isinstance(auth, str):
                    auth_str = auth.strip()
                    if auth_str:
                        authors.append(auth_str)
            
            # Clean and process institutions - handle None values safely
            institutions = []
            for aff in item.get('affiliation', []) or []:  # Use empty list if None
                if isinstance(aff, dict):
                    inst_name = (aff.get('affilname') or '').strip()
                    if inst_name:
                        institutions.append(inst_name)
                elif isinstance(aff, str):
                    inst_str = aff.strip()
                    if inst_str:
                        institutions.append(inst_str)
            
            # Clean and process keywords - handle None values safely
            keywords = []
            authkeywords = item.get('authkeywords')
            if isinstance(authkeywords, str):
                keywords = [kw.strip() for kw in authkeywords.split(' | ') 
                          if kw and kw.strip() and len(kw.strip()) > 1]
            
            # Safely get title
            title = (item.get('dc:title') or '').strip()
            
            record = {
                "Year": year,
                "Authors": "; ".join(filter(None, authors)),
                "Institutions": "; ".join(filter(None, set(institutions))),
                "Keywords": "; ".join(filter(None, keywords)),
                "Citations": int(item.get('citedby-count', 0)),
                "Title": title
            }
            records.append(record)
            
        except Exception as e:
            st.warning(f"Error processing record: {str(e)}")
            continue
            
    if not records:
        st.warning("No valid records found in the API response")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # Ensure Year is numeric and handle invalid years
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Year'])  # Remove rows with invalid years
    
    return df

def fetch_all_results(api_key, query, max_results=200):
    all_results = []
    start = 0
    count = 25
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        while start < max_results:
            status_text.text(f"Fetching results {start+1} to {min(start+count, max_results)}...")
            data = search_scopus(api_key, query, count=count, start=start)
            
            # Validate API response structure
            if not isinstance(data, dict) or 'search-results' not in data:
                raise ValueError("Invalid API response format")
                
            search_results = data.get('search-results', {})
            total_results = int(search_results.get('opensearch:totalResults', 0))
            
            if total_results == 0:
                status_text.warning("No results found for the given query.")
                break
                
            entries = search_results.get('entry', [])
            if not entries:
                break
                
            all_results.extend(entries)
            start += count
            
            # Update progress
            progress = min(start / max_results, 1.0)
            progress_bar.progress(progress)
            
            if start >= total_results:
                break
                
        status_text.text(f"Retrieved {len(all_results)} results.")
        return all_results
        
    except Exception as e:
        status_text.text("Error fetching results")
        raise e
    finally:
        progress_bar.empty()

def calculate_h_index(citations):
    citations = sorted(citations, reverse=True)
    for i, c in enumerate(citations):
        if c < i+1:
            return i
    return len(citations)

def main():
    st.title("🔍 Scopus Bibliometric Analyzer")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter Scopus API Key", type="password")
    
    # Topic and keyword inputs
    st.sidebar.header("Search Parameters")
    
    search_type = st.sidebar.radio(
        "Search Type",
        ["Simple Topic Search", "Advanced Keyword Search"],
        help="Choose between simple topic search or advanced keyword combinations"
    )
    
    if search_type == "Simple Topic Search":
        topic = st.sidebar.text_input(
            "Enter Research Topic", 
            value="bibliometric analysis",
            help="Enter the main topic you want to analyze"
        )
        search_query = f'TITLE-ABS-KEY("{topic}")'
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            title_kw = st.text_input(
                "Title Keywords",
                help="Keywords to search in title (separate with OR)"
            )
        with col2:
            abstract_kw = st.text_input(
                "Abstract Keywords",
                help="Keywords to search in abstract (separate with OR)"
            )
            
        keyword_kw = st.sidebar.text_input(
            "Author Keywords",
            help="Keywords to search in author keywords (separate with OR)"
        )
        
        # Build advanced query
        query_parts = []
        if title_kw:
            query_parts.append(f'TITLE("{title_kw}")')
        if abstract_kw:
            query_parts.append(f'ABS("{abstract_kw}")')
        if keyword_kw:
            query_parts.append(f'KEY("{keyword_kw}")')
            
        search_query = ' AND '.join(query_parts) if query_parts else 'TITLE-ABS-KEY("*")'
    
    # Year range for search
    current_year = pd.Timestamp.now().year
    search_year_range = st.sidebar.slider(
        "Select publication year range",
        min_value=1960,
        max_value=current_year,
        value=(current_year-5, current_year),
        help="Filter publications by year range"
    )
    
    # Add year filter to query
    search_query = (f'({search_query}) AND '
                   f'PUBYEAR > {search_year_range[0]-1} AND '
                   f'PUBYEAR < {search_year_range[1]+1}')
    
    # Show the constructed query with option to edit
    st.sidebar.subheader("Advanced Query")
    search_query = st.sidebar.text_area(
        "Search Query (you can modify)", 
        value=search_query,
        help="The search query in Scopus syntax. Edit if needed."
    )
    
    # Add search button
    search_col1, search_col2 = st.sidebar.columns([2, 1])
    with search_col1:
        search_button = st.button(
            "🔍 Search",
            help="Click to execute the search",
            type="primary",
            use_container_width=True
        )
    with search_col2:
        max_results = st.number_input(
            "Max Results",
            min_value=10,
            max_value=2000,
            value=200,
            step=10,
            help="Maximum number of results to retrieve"
        )
    
    # Show initial instructions if no search has been performed
    if not search_button:
        st.info("👈 Enter your search parameters in the sidebar and click 'Search' to begin analysis.")
        return
        
    if not api_key:
        st.error("Please enter your Scopus API key to proceed.")
        return
        
    # Execute search when button is clicked
    if api_key and search_query and search_button:
        try:
            with st.spinner("Fetching data from Scopus..."):
                results = fetch_all_results(api_key, search_query, max_results=max_results)
                df = process_scopus_data(results)
            
            if df.empty:
                st.warning("No results found for your search criteria.")
                return
                
            # Convert Year to numeric and handle invalid years
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df = df.dropna(subset=['Year'])
            df['Year'] = df['Year'].astype(int)
            
            if df.empty:
                st.warning("No valid results found after filtering.")
                return
                
            st.success(f"Retrieved {len(df)} records from Scopus!")
            
            # Show basic info
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Publications", len(df))
            col2.metric("Time Range", f"{int(df['Year'].min())} - {int(df['Year'].max())}")
            col3.metric("Total Citations", df['Citations'].sum())
            
            # Analysis parameters
            st.sidebar.header("Analysis Parameters")
            
            # Get year range from the data
            min_year = int(df['Year'].min())
            max_year = int(df['Year'].max())
            
            # Ensure min_year is different from max_year to avoid slider error
            if min_year == max_year:
                min_year = max_year - 1  # Go back one year if only one year present
                if min_year < 1900:  # Ensure we don't go below reasonable years
                    min_year = 1900
                    max_year = 1901
            
            analysis_year_range = st.sidebar.slider(
                "Select year range for analysis",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
            
            # Filter data
            filtered_df = df[
                (df['Year'] >= analysis_year_range[0]) & 
                (df['Year'] <= analysis_year_range[1])
            ]
            
            if filtered_df.empty:
                st.warning("No data available for the selected year range.")
                return
            
            # Analysis tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Publication Trends", 
                "Author Analysis", 
                "Institutional Analysis",
                "Keyword Analysis",
                "Network Analysis"
            ])

            with tab1:
                st.subheader("Publication Trends Over Time")
                # Fix duplicate index issue by using value_counts with sort=True
                annual_counts = filtered_df['Year'].value_counts(sort=True).sort_index()
                
                # Convert index to string to avoid any numeric type issues
                annual_counts.index = annual_counts.index.astype(str)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(data=pd.DataFrame({'Year': annual_counts.index, 'Count': annual_counts.values}),
                            x='Year', y='Count', marker='o')
                plt.title("Annual Publications")
                plt.xlabel("Year")
                plt.ylabel("Number of Publications")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with tab2:
                st.subheader("Author Analysis")
                
                # Process authors - avoid duplicate counting
                all_authors = []
                for authors in filtered_df['Authors'].dropna():
                    all_authors.extend([author.strip() for author in authors.split(';')])
                
                author_counts = pd.Series(all_authors).value_counts().head(20)
                
                # Top authors chart
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=author_counts.values, 
                           y=author_counts.index,
                           palette="viridis")
                plt.title("Top 20 Most Prolific Authors")
                plt.xlabel("Number of Publications")
                st.pyplot(fig)
                
                # Author citation metrics
                st.subheader("Author Citation Metrics")
                author_citations = (filtered_df
                    .assign(Authors=filtered_df['Authors'].str.split(';'))
                    .explode('Authors')
                    .groupby('Authors')['Citations']
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.bar_chart(author_citations)

            with tab3:
                st.subheader("Institutional Analysis")
                
                # Process institutions
                all_institutions = [inst.strip() for insts in filtered_df['Institutions'].dropna() 
                                   for inst in insts.split('; ')]
                institution_counts = Counter(all_institutions).most_common(20)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(y=[x[0] for x in institution_counts], 
                            x=[x[1] for x in institution_counts], 
                            palette="rocket")
                plt.title("Top 20 Contributing Institutions")
                plt.xlabel("Number of Publications")
                st.pyplot(fig)

            with tab4:
                st.subheader("Keyword Analysis")
                
                # Process keywords with better cleaning
                all_keywords = []
                for kws in filtered_df['Keywords'].dropna():
                    keywords = [kw.strip().lower() for kw in kws.split(';') 
                               if kw.strip() and len(kw.strip()) > 1]
                    all_keywords.extend(keywords)
                
                if not all_keywords:
                    st.warning("No keywords found in the dataset.")
                    return
                    
                # Keyword statistics
                keyword_counts = Counter(all_keywords)
                total_keywords = len(all_keywords)
                unique_keywords = len(keyword_counts)
                
                # Keyword metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Keywords", total_keywords)
                col2.metric("Unique Keywords", unique_keywords)
                col3.metric("Keywords per Paper", round(total_keywords / len(filtered_df), 2))
                
                # Keyword frequency analysis
                st.subheader("Top Keywords")
                
                num_keywords = st.slider("Number of top keywords to show", 10, 50, 20)
                
                # Create frequency table
                keyword_df = pd.DataFrame(keyword_counts.most_common(num_keywords),
                                        columns=['Keyword', 'Frequency'])
                keyword_df['Percentage'] = keyword_df['Frequency'] / len(filtered_df) * 100
                
                # Display keyword table
                st.dataframe(
                    keyword_df.style.format({
                        'Frequency': '{:,.0f}',
                        'Percentage': '{:.1f}%'
                    })
                )
                
                # Visualization options
                viz_type = st.radio(
                    "Visualization Type",
                    ["Bar Chart", "Word Cloud", "Tree Map"],
                    horizontal=True
                )
                
                if viz_type == "Bar Chart":
                    fig, ax = plt.subplots(figsize=(12, 6))
                    sns.barplot(data=keyword_df.head(20), 
                               y='Keyword', x='Frequency',
                               palette="viridis")
                    plt.title("Top 20 Keywords")
                    st.pyplot(fig)
                    
                elif viz_type == "Word Cloud":
                    wordcloud = WordCloud(
                        width=800, height=400,
                        background_color='white',
                        colormap='viridis',
                        max_words=100
                    ).generate_from_frequencies(keyword_counts)
                    
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                    
                else:  # Tree Map
                    fig, ax = plt.subplots(figsize=(12, 8))
                    squarify.plot(sizes=keyword_df['Frequency'].head(20),
                                 label=keyword_df['Keyword'].head(20),
                                 alpha=.8,
                                 color=sns.color_palette("viridis", 20))
                    plt.axis('off')
                    plt.title("Keyword Distribution (Tree Map)")
                    st.pyplot(fig)

            with tab5:
                st.subheader("Co-occurrence Network Analysis")
                
                # Create network graph
                keywords_list = filtered_df['Keywords'].dropna().apply(
                    lambda x: [kw.strip() for kw in x.split('; ')]).tolist()
                
                G = nx.Graph()
                for keywords in keywords_list:
                    for i in range(len(keywords)):
                        for j in range(i+1, len(keywords)):
                            if G.has_edge(keywords[i], keywords[j]):
                                G[keywords[i]][keywords[j]]['weight'] += 1
                            else:
                                G.add_edge(keywords[i], keywords[j], weight=1)
                
                # Visualize with pyvis
                nt = Network(height="600px", width="100%", bgcolor="white", font_color="black")
                nt.from_nx(G)
                
                # Save and display network
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
                    nt.save_graph(tmpfile.name)
                    st.components.v1.html(open(tmpfile.name, 'r', encoding='utf-8').read(), height=600)

        except requests.HTTPError as e:
            st.error(f"Scopus API Error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
