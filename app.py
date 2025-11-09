import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from collections import Counter
import tempfile
import os

# Page config
st.set_page_config(
    page_title="TOP-250 Movies Explorer",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for minimalistic design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    h1 {
        color: #1f1f1f;
        font-weight: 300;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 TOP-250 Movies Explorer")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload your TOP-250 movies file",
    type=['csv', 'json'],
    help="Upload a CSV or JSON file containing movie data"
)

# Load data function
@st.cache_data
def load_data(file, file_type):
    if file_type == 'csv':
        df = pd.read_csv(file)
    else:
        df = pd.read_json(file)
    return df

# Helper function to safely get list data
def get_list_value(series, idx):
    """Safely extract list values from series"""
    try:
        val = series.iloc[idx]
        if isinstance(val, list):
            return val
        elif isinstance(val, str):
            return eval(val) if val.startswith('[') else [val]
        return []
    except:
        return []

# Helper function to format directors
def get_directors_str(directors):
    """Convert director list to string"""
    if isinstance(directors, list):
        return ', '.join(directors)
    elif isinstance(directors, str):
        try:
            dir_list = eval(directors) if directors.startswith('[') else [directors]
            return ', '.join(dir_list)
        except:
            return directors
    return str(directors)

# Helper function to format genres
def get_genres_str(genres):
    """Convert genre list to string"""
    if isinstance(genres, list):
        return ', '.join(genres)
    elif isinstance(genres, str):
        try:
            genre_list = eval(genres) if genres.startswith('[') else [genres]
            return ', '.join(genre_list)
        except:
            return genres
    return str(genres)

# Process data
if uploaded_file is not None:
    try:
        # Determine file type
        file_type = 'csv' if uploaded_file.name.endswith('.csv') else 'json'
        
        # Load data
        df = load_data(uploaded_file, file_type)
        
        st.success(f"✅ Loaded {len(df)} movies successfully!")
        
        # Display raw data
        with st.expander("📊 View Raw Data Table"):
            st.dataframe(df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⏱️ Patience",
            "🎥 Binge Watching Steven Spielberg",
            "👤 This is about me",
            "🏭 Workhorse",
            "💰 Cash Horse"
        ])
        
        # TAB 1: PATIENCE - Longest movies
        with tab1:
            st.header("⏱️ Patience: Longest Movies")
            st.markdown("*Movies that require the most patience to watch*")
            
            # Slider for interactivity
            top_n = st.slider(
                "Number of movies to display",
                min_value=5,
                max_value=20,
                value=10,
                key="patience_slider"
            )
            
            # Get longest movies
            longest = df.nlargest(top_n, 'duration').copy()
            
            # Create display dataframe
            display_df = longest[['title', 'duration', 'year', 'ratingValue']].copy()
            display_df['directors'] = longest['directorList'].apply(get_directors_str)
            display_df['genres'] = longest['genreList'].apply(get_genres_str)
            
            # Reorder columns
            display_df = display_df[['title', 'duration', 'year', 'ratingValue', 'directors', 'genres']]
            display_df.columns = ['Title', 'Duration (min)', 'Year', 'Rating', 'Director(s)', 'Genres']
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Bar chart
            fig = px.bar(
                longest,
                x='duration',
                y='title',
                orientation='h',
                title=f"Top {top_n} Longest Movies",
                labels={'duration': "Runtime (minutes)", 'title': ''},
                color='duration',
                color_continuous_scale="Blues"
            )
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=400 + (top_n * 20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Longest Movie", f"{longest.iloc[0]['duration']:.0f} min")
            with col2:
                st.metric("Average Duration", f"{longest['duration'].mean():.0f} min")
            with col3:
                st.metric("Total Watch Time", f"{longest['duration'].sum()/60:.1f} hours")
        
        # TAB 2: BINGE WATCHING STEVEN SPIELBERG
        with tab2:
            st.header("🎥 Binge Watching Steven Spielberg")
            st.markdown("*All movies directed by Steven Spielberg*")
            
            # Find Spielberg movies
            spielberg_mask = df['directorList'].apply(
                lambda x: any('Spielberg' in director for director in (x if isinstance(x, list) else []))
            )
            spielberg = df[spielberg_mask].copy()
            
            if not spielberg.empty:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Spielberg Movies", len(spielberg))
                with col2:
                    st.metric("Average Rating", f"{spielberg['ratingValue'].mean():.2f}")
                with col3:
                    st.metric("Total Runtime", f"{spielberg['duration'].sum()/60:.1f} hours")
                
                # Create display dataframe
                display_df = spielberg[['title', 'year', 'ratingValue', 'duration']].copy()
                display_df['genres'] = spielberg['genreList'].apply(get_genres_str)
                display_df = display_df.sort_values('year')
                display_df.columns = ['Title', 'Year', 'Rating', 'Duration (min)', 'Genres']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Timeline of Spielberg movies
                year_counts = spielberg.groupby('year').size().reset_index(name='count')
                fig = px.scatter(
                    spielberg,
                    x='year',
                    y='ratingValue',
                    size='duration',
                    hover_data=['title'],
                    title="Spielberg Movies: Rating vs Year (size = duration)",
                    labels={'year': 'Year', 'ratingValue': 'Rating'},
                    color='ratingValue',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No Steven Spielberg movies found in this dataset")
        
        # TAB 3: THIS IS ABOUT ME - Personal stats/recommendations
        with tab3:
            st.header("👤 This is about me")
            st.markdown("*Personalized movie statistics and insights*")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_year = df['year'].mean()
                st.metric("Average Year", f"{avg_year:.0f}")
            
            with col2:
                avg_rating = df['ratingValue'].mean()
                st.metric("Average Rating", f"{avg_rating:.2f}")
            
            with col3:
                total_runtime = df['duration'].sum()
                st.metric("Total Watch Time", f"{total_runtime/60:.0f} hrs")
            
            with col4:
                total_movies = len(df)
                st.metric("Total Movies", total_movies)
            
            st.markdown("---")
            
            # Genre analysis
            st.subheader("🎨 Your Movie Collection by Decade")
            
            # Create decade bins
            df_copy = df.copy()
            df_copy['decade'] = (df_copy['year'] // 10) * 10
            decade_stats = df_copy.groupby('decade').agg({
                'title': 'count',
                'ratingValue': 'mean',
                'duration': 'sum'
            }).reset_index()
            decade_stats.columns = ['Decade', 'Movies', 'Avg Rating', 'Total Duration']
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=decade_stats['Decade'],
                y=decade_stats['Movies'],
                name='Number of Movies',
                marker_color='steelblue'
            ))
            fig.update_layout(
                title="Movies by Decade",
                xaxis_title="Decade",
                yaxis_title="Number of Movies"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top genres
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎭 Top Genres")
                all_genres = []
                for genres in df['genreList']:
                    if isinstance(genres, list):
                        all_genres.extend(genres)
                
                genre_counts = Counter(all_genres).most_common(10)
                genre_df = pd.DataFrame(genre_counts, columns=['Genre', 'Count'])
                
                fig = px.bar(
                    genre_df,
                    x='Count',
                    y='Genre',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Teal'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🌍 Top Countries")
                all_countries = []
                for countries in df['countryList']:
                    if isinstance(countries, list):
                        all_countries.extend(countries)
                
                country_counts = Counter(all_countries).most_common(10)
                country_df = pd.DataFrame(country_counts, columns=['Country', 'Count'])
                
                fig = px.bar(
                    country_df,
                    x='Count',
                    y='Country',
                    orientation='h',
                    color='Count',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # TAB 4: WORKHORSE - Most prolific directors/actors
        with tab4:
            st.header("🏭 Workhorse: Most Prolific Contributors")
            st.markdown("*Directors and actors with the most movies*")
            
            top_n_work = st.slider(
                "Number of contributors to display",
                min_value=5,
                max_value=20,
                value=10,
                key="workhorse_slider"
            )
            
            col1, col2 = st.columns(2)
            
            # Most prolific directors
            with col1:
                st.subheader("🎬 Top Directors")
                all_directors = []
                for directors in df['directorList']:
                    if isinstance(directors, list):
                        all_directors.extend(directors)
                
                director_counts = Counter(all_directors).most_common(top_n_work)
                director_df = pd.DataFrame(
                    director_counts,
                    columns=['Director', 'Movies']
                )
                
                fig = px.bar(
                    director_df,
                    x='Movies',
                    y='Director',
                    orientation='h',
                    title=f"Top {top_n_work} Directors",
                    color='Movies',
                    color_continuous_scale="Viridis"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(director_df, use_container_width=True, hide_index=True)
            
            # Most prolific actors
            with col2:
                st.subheader("⭐ Top Actors")
                all_actors = []
                for actors in df['castList']:
                    if isinstance(actors, list):
                        all_actors.extend(actors)
                
                actor_counts = Counter(all_actors).most_common(top_n_work)
                actor_df = pd.DataFrame(
                    actor_counts,
                    columns=['Actor', 'Movies']
                )
                
                fig = px.bar(
                    actor_df,
                    x='Movies',
                    y='Actor',
                    orientation='h',
                    title=f"Top {top_n_work} Actors",
                    color='Movies',
                    color_continuous_scale="Plasma"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(actor_df, use_container_width=True, hide_index=True)
        
        # TAB 5: CASH HORSE - Highest grossing/rated movies
        with tab5:
            st.header("💰 Cash Horse: Top Performing Movies")
            st.markdown("*Movies with the highest ratings and box office performance*")
            
            top_n_cash = st.slider(
                "Number of movies to display",
                min_value=5,
                max_value=20,
                value=10,
                key="cash_slider"
            )
            
            # Two sections: by rating and by gross
            option = st.radio(
                "Sort by:",
                ["Rating", "Box Office Gross"],
                horizontal=True
            )
            
            if option == "Rating":
                top_movies = df.nlargest(top_n_cash, 'ratingValue').copy()
                sort_col = 'ratingValue'
                sort_label = 'Rating'
            else:
                # Filter out movies with gross data
                df_with_gross = df[df['gross'] > 0].copy()
                top_movies = df_with_gross.nlargest(top_n_cash, 'gross').copy()
                sort_col = 'gross'
                sort_label = 'Gross ($)'
            
            # Create display dataframe
            display_df = top_movies[['title', 'year', 'ratingValue', 'gross', 'duration']].copy()
            display_df['directors'] = top_movies['directorList'].apply(get_directors_str)
            display_df['genres'] = top_movies['genreList'].apply(get_genres_str)
            display_df['gross_millions'] = display_df['gross'] / 1_000_000
            
            display_df = display_df[[
                'title', 'year', 'ratingValue', 'gross_millions', 
                'duration', 'directors', 'genres'
            ]]
            display_df.columns = [
                'Title', 'Year', 'Rating', 'Gross ($M)', 
                'Duration (min)', 'Director(s)', 'Genres'
            ]
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Visualization
            if option == "Rating":
                fig = px.scatter(
                    top_movies,
                    x='year',
                    y='ratingValue',
                    size='ratingCount',
                    hover_data=['title'],
                    title=f"Top {top_n_cash} Rated Movies (size = rating count)",
                    color='ratingValue',
                    color_continuous_scale="YlOrRd",
                    labels={'year': 'Year', 'ratingValue': 'Rating'}
                )
            else:
                top_movies['gross_millions'] = top_movies['gross'] / 1_000_000
                fig = px.bar(
                    top_movies,
                    x='gross_millions',
                    y='title',
                    orientation='h',
                    title=f"Top {top_n_cash} Box Office Movies",
                    labels={'gross_millions': 'Gross Revenue ($M)', 'title': ''},
                    color='gross_millions',
                    color_continuous_scale="Greens"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Rating", f"{top_movies['ratingValue'].mean():.2f}")
            with col2:
                if top_movies['gross'].sum() > 0:
                    st.metric("Total Gross", f"${top_movies['gross'].sum()/1e9:.2f}B")
                else:
                    st.metric("Total Gross", "N/A")
            with col3:
                st.metric("Average Duration", f"{top_movies['duration'].mean():.0f} min")
        
        # WOW FEATURES SECTION
        st.markdown("---")
        st.header("✨ WOW Features")
        
        wow_tab1, wow_tab2, wow_tab3, wow_tab4 = st.tabs([
            "📅 Timeline View",
            "☁️ Genre Word Cloud",
            "🕸️ Top Collaborations",
            "💎 Hidden Gems"
        ])
        
        # WOW 1: Timeline View
        with wow_tab1:
            st.subheader("📅 Movies Released Per Year")
            
            year_counts = df['year'].value_counts().sort_index()
            
            # Create figure with dual axis
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=year_counts.index,
                y=year_counts.values,
                name='Number of Movies',
                marker_color='steelblue'
            ))
            
            fig.update_layout(
                title="Number of Movies per Year",
                xaxis_title="Year",
                yaxis_title="Number of Movies",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Most Productive Year", year_counts.idxmax())
            with col2:
                st.metric("Movies That Year", year_counts.max())
            with col3:
                st.metric("Year Range", f"{year_counts.index.min()} - {year_counts.index.max()}")
            with col4:
                st.metric("Decades Covered", len(df['year'].apply(lambda x: x // 10).unique()))
            
            # Average rating by year
            st.subheader("📈 Average Rating Trend Over Time")
            year_rating = df.groupby('year')['ratingValue'].mean().reset_index()
            
            fig2 = px.line(
                year_rating,
                x='year',
                y='ratingValue',
                title="Average Movie Rating by Year",
                markers=True,
                labels={'year': 'Year', 'ratingValue': 'Average Rating'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # WOW 2: Genre Word Cloud
        with wow_tab2:
            st.subheader("☁️ Genre Word Cloud")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Extract all genres
                all_genres = []
                for genres in df['genreList']:
                    if isinstance(genres, list):
                        all_genres.extend(genres)
                
                genre_text = ' '.join(all_genres)
                
                # Generate word cloud
                wordcloud = WordCloud(
                    width=1200,
                    height=600,
                    background_color='white',
                    colormap='viridis',
                    relative_scaling=0.5,
                    min_font_size=10
                ).generate(genre_text)
                
                # Display
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                st.markdown("### 📊 Genre Stats")
                genre_counts = Counter(all_genres)
                total_genres = sum(genre_counts.values())
                
                st.metric("Unique Genres", len(genre_counts))
                st.metric("Total Genre Tags", total_genres)
                st.metric("Avg Genres/Movie", f"{total_genres/len(df):.1f}")
                
                st.markdown("### Top 5 Genres")
                for genre, count in genre_counts.most_common(5):
                    pct = (count / len(df)) * 100
                    st.write(f"**{genre}**: {count} ({pct:.1f}%)")
        
        # WOW 3: Top Collaborations Network
        with wow_tab3:
            st.subheader("🕸️ Top 10 Actor-Director Collaborations")
            
            st.info("This network shows the most frequent actor-director partnerships in the TOP-250")
            
            # Build collaborations
            collaborations = []
            
            for _, row in df.iterrows():
                directors = row['directorList'] if isinstance(row['directorList'], list) else []
                actors = row['castList'] if isinstance(row['castList'], list) else []
                
                for director in directors:
                    for actor in actors:
                        collaborations.append((director, actor))
            
            # Count collaborations
            collab_counts = Counter(collaborations)
            top_collabs = collab_counts.most_common(10)
            
            # Create network graph
            G = nx.Graph()
            
            for (director, actor), count in top_collabs:
                G.add_edge(
                    director, 
                    actor, 
                    weight=count, 
                    title=f"{count} collaborations"
                )
            
            # Separate directors and actors for coloring
            directors_in_graph = set([d for d, a in top_collabs])
            actors_in_graph = set([a for d, a in top_collabs])
            
            # Create PyVis network
            net = Network(
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="#000000"
            )
            
            # Add nodes with colors
            for node in G.nodes():
                if node in directors_in_graph:
                    net.add_node(
                        node,
                        label=node,
                        color='#ff6b6b',
                        title=f"Director: {node}"
                    )
                else:
                    net.add_node(
                        node,
                        label=node,
                        color='#4ecdc4',
                        title=f"Actor: {node}"
                    )
            
            # Add edges
            for edge in G.edges(data=True):
                net.add_edge(
                    edge[0],
                    edge[1],
                    value=edge[2]['weight'],
                    title=edge[2]['title']
                )
            
            # Customize physics
            net.set_options("""
            {
              "nodes": {
                "font": {"size": 14},
                "borderWidth": 2,
                "size": 25
              },
              "edges": {
                "color": {"inherit": true},
                "smooth": false,
                "width": 2
              },
              "physics": {
                "enabled": true,
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "springLength": 250,
                  "springConstant": 0.04
                },
                "minVelocity": 0.75
              }
            }
            """)
            
            # Save and display
            tmp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.html',
                mode='w'
            )
            net.save_graph(tmp_file.name)
            tmp_file.close()
            
            with open(tmp_file.name, 'r', encoding='utf-8') as f:
                html = f.read()
                components.html(html, height=600)
            
            os.unlink(tmp_file.name)
            
            # Legend
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("🔴 **Red nodes** = Directors")
            with col2:
                st.markdown("🔵 **Cyan nodes** = Actors")
            
            # Show table
            st.markdown("---")
            st.markdown("**Top 10 Collaborations:**")
            collab_df = pd.DataFrame(
                [(d, a, c) for (d, a), c in top_collabs],
                columns=['Director', 'Actor', 'Collaborations']
            )
            st.dataframe(collab_df, use_container_width=True, hide_index=True)
        
        # WOW 4: Hidden Gems (Bonus!)
        with wow_tab4:
            st.subheader("💎 Hidden Gems: Underrated Movies")
            st.markdown("*Movies with high ratings but fewer rating counts*")
            
            # Calculate "hidden gem" score
            df_gems = df.copy()
            df_gems['gem_score'] = (
                df_gems['ratingValue'] / df_gems['ratingCount'].quantile(0.5)
            )
            
            # Get movies with high rating but low rating count
            hidden_gems = df_gems[
                (df_gems['ratingValue'] >= df_gems['ratingValue'].quantile(0.75)) &
                (df_gems['ratingCount'] <= df_gems['ratingCount'].quantile(0.25))
            ].nlargest(15, 'ratingValue')
            
            if not hidden_gems.empty:
                # Display
                display_df = hidden_gems[['title', 'year', 'ratingValue', 'ratingCount']].copy()
                display_df['directors'] = hidden_gems['directorList'].apply(get_directors_str)
                display_df['genres'] = hidden_gems['genreList'].apply(get_genres_str)
                display_df.columns = [
                    'Title', 'Year', 'Rating', 'Rating Count', 
                    'Director(s)', 'Genres'
                ]
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Scatter plot
                fig = px.scatter(
                    hidden_gems,
                    x='ratingCount',
                    y='ratingValue',
                    size='duration',
                    hover_data=['title', 'year'],
                    title="Hidden Gems: High Rating vs Low Rating Count",
                    labels={
                        'ratingCount': 'Number of Ratings',
                        'ratingValue': 'Rating'
                    },
                    color='year',
                    color_continuous_scale='Sunset'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hidden gems found with current criteria")
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.exception(e)
        st.info("Please ensure your file matches the expected schema")
else:
    st.info("👆 Please upload a CSV or JSON file to get started")
    
    # Sample data structure
    with st.expander("ℹ️ Expected Data Schema"):
        st.markdown("""
        Your file should contain the following columns:
        - **title**: Movie title (string)
        - **year**: Release year (number)
        - **ratingValue**: IMDb rating (number)
        - **ratingCount**: Number of ratings (number)
        - **duration**: Runtime in minutes (number)
        - **directorList**: List of director names (array)
        - **castList**: List of actor names (array)
        - **genreList**: List of genres (array)
        - **countryList**: List of countries (array)
        - **gross**: Box office gross (number)
        - **budget**: Production budget (number or string)
        - **description**: Movie description (string)
        - **characterList**: List of character names (array)
        - **url**: IMDb URL (string)
        """)
